"""End-to-end proof that an external non-GAN method plugs in through explicit definitions.

Configuration, training, checkpoint ranking, resume, inference-only reconstruction and
the tracked train/infer applications all run the fixture without framework changes.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml
from torch.utils.data import DataLoader

from tests.external_method.tiny_reconstruction import (
    TINY_CONV,
    TINY_RESIDUAL,
    TinyReconstruction,
    TinyRuntime,
)
from tests.image_helpers import write_rgb_image
from tests.manifest_helpers import make_manifest_record, manifest_metadata
from virtual_staining.applications.infer import infer
from virtual_staining.applications.infer_images import infer_images
from virtual_staining.applications.train import train
from virtual_staining.checkpoint_contract import CheckpointCompatibilityError
from virtual_staining.config.run import RunConfig
from virtual_staining.data.layout import DatasetLayout
from virtual_staining.data.manifest import DatasetManifest
from virtual_staining.definitions import DefinitionNotAvailableError, Definitions
from virtual_staining.experiment.run_layout import RunLayout, ensure_run_directories
from virtual_staining.inference.runner import load_inference_generator
from virtual_staining.methods.builtin import builtin_definitions
from virtual_staining.training.trainer import Trainer

_CPU = torch.device("cpu")
_GAN_KEYS = frozenset(
    {
        "generator",
        "discriminator",
        "lr_g",
        "lr_d",
        "beta1",
        "beta2",
        "losses",
        "scheduler",
        "replay_buffer_size",
    }
)


def _definitions() -> tuple[Definitions, TinyReconstruction]:
    tiny = TinyReconstruction()
    return builtin_definitions().extend(methods=[tiny], components=[TINY_CONV, TINY_RESIDUAL]), tiny


def _mapping(tmp_path: Path, **options: Any) -> dict[str, Any]:
    return {
        "dataset_root": str(tmp_path / "dataset"),
        "results_path": str(tmp_path / "results"),
        "run_name": "external_example",
        "image_size": [32, 32],
        "method": {
            "name": "tiny_reconstruction",
            "options": {"architecture": "tiny_conv", "learning_rate": 0.001, **options},
        },
        "data": {"pairing": "paired", "group_validation": "unavailable"},
        "model": {"inputs": ["source"], "target": "target"},
        "training": {
            "batch_size": 2,
            "epochs": 2,
            "seed": 123,
            "num_workers": 0,
            "validate_rate": 1,
            "checkpoint_rate": 1,
        },
        "inference": {"checkpoint_policy": "best", "checkpoint_metric": "val_abs_bias"},
    }


def _resolve(tmp_path: Path, definitions: Definitions, **options: Any) -> RunConfig:
    return RunConfig.from_mapping(_mapping(tmp_path, **options), definitions)


def _keys(value: Any) -> Iterator[str]:
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _keys(item)


# --- configuration ---------------------------------------------------------------------


def test_explicit_registration_resolves_yaml_deterministically(tmp_path: Path) -> None:
    definitions, tiny = _definitions()
    path = tmp_path / "external.yaml"
    path.write_text(yaml.safe_dump(_mapping(tmp_path, width=4)), encoding="utf-8")

    config = RunConfig.from_yaml(path, definitions)
    resolved = config.to_dict()

    assert config.method.definition is tiny
    assert config.method.options.network.definition is TINY_CONV
    assert resolved["method"] == {
        "name": "tiny_reconstruction",
        "options": {"architecture": "tiny_conv", "width": 4, "learning_rate": 0.001},
    }
    # The resolved spelling is stable and resolves back to itself.
    assert RunConfig.from_mapping(resolved, definitions).to_dict() == resolved
    assert json.dumps(resolved, sort_keys=True) == json.dumps(
        RunConfig.from_yaml(path, definitions).to_dict(), sort_keys=True
    )
    # Built-in resolution is unaffected by the extra registrations.
    example = Path(__file__).parents[2] / "config" / "runs" / "example.yaml"
    assert RunConfig.from_yaml(example, definitions).to_dict() == (
        RunConfig.from_yaml(example).to_dict()
    )


def test_external_config_carries_no_gan_settings(tmp_path: Path) -> None:
    definitions, _ = _definitions()
    resolved = _resolve(tmp_path, definitions).to_dict()

    assert resolved["model"] == {"inputs": ["source"], "target": "target"}
    assert set(resolved["method"]) == {"name", "options"}
    assert set(resolved["training"]) == {
        "batch_size",
        "epochs",
        "seed",
        "num_workers",
        "validate_rate",
        "checkpoint_rate",
        "checkpoint_top_k",
        "log_rate",
        "augmentation",
    }
    assert not _GAN_KEYS & set(_keys(resolved))


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("model", "generator", {"architecture": "concat_unet"}),
        ("model", "discriminator", {"ndf": 4}),
        ("training", "lr_g", 0.1),
        ("training", "losses", {"generator": []}),
        ("training", "scheduler", {"name": "none"}),
        ("method", "replay_buffer_size", 1),
    ],
)
def test_external_method_rejects_builtin_gan_keys(
    tmp_path: Path, section: str, key: str, value: Any
) -> None:
    definitions, _ = _definitions()
    data = _mapping(tmp_path)
    data[section][key] = value

    with pytest.raises(ValueError, match=f"Unknown key.*in {section}: {key}"):
        RunConfig.from_mapping(data, definitions)


def test_unregistered_method_fails_without_import(tmp_path: Path) -> None:
    with pytest.raises(DefinitionNotAvailableError, match="not a registered method definition"):
        RunConfig.from_mapping(_mapping(tmp_path))
    data = _mapping(tmp_path)
    data["method"]["name"] = "tests.external_method.tiny_reconstruction:TinyReconstruction"
    with pytest.raises(DefinitionNotAvailableError, match="not a registered method definition"):
        RunConfig.from_mapping(data, _definitions()[0])


def test_unknown_component_fails(tmp_path: Path) -> None:
    definitions, _ = _definitions()
    with pytest.raises(DefinitionNotAvailableError, match="'tiny_unet' is not a registered comp"):
        _resolve(tmp_path, definitions, architecture="tiny_unet")
    with pytest.raises(ValueError, match="architecture must be one of"):
        _resolve(tmp_path, definitions, architecture="concat_unet")


def test_duplicate_registration_is_rejected() -> None:
    definitions, tiny = _definitions()

    with pytest.raises(ValueError, match="Duplicate method definition 'tiny_reconstruction'"):
        definitions.extend(methods=[TinyReconstruction()])
    with pytest.raises(ValueError, match="Duplicate method definition 'pix2pix'"):
        definitions.extend(methods=[builtin_definitions().methods["pix2pix"]])
    with pytest.raises(ValueError, match="Duplicate component definition 'tiny_conv'"):
        definitions.extend(components=[TINY_CONV])
    with pytest.raises(ValueError, match="Duplicate component definition 'resnet'"):
        Definitions().extend(components=[builtin_definitions().components["resnet"]] * 2)
    # The original registry is immutable and unchanged by failed extensions.
    assert definitions.methods["tiny_reconstruction"] is tiny
    with pytest.raises(TypeError):
        definitions.methods["other"] = tiny  # type: ignore[index]


@pytest.mark.parametrize(
    ("options", "error", "message"),
    [
        ({"dropout": 0.1}, ValueError, r"Unknown key.*method\.options: dropout"),
        ({"learning_rate": -1.0}, ValueError, "learning_rate must be a finite number > 0"),
        ({"learning_rate": "fast"}, TypeError, "learning_rate must be a number"),
        ({"width": 0}, ValueError, r"method\.options\.width must be an integer >= 1"),
        ({"plateau_monitor": "val_ssim"}, ValueError, "not a checkpoint metric"),
    ],
)
def test_unknown_or_invalid_method_options_fail(
    tmp_path: Path, options: dict[str, Any], error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        _resolve(tmp_path, _definitions()[0], **options)


def test_custom_metric_is_a_valid_monitor_with_the_methods_direction(tmp_path: Path) -> None:
    definitions, _ = _definitions()
    data = _mapping(tmp_path)
    data["training"]["early_stopping"] = {"patience": 3}
    config = RunConfig.from_mapping(data, definitions)

    assert config.training is not None and config.training.early_stopping is not None
    assert config.training.early_stopping.monitor == "val_abs_bias"
    assert config.training.early_stopping.mode == "min"
    assert config.inference is not None and config.inference.checkpoint_metric == "val_abs_bias"
    data["inference"]["checkpoint_metric"] = "val_ssim"
    with pytest.raises(ValueError, match="'val_ssim' is not a checkpoint metric"):
        RunConfig.from_mapping(data, definitions)


# --- training, ranking, resume ------------------------------------------------------------


def _samples(count: int = 4) -> list[dict[str, Any]]:
    generator = torch.Generator().manual_seed(0)
    samples = []
    for _ in range(count):
        source = torch.rand(3, 32, 32, generator=generator) * 2 - 1
        samples.append({"inputs": {"source": source}, "target": -source, "masks": {}})
    return samples


def _trainer(config: RunConfig, runtime: TinyRuntime) -> Trainer:
    assert config.training is not None
    layout = RunLayout.from_project(config.project)
    ensure_run_directories(layout)
    loader = DataLoader(_samples(), batch_size=config.training.batch_size)  # pyright: ignore[reportArgumentType]
    return Trainer(config.training, layout, runtime, loader, loader, _CPU, config_hash="sha256:x")


def _train(config: RunConfig) -> tuple[TinyRuntime, RunLayout]:
    torch.manual_seed(0)
    runtime = config.method.definition.build_training_runtime(config, _CPU, seed=0)
    assert isinstance(runtime, TinyRuntime)
    _trainer(config, runtime).train(seed=0)
    return runtime, RunLayout.from_project(config.project)


def test_trainer_runs_the_external_method_with_its_own_objective(tmp_path: Path) -> None:
    definitions, tiny = _definitions()
    data = _mapping(tmp_path, plateau_monitor="val_abs_bias")
    data["training"]["early_stopping"] = {"patience": 5}
    config = RunConfig.from_mapping(data, definitions)

    runtime, layout = _train(config)

    modules = [value for value in vars(runtime).values() if isinstance(value, torch.nn.Module)]
    optimizers = [
        value for value in vars(runtime).values() if isinstance(value, torch.optim.Optimizer)
    ]
    assert [type(module).__name__ for module in modules] == ["_TinyNetwork", "L1Loss"]
    assert len(optimizers) == 1
    assert tiny.built == {"network": 1, "optimizer": 1, "objective": 1, "scheduler": 1}
    with layout.epochs_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    columns = ["epoch", "loss_recon_train", "loss_recon_val", "val_abs_bias"]
    assert reader.fieldnames == columns
    assert [row["epoch"] for row in rows] == ["0", "1"]
    assert all(row[name] for row in rows for name in columns)
    assert sorted(path.name for path in layout.checkpoints_dir.iterdir()) == [
        "best.json",
        "ep000.pth",
        "ep001.pth",
    ]
    best = json.loads((layout.checkpoints_dir / "best.json").read_text(encoding="utf-8"))
    assert best["schema_version"] == 2
    assert set(best["metrics"]) == {"val_abs_bias", "loss_recon_val"}
    assert best["metrics"]["val_abs_bias"]["mode"] == "min"
    record = best["metrics"]["val_abs_bias"]["best"]
    assert record["objective_metadata"] == runtime.objective_metadata()
    assert "loss_config" not in record
    payload = torch.load(layout.checkpoints_dir / "ep001.pth", weights_only=True)
    assert payload["method"]["name"] == "tiny_reconstruction"
    assert payload["method"]["implementation"] == {
        "version": "1",
        "source": "tests.external_method.tiny_reconstruction",
    }
    assert payload["method"]["components"] == {
        "network": {
            "name": "tiny_conv",
            "version": "1",
            "source": "tests.external_method.tiny_reconstruction",
            "options": {"width": 8},
        }
    }
    assert set(payload["state"]) == {"network", "optimizer", "scheduler"}


def test_resume_restores_state_after_identity_validation(tmp_path: Path) -> None:
    definitions, _ = _definitions()
    config = RunConfig.from_mapping(_mapping(tmp_path), definitions)
    source, _ = _train(config)

    resumed = config.method.definition.build_training_runtime(config, _CPU, seed=0)
    assert isinstance(resumed, TinyRuntime)
    assert _trainer(config, resumed).resume("latest") == 2
    for name, value in source.network.state_dict().items():
        assert torch.equal(value, resumed.network.state_dict()[name])


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"architecture": "tiny_residual"}, "method.components.network.name is 'tiny_conv'"),
        ({"width": 4}, "method.components.network.options.width is 8"),
    ],
)
def test_resume_rejects_changed_component_before_touching_state(
    tmp_path: Path, options: dict[str, Any], message: str
) -> None:
    definitions, _ = _definitions()
    _train(RunConfig.from_mapping(_mapping(tmp_path), definitions))
    changed = _resolve(tmp_path, definitions, **options)
    torch.manual_seed(1)
    target = changed.method.definition.build_training_runtime(changed, _CPU, seed=0)
    assert isinstance(target, TinyRuntime)
    before = {key: value.clone() for key, value in target.network.state_dict().items()}

    with pytest.raises(CheckpointCompatibilityError, match=message):
        _trainer(changed, target).resume("latest")
    assert all(torch.equal(before[k], v) for k, v in target.network.state_dict().items())


# --- inference-only reconstruction ---------------------------------------------------------


@pytest.mark.parametrize("architecture", ["tiny_conv", "tiny_residual"])
def test_inference_builds_only_the_prediction_network(tmp_path: Path, architecture: str) -> None:
    definitions, tiny = _definitions()
    config = _resolve(tmp_path, definitions, architecture=architecture)
    source, layout = _train(config)
    tiny.built.clear()

    model, checkpoint = load_inference_generator(config, layout, _CPU)

    assert tiny.built == {"network": 1}
    assert checkpoint.parent == layout.checkpoints_dir
    assert [type(module).__name__ for module in model.children()] == (
        ["Conv2d", "Conv2d", "Conv2d"] if architecture == "tiny_residual" else ["Conv2d", "Conv2d"]
    )
    assert not model.training
    best_epoch = json.loads((layout.checkpoints_dir / "best.json").read_text())["metrics"][
        "val_abs_bias"
    ]["best"]["epoch"]
    assert checkpoint.name == f"ep{best_epoch:03d}.pth"


def test_output_and_reporting_paths_do_not_affect_reconstruction(tmp_path: Path) -> None:
    definitions, _ = _definitions()
    _, layout = _train(_resolve(tmp_path, definitions))
    data = _mapping(tmp_path)
    data["inference"]["output_dir"] = str(tmp_path / "elsewhere" / "predictions")
    data["evaluation"] = {"output_dir": str(tmp_path / "reports"), "save_graphs": True}
    data["training"].update({"log_rate": 7, "checkpoint_top_k": 1})

    model, _ = load_inference_generator(RunConfig.from_mapping(data, definitions), layout, _CPU)

    assert model.input_names == ("source",)


def test_wrong_architecture_is_never_loaded_silently(tmp_path: Path) -> None:
    definitions, tiny = _definitions()
    _, layout = _train(_resolve(tmp_path, definitions, architecture="tiny_residual"))
    conv = _resolve(tmp_path, definitions, architecture="tiny_conv")
    tiny.built.clear()

    with pytest.raises(CheckpointCompatibilityError, match="network.name is 'tiny_residual'"):
        load_inference_generator(conv, layout, _CPU)
    assert tiny.built == {}


def test_checkpoint_naming_unsupplied_definitions_fails_before_building(tmp_path: Path) -> None:
    definitions, tiny = _definitions()
    _, layout = _train(_resolve(tmp_path, definitions, architecture="tiny_residual"))
    checkpoint = layout.checkpoints_dir / "ep001.pth"
    builtin = RunConfig.from_mapping(
        {
            **{key: _mapping(tmp_path)[key] for key in ("dataset_root", "results_path")},
            "run_name": "external_example",
            "image_size": [32, 32],
            "model": {"inputs": ["source"], "target": "target"},
            "inference": {"checkpoint_path": str(checkpoint)},
        }
    )
    with pytest.raises(DefinitionNotAvailableError, match="method 'tiny_reconstruction'.*not"):
        load_inference_generator(builtin, layout, _CPU)

    without_residual = builtin_definitions().extend(methods=[tiny], components=[TINY_CONV])
    config = RunConfig.from_mapping(_mapping(tmp_path), without_residual)
    tiny.built.clear()
    with pytest.raises(DefinitionNotAvailableError, match="uses 'tiny_residual'.*not available"):
        load_inference_generator(config, layout, _CPU, checkpoint)
    assert tiny.built == {}


# --- tracked applications ----------------------------------------------------------------


def _paired_dataset(root: Path) -> None:
    records = []
    for index, split in enumerate(("train", "train", "val", "test")):
        sample_id = f"{index * 256:05}_00000"
        record = make_manifest_record(
            sample_id,
            split,
            set_id=f"S{index}",
            ext=".png",
            input_paths={"source": Path(f"splits/{split}/{sample_id}__input__source.png")},
        )
        records.append(record)
        for offset, path in enumerate((*record.input_paths.values(), record.target_path)):
            write_rgb_image(root / path, size=(32, 32), color=(40 * index, 30 * offset, 7))
    metadata = manifest_metadata(("source",))
    metadata = type(metadata)(metadata.schema_version, ("source",), "source", "target")
    manifest = DatasetManifest(tuple(records), root, metadata)
    layout = DatasetLayout(root)
    manifest.to_csv(layout.manifest_path)
    layout.manifest_metadata_path.write_text(json.dumps(metadata.to_dict()), encoding="utf-8")


def test_tracked_train_and_infer_applications_run_the_external_method(tmp_path: Path) -> None:
    definitions, _ = _definitions()
    _paired_dataset(tmp_path / "dataset")
    config_path = tmp_path / "external.yaml"
    config_path.write_text(yaml.safe_dump(_mapping(tmp_path)), encoding="utf-8")
    config = RunConfig.from_yaml(config_path, definitions)

    result = train(config, config_path)
    run = RunLayout.from_project(config.project)
    produced = infer(config, config_path)
    image = write_rgb_image(tmp_path / "single" / "sample.png", size=(32, 32))
    single = infer_images(
        config_path, (str(image),), tmp_path / "out.png", mode="resize", definitions=definitions
    )

    assert result.best_checkpoint_path is not None
    assert produced.num_samples == 1
    assert single.output_path.is_file()  # type: ignore[union-attr]
    resolved = yaml.safe_load(
        next(run.config_dir.glob("**/resolved*.yaml")).read_text(encoding="utf-8")
    )
    assert resolved["method"] == config.to_dict()["method"]
    stage = json.loads(run.stage_record("train").read_text(encoding="utf-8"))
    assert stage["consumed_data"]["snapshot_id"]
    assert stage["details"]["method_definition"] == {
        "name": "tiny_reconstruction",
        "version": "1",
        "source": "tests.external_method.tiny_reconstruction",
    }
    with pytest.raises(DefinitionNotAvailableError):
        infer_images(config_path, (str(image),), tmp_path / "builtin.png", mode="resize")
