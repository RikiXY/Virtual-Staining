from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import pytest
import torch

from virtual_staining.checkpoint_contract import CheckpointCompatibilityError
from virtual_staining.config.inference import InferenceConfig
from virtual_staining.config.method import MethodConfig
from virtual_staining.config.model import ModelConfig
from virtual_staining.config.project import ProjectConfig
from virtual_staining.config.run import RunConfig
from virtual_staining.config.training import TrainingConfig
from virtual_staining.experiment.run_layout import RunLayout, ensure_run_directories
from virtual_staining.inference.runner import load_inference_generator, predict_batch
from virtual_staining.methods.pix2pix import Pix2PixMethod
from virtual_staining.training.checkpoints import MethodCheckpointManager

_CPU = torch.device("cpu")


def _config(tmp_path: Path, *, image_size: tuple[int, int] = (64, 64), **model: Any) -> RunConfig:
    model_mapping: dict[str, Any] = {
        "inputs": ["LF", "AF"],
        "target": "stained",
        "generator": {"base_channels": 4},
        "discriminator": {"ndf": 4},
    }
    model_mapping.update(model)
    return RunConfig(
        project=ProjectConfig(
            dataset_root=tmp_path / "dataset",
            results_path=tmp_path / "results",
            run_name="run",
            image_size=image_size,
        ),
        method=MethodConfig(),
        model=ModelConfig.from_mapping(model_mapping),
        training=TrainingConfig(
            batch_size=1,
            epochs=2,
            lr_g=2e-4,
            lr_d=2e-4,
            beta1=0.5,
            beta2=0.999,
            seed=0,
            num_workers=0,
            validate_rate=1,
            checkpoint_rate=1,
        ),
        inference=InferenceConfig(checkpoint_policy="latest"),
        preprocessing=None,
        evaluation=None,
    )


def _manager(config: RunConfig, method: Pix2PixMethod) -> MethodCheckpointManager:
    paths = RunLayout.from_project(config.project)
    ensure_run_directories(paths)
    return MethodCheckpointManager(
        method,
        paths.checkpoints_dir,
        image_size=config.project.image_size,
        device=_CPU,
        config_hash="sha256:test",
    )


def _batch() -> dict[str, Any]:
    torch.manual_seed(1)
    return {
        "inputs": {"LF": torch.rand(2, 3, 64, 64) * 2 - 1, "AF": torch.rand(2, 3, 64, 64) * 2 - 1},
        "target": torch.rand(2, 3, 64, 64) * 2 - 1,
        "masks": {},
    }


def _trained_checkpoint(tmp_path: Path, epoch: int = 4) -> tuple[RunConfig, Pix2PixMethod, Path]:
    config = _config(tmp_path)
    torch.manual_seed(0)
    method = Pix2PixMethod(config, _CPU)
    method.train_mode()
    method.step(_batch(), epoch=0, global_step=0)
    return config, method, _manager(config, method).save(epoch)


def _assert_modules_equal(left: torch.nn.Module, right: torch.nn.Module) -> None:
    right_state = right.state_dict()
    for name, value in left.state_dict().items():
        assert torch.equal(value, right_state[name]), name


def test_pix2pix_v4_payload_holds_method_owned_state_only(tmp_path: Path) -> None:
    _config_, _method, path = _trained_checkpoint(tmp_path)
    payload = torch.load(path, map_location="cpu", weights_only=False)

    assert set(payload) == {
        "format_version",
        "epoch",
        "method",
        "image_size",
        "normalization",
        "config_hash",
        "state",
    }
    assert payload["format_version"] == 4
    assert payload["method"]["name"] == "pix2pix"
    assert payload["method"]["pairing"] == "paired"
    assert payload["method"]["inputs"] == ["LF", "AF"]
    assert payload["method"]["outputs"] == ["stained"]
    assert payload["method"]["prediction_directions"] == ["forward"]
    assert payload["method"]["components"]["generator"]["class"] == "ConcatUNetGenerator"
    assert payload["method"]["components"]["generator"]["base_channels"] == 4
    assert payload["method"]["components"]["discriminator"]["class"] == "PatchGANDiscriminator"
    assert set(payload["state"]) == {"models", "optimizers", "scalers", "schedulers"}


def test_pix2pix_v4_save_resume_round_trip(tmp_path: Path) -> None:
    config, source, path = _trained_checkpoint(tmp_path, epoch=4)
    torch.manual_seed(99)
    resumed = Pix2PixMethod(config, _CPU)

    assert _manager(config, resumed).load(path) == 5

    _assert_modules_equal(source.generator, resumed.generator)
    _assert_modules_equal(source.discriminator, resumed.discriminator)
    source_state, resumed_state = source.state_dict(), resumed.state_dict()
    for name in ("generator", "discriminator"):
        source_opt = source_state["optimizers"][name]
        resumed_opt = resumed_state["optimizers"][name]
        assert source_opt["param_groups"] == resumed_opt["param_groups"]
        for key, values in source_opt["state"].items():
            for field, value in values.items():
                assert torch.equal(value, resumed_opt["state"][key][field])
    assert source.learning_rates() == resumed.learning_rates()


def test_pix2pix_v4_inference_round_trip(tmp_path: Path) -> None:
    config, source, path = _trained_checkpoint(tmp_path)
    paths = RunLayout.from_project(config.project)

    generator, resolved_path = load_inference_generator(config, paths, _CPU)

    assert resolved_path == path
    assert not generator.training
    _assert_modules_equal(source.generator, generator)
    source.generator.eval()
    inputs = _batch()["inputs"]
    assert torch.equal(
        predict_batch(generator, inputs, _CPU), predict_batch(source.generator, inputs, _CPU)
    )


@pytest.mark.parametrize(
    ("changes", "field"),
    [
        ({"inputs": ["AF", "LF"]}, r"method.inputs\[0\]"),
        ({"target": "other"}, r"method.outputs\[0\]"),
        ({"generator": {"base_channels": 8}}, "method.components.generator.base_channels"),
        ({"generator": {"base_channels": 4, "norm": "instance"}}, "generator.norm"),
        ({"generator": {"base_channels": 4, "dropout": True}}, "generator.dropout"),
        ({"discriminator": {"ndf": 8}}, "method.components.discriminator.ndf"),
    ],
)
def test_pix2pix_config_mismatch_is_rejected_for_training_and_inference(
    tmp_path: Path, changes: dict[str, Any], field: str
) -> None:
    _trained_checkpoint(tmp_path)
    config = _config(tmp_path, **changes)
    method = Pix2PixMethod(config, _CPU)
    before = {name: value.clone() for name, value in method.generator.state_dict().items()}

    with pytest.raises(CheckpointCompatibilityError, match=field):
        _manager(config, method).load(
            RunLayout.from_project(config.project).checkpoints_dir / "ep004.pth"
        )
    with pytest.raises(CheckpointCompatibilityError, match=field):
        load_inference_generator(config, RunLayout.from_project(config.project), _CPU)
    for name, value in method.generator.state_dict().items():
        assert torch.equal(value, before[name])


def test_pix2pix_image_size_mismatch_is_rejected(tmp_path: Path) -> None:
    config, _method, path = _trained_checkpoint(tmp_path)
    resized = dataclasses.replace(
        config, project=dataclasses.replace(config.project, image_size=(32, 32))
    )
    with pytest.raises(CheckpointCompatibilityError, match="image_size"):
        _manager(resized, Pix2PixMethod(resized, _CPU)).load(path)
    with pytest.raises(CheckpointCompatibilityError, match="image_size"):
        load_inference_generator(resized, RunLayout.from_project(resized.project), _CPU)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda payload: payload["method"]["components"]["generator"].update(
                {"class": "ResNetGenerator"}
            ),
            "generator.class",
        ),
        (lambda payload: payload["method"].update({"name": "cyclegan"}), "method.name"),
        (lambda payload: payload["method"].update({"pairing": "unpaired"}), "method.pairing"),
        (
            lambda payload: payload["method"].update({"prediction_directions": ["backward"]}),
            "method.prediction_directions",
        ),
        (
            lambda payload: payload["normalization"].update({"output_range": "[0, 1]"}),
            "normalization.output_range",
        ),
        (lambda payload: payload.update({"format_version": 3}), "format version 3"),
        (lambda payload: payload.pop("format_version"), "unversioned"),
        (lambda payload: payload.pop("state"), "no method state"),
    ],
)
def test_pix2pix_stored_identity_mismatch_is_rejected(tmp_path: Path, mutate, message) -> None:
    config, _method, path = _trained_checkpoint(tmp_path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    mutate(payload)
    torch.save(payload, path)

    with pytest.raises(CheckpointCompatibilityError, match=message):
        _manager(config, Pix2PixMethod(config, _CPU)).load(path)
    with pytest.raises(CheckpointCompatibilityError, match=message):
        load_inference_generator(config, RunLayout.from_project(config.project), _CPU)


def test_pix2pix_rejects_legacy_v3_layout(tmp_path: Path) -> None:
    config, method, path = _trained_checkpoint(tmp_path)
    torch.save(
        {
            "format_version": 3,
            "epoch": 4,
            "architecture": {"generator": {}, "discriminator": {}},
            "generator_state_dict": method.generator.state_dict(),
            "discriminator_state_dict": method.discriminator.state_dict(),
            "image_size": list(config.project.image_size),
        },
        path,
    )
    with pytest.raises(CheckpointCompatibilityError, match="unsupported format version 3"):
        _manager(config, Pix2PixMethod(config, _CPU)).load(path)
    with pytest.raises(CheckpointCompatibilityError, match="unsupported format version 3"):
        load_inference_generator(config, RunLayout.from_project(config.project), _CPU)


def test_pix2pix_rejects_malformed_method_owned_state(tmp_path: Path) -> None:
    config, _method, path = _trained_checkpoint(tmp_path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    del payload["state"]["scalers"]
    torch.save(payload, path)

    with pytest.raises(KeyError, match="scalers"):
        _manager(config, Pix2PixMethod(config, _CPU)).load(path)
