from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import yaml

from virtual_staining.applications.evaluate import _paired_samples
from virtual_staining.config.run import RunConfig
from virtual_staining.data.dataset import UnpairedImageDataset, resolve_domain_images
from virtual_staining.data.manifest import DatasetManifest, ManifestMetadata, ManifestRecord
from virtual_staining.inference.runner import inference_input_names
from virtual_staining.methods.cyclegan import CycleGANMethod
from virtual_staining.methods.pix2pix import Pix2PixMethod
from virtual_staining.methods.registry import resolve_training_method
from virtual_staining.models.factory import build_generator
from virtual_staining.training.checkpoints import MethodCheckpointManager


class TinyCustomGenerator(nn.Module):
    def __init__(self, channels: int = 3) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.conv(value))


class InvalidCustomMethod:
    def __init__(self, **_kwargs: object) -> None:
        pass


def _write_config(tmp_path: Path, *, method: str = "cyclegan") -> RunConfig:
    source = tmp_path / "a"
    target = tmp_path / "b"
    source.mkdir(exist_ok=True)
    target.mkdir(exist_ok=True)
    data: dict[str, object] = {
        "dataset_root": str(tmp_path),
        "results_path": str(tmp_path / "results"),
        "run_name": "method_test",
        "image_size": [64, 64],
        "method": {"name": method},
        "data": {
            "pairing": "unpaired" if method == "cyclegan" else "paired",
            **({"domains": {"source": "a", "target": "b"}} if method == "cyclegan" else {}),
        },
        "model": {
            "inputs": ["source"],
            "target": "target",
            "generator": {
                "architecture": "resnet" if method == "cyclegan" else "concat_unet",
                "base_channels": 8,
                "blocks": 1,
                "norm": "instance",
            },
            "discriminator": {"ndf": 8, "norm": "instance"},
        },
        "training": {
            "batch_size": 1,
            "epochs": 1,
            "num_workers": 0,
            "validate_rate": 1,
            "checkpoint_rate": 1,
            "losses": (
                {
                    "generator": [
                        {"name": "adversarial_lsgan", "weight": 1.0},
                        {"name": "cycle_l1", "weight": 10.0},
                        {"name": "identity_l1", "weight": 5.0},
                    ],
                    "discriminator": [{"name": "adversarial_lsgan", "weight": 1.0}],
                }
                if method == "cyclegan"
                else {
                    "generator": [
                        {"name": "adversarial_bce", "weight": 1.0},
                        {"name": "l1", "weight": 10.0},
                    ],
                    "discriminator": [{"name": "adversarial_bce", "weight": 1.0}],
                }
            ),
        },
        "inference": {"checkpoint_policy": "latest"},
    }
    path = tmp_path / f"{method}.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return RunConfig.from_yaml(path)


def test_existing_config_defaults_to_paired_pix2pix(tmp_path: Path) -> None:
    config = _write_config(tmp_path, method="pix2pix")
    assert config.method.name == "pix2pix"
    assert config.data.pairing == "paired"
    assert isinstance(resolve_training_method(config, torch.device("cpu")), Pix2PixMethod)


def test_cyclegan_constructs_and_defaults_inference_direction(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    method = resolve_training_method(config, torch.device("cpu"))
    assert isinstance(method, CycleGANMethod)
    assert config.inference is not None
    assert config.inference.direction == "A_to_B"
    assert inference_input_names(config) == ("source",)
    object.__setattr__(config, "inference", replace(config.inference, direction="B_to_A"))
    assert inference_input_names(config) == ("target",)


def test_unpaired_dataset_supports_unequal_domain_sizes(tmp_path: Path) -> None:
    paths_a = tuple(tmp_path / f"a{index}.png" for index in range(3))
    paths_b = (tmp_path / "b0.png",)
    dataset = UnpairedImageDataset(paths_a, paths_b, random_pairing=False)
    assert len(dataset) == 3


def test_unpaired_domain_pattern_reuses_mixed_split_directory(tmp_path: Path) -> None:
    split = tmp_path / "splits" / "train"
    split.mkdir(parents=True)
    (split / "one_source.tif").touch()
    (split / "one_target.tif").touch()
    paths = resolve_domain_images(tmp_path, Path("splits/{split}/*_source.tif"), "train")
    assert paths == (split / "one_source.tif",)


def test_paired_evaluation_maps_cyclegan_outputs_in_both_directions(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    record = ManifestRecord(
        sample_id="sample",
        set_id="set",
        split="test",
        input_paths={"source": Path("splits/test/sample_source.png")},
        target_path=Path("splits/test/sample_target.png"),
        x=0,
        y=0,
        width=64,
        height=64,
    )
    manifest = DatasetManifest(
        (record,),
        tmp_path,
        ManifestMetadata("3.0", ("source",), "source", "target"),
    )
    generated_dir = tmp_path / "generated"

    a_to_b = _paired_samples(config, manifest, generated_dir)[0]
    assert a_to_b.target_path == tmp_path / "splits/test/sample_target.png"
    assert a_to_b.generated_path == generated_dir / "sample_source_generated.png"

    assert config.inference is not None
    object.__setattr__(config, "inference", replace(config.inference, direction="B_to_A"))
    b_to_a = _paired_samples(config, manifest, generated_dir)[0]
    assert b_to_a.target_path == tmp_path / "splits/test/sample_source.png"
    assert b_to_a.generated_path == generated_dir / "sample_target_generated.png"


def test_cyclegan_cpu_training_step(tmp_path: Path) -> None:
    method = CycleGANMethod(_write_config(tmp_path), torch.device("cpu"))
    losses = method.step(
        {"domain_a": torch.randn(1, 3, 64, 64), "domain_b": torch.randn(1, 3, 64, 64)},
        epoch=0,
        global_step=0,
    )
    assert losses.loss_G > 0
    assert losses.loss_D > 0
    assert losses.raw is not None
    assert "generator_cycle_l1" in losses.raw


@pytest.mark.parametrize("method_name", ["pix2pix", "cyclegan"])
def test_method_checkpoint_round_trip(tmp_path: Path, method_name: str) -> None:
    config = _write_config(tmp_path, method=method_name)
    first = resolve_training_method(config, torch.device("cpu"))
    manager = MethodCheckpointManager(
        tmp_path / "checkpoints",
        first,
        (None, None),
        image_size=(64, 64),
        device=torch.device("cpu"),
        resolved_config=config.to_dict(),
    )
    path = manager.save(2)
    second = resolve_training_method(config, torch.device("cpu"))
    loader = MethodCheckpointManager(
        tmp_path / "checkpoints",
        second,
        (None, None),
        image_size=(64, 64),
        device=torch.device("cpu"),
        resolved_config=config.to_dict(),
    )
    assert loader.load(path) == 3
    first_model = next(iter(first.state_dict()["models"].values()))
    second_model = next(iter(second.state_dict()["models"].values()))
    parameter_name = next(iter(first_model))
    assert torch.equal(first_model[parameter_name], second_model[parameter_name])


def test_custom_component_loading_and_early_failure() -> None:
    from virtual_staining.config.model import ModelConfig

    config = ModelConfig.from_mapping(
        {
            "inputs": ["source"],
            "target": "target",
            "generator": {
                "architecture": "custom",
                "class_path": "tests.methods.test_method_framework:TinyCustomGenerator",
                "params": {"channels": 3},
            },
        }
    )
    assert type(build_generator(config)).__name__ == "TinyCustomGenerator"

    invalid = ModelConfig.from_mapping(
        {
            "inputs": ["source"],
            "target": "target",
            "generator": {"architecture": "custom", "class_path": "builtins:str"},
        }
    )
    with pytest.raises(TypeError, match="torch.nn.Module"):
        build_generator(invalid)


def test_invalid_custom_method_fails_with_contract_message(tmp_path: Path) -> None:
    config = _write_config(tmp_path, method="pix2pix")
    object.__setattr__(config.method, "name", "custom")
    object.__setattr__(
        config.method,
        "class_path",
        "tests.methods.test_method_framework:InvalidCustomMethod",
    )
    with pytest.raises(TypeError, match="training method contract"):
        resolve_training_method(config, torch.device("cpu"))
