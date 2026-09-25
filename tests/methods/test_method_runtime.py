from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tests.config_helpers import write_yaml
from virtual_staining.config.run import RunConfig
from virtual_staining.methods.pix2pix import Pix2PixMethod
from virtual_staining.methods.registry import resolve_training_method


def _yaml(root: Path, *, method: str | None = None, extra_method: str = "") -> Path:
    method_section = ""
    if method is not None:
        method_section = f"method:\n  name: {method}\n{extra_method}"
    return write_yaml(
        root / "run.yaml",
        f"""
dataset_root: {root / "dataset"}
results_path: {root / "results"}
run_name: method_test
image_size: [16, 16]
{method_section}
model:
  inputs: [source]
  target: target
  generator:
    base_channels: 4
  discriminator:
    ndf: 4
training:
  batch_size: 1
  epochs: 1
  num_workers: 0
  losses:
    generator:
      - name: l1
        weight: 1.0
    discriminator: []
""",
    )


def test_method_defaults_to_pix2pix_and_is_persisted_in_resolved_config(tmp_path: Path) -> None:
    config = RunConfig.from_yaml(_yaml(tmp_path))
    assert config.method.name == "pix2pix"
    assert config.to_dict()["method"] == {"name": "pix2pix"}


def test_method_config_recognizes_cyclegan(tmp_path: Path) -> None:
    config = RunConfig.from_yaml(_yaml(tmp_path, method="cyclegan"))
    assert config.method.name == "cyclegan"
    assert config.to_dict()["method"] == {"name": "cyclegan"}


def test_method_config_rejects_unknown_method(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="method.name must be one of"):
        RunConfig.from_yaml(_yaml(tmp_path, method="stylegan"))


def test_method_config_rejects_dynamic_plugin_fields(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="class_path"):
        RunConfig.from_yaml(
            _yaml(
                tmp_path,
                method="pix2pix",
                extra_method="  class_path: package.module:Method\n",
            )
        )


def test_resolver_builds_pix2pix_runtime_without_exposing_optimizer_count(
    tmp_path: Path,
) -> None:
    config = RunConfig.from_yaml(_yaml(tmp_path))

    method = resolve_training_method(config, torch.device("cpu"))

    assert isinstance(method, Pix2PixMethod)
    assert method.name == "pix2pix"
    assert method.pairing == "paired"
    assert method.prediction_directions == ("forward",)
    assert not hasattr(method, "optimizers")
    assert method.input_names == ("source",)
    assert method.output_names == ("target",)
    assert set(method.component_metadata()) == {"generator", "discriminator"}


def test_resolver_reports_cyclegan_runtime_as_pending(tmp_path: Path) -> None:
    config = RunConfig.from_yaml(_yaml(tmp_path, method="cyclegan"))

    with pytest.raises(NotImplementedError, match="cyclegan.*not implemented"):
        resolve_training_method(config, torch.device("cpu"))


def test_pix2pix_method_state_round_trip(tmp_path: Path) -> None:
    config = RunConfig.from_yaml(_yaml(tmp_path))
    first = Pix2PixMethod(config, torch.device("cpu"))
    second = Pix2PixMethod(config, torch.device("cpu"))

    state = first.state_dict()
    second.load_state_dict(state)

    first_weight = next(iter(first.generator.state_dict().values()))
    second_weight = next(iter(second.generator.state_dict().values()))
    assert torch.equal(first_weight, second_weight)
