from __future__ import annotations

from pathlib import Path

import pytest

from tests.config_helpers import write_yaml
from virtual_staining.applications.train import _requires_foreground_masks
from virtual_staining.config.run import RunConfig


def _yaml(root: Path, *, target: str = "target") -> Path:
    return write_yaml(
        root / "run.yaml",
        f"""
dataset_root: {root / "dataset"}
results_path: {root / "results"}
run_name: run
image_size: [16, 16]
model:
  inputs: [LF, AF]
  target: {target}
preprocessing:
  inputs:
    inventory: inputs/slides.csv
    modalities: [LF, AF]
    reference: LF
    target_modality: target
  split:
    unit: set
    train: 0.8
    val: 0.1
    test: 0.1
training:
  epochs: 1
  losses:
    generator:
      - name: l1
        weight: 1.0
    discriminator: []
""",
    )


def test_training_config_uses_named_model_contract(tmp_path: Path) -> None:
    config = RunConfig.from_yaml(_yaml(tmp_path))
    assert config.model.inputs == ("LF", "AF")
    assert config.model.target == "target"
    assert config.training is not None
    assert _requires_foreground_masks(config) is False


def test_run_config_rejects_model_target_mismatch(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="target_modality"):
        RunConfig.from_yaml(_yaml(tmp_path, target="other"))
