from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from tests.config_helpers import write_yaml
from virtual_staining.config import load_yaml_mapping, parse_bool_strict, reject_unknown_keys
from virtual_staining.config.run import RunConfig
from virtual_staining.experiment.snapshots import compute_config_hash, save_resolved_config


def _canonical_yaml() -> str:
    return """
dataset_root: /tmp/ds
results_path: /tmp/results
run_name: test_run
image_size: [256, 256]
model:
  generator:
    dropout: true
preprocessing:
  source_name: source.tif
  target_name: target.tif
training:
  epochs: 10
  augmentation:
    enabled: false
  losses:
    generator:
      - name: l1
        weight: 25.0
    discriminator:
      - name: adversarial_bce
        weight: 1.0
inference:
  checkpoint_policy: latest
evaluation:
  save_graphs: true
"""


def test_loader_and_validation_helpers(tmp_path: Path) -> None:
    path = write_yaml(tmp_path / "mapping.yaml", "key: value")
    assert load_yaml_mapping(path) == {"key": "value"}
    assert parse_bool_strict(True, "flag") is True
    reject_unknown_keys({"a": 1}, frozenset({"a"}), "test")

    write_yaml(path, "- item")
    with pytest.raises(ValueError, match="YAML mapping"):
        load_yaml_mapping(path)
    with pytest.raises(TypeError, match="YAML boolean"):
        parse_bool_strict("false", "flag")
    with pytest.raises(ValueError, match="Unknown key"):
        reject_unknown_keys({"b": 1}, frozenset({"a"}), "test")


def test_run_config_composes_domains_and_round_trips(tmp_path: Path) -> None:
    source = write_yaml(tmp_path / "run.yaml", _canonical_yaml())
    config = RunConfig.from_yaml(source)
    resolved = config.to_dict()
    resolved_path = tmp_path / "resolved.yaml"
    save_resolved_config(resolved, resolved_path)

    assert config.training is not None
    assert config.training.losses.generator[0].weight == 25.0
    assert config.training.augmentation.enabled is False
    assert config.preprocessing is not None
    assert config.preprocessing.dataset_root == config.project.dataset_root
    assert RunConfig.from_yaml(resolved_path) == config


@pytest.mark.parametrize(
    ("fragment", "match"),
    [
        ("model_image_size: [256, 256]", "model_image_size"),
        ("augmentation: {}", "augmentation"),
        ("losses: {}", "losses"),
        ("compare: {}", "compare"),
        ("compare_panels: {}", "compare_panels"),
        ("organize: {}", "organize"),
        ("save_graphs: true", "save_graphs"),
    ],
)
def test_run_config_rejects_legacy_top_level_forms(
    tmp_path: Path, fragment: str, match: str
) -> None:
    path = write_yaml(tmp_path / "run.yaml", _canonical_yaml() + "\n" + fragment)
    with pytest.raises(ValueError, match=match):
        RunConfig.from_yaml(path)


def test_model_validation_remains_strict(tmp_path: Path) -> None:
    path = write_yaml(
        tmp_path / "run.yaml",
        """
dataset_root: /tmp/ds
results_path: /tmp/results
run_name: test
model:
  discriminator:
    use_sigmoid: true
""",
    )
    with pytest.raises(ValueError, match="BCEWithLogitsLoss"):
        RunConfig.from_yaml(path)


def test_resolved_hash_is_stable_for_equivalent_mappings(tmp_path: Path) -> None:
    left = tmp_path / "left.yaml"
    right = tmp_path / "right.yaml"
    save_resolved_config({"training": {"epochs": 10}, "run_name": "x"}, left)
    save_resolved_config({"run_name": "x", "training": {"epochs": 10}}, right)
    assert yaml.safe_load(left.read_text()) == yaml.safe_load(right.read_text())
    assert compute_config_hash(left) == compute_config_hash(right)
