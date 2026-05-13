from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from virtual_staining.config import (
    load_yaml_mapping,
    parse_bool_strict,
    reject_unknown_keys,
    section_with_shared_fields,
)
from virtual_staining.config.run import RunConfig
from virtual_staining.experiment.snapshots import compute_config_hash, save_resolved_config


def test_load_yaml_mapping_valid(tmp_path: Path) -> None:
    f = tmp_path / "test.yaml"
    f.write_text("key: value\nnumber: 42\n")
    assert load_yaml_mapping(f) == {"key": "value", "number": 42}


def test_load_yaml_mapping_non_mapping_raises(tmp_path: Path) -> None:
    f = tmp_path / "test.yaml"
    f.write_text("- item1\n- item2\n")
    with pytest.raises(ValueError, match="YAML mapping"):
        load_yaml_mapping(f)


def test_reject_unknown_keys_no_unknown() -> None:
    reject_unknown_keys({"a": 1, "b": 2}, frozenset({"a", "b", "c"}), "test")


def test_reject_unknown_keys_raises_on_unknown() -> None:
    with pytest.raises(ValueError, match="Unknown key"):
        reject_unknown_keys({"a": 1, "z": 99}, frozenset({"a", "b"}), "test")


def test_parse_bool_strict_true() -> None:
    assert parse_bool_strict(True, "flag") is True


def test_parse_bool_strict_false() -> None:
    assert parse_bool_strict(False, "flag") is False


def test_parse_bool_strict_string_raises() -> None:
    with pytest.raises(TypeError, match="YAML boolean"):
        parse_bool_strict("false", "flag")


def test_parse_bool_strict_int_raises() -> None:
    with pytest.raises(TypeError):
        parse_bool_strict(0, "flag")


def test_section_with_shared_fields_no_section_returns_full_data() -> None:
    data = {"a": 1, "b": 2}
    result = section_with_shared_fields(data, "training", {"a"})
    assert result == {"a": 1, "b": 2}


def test_section_with_shared_fields_merges_shared() -> None:
    data = {"a": 10, "training": {"b": 20}}
    result = section_with_shared_fields(data, "training", {"a"})
    assert result["a"] == 10
    assert result["b"] == 20


def test_section_with_shared_fields_section_overrides_shared() -> None:
    data = {"a": 10, "training": {"a": 99, "b": 20}}
    result = section_with_shared_fields(data, "training", {"a"})
    assert result["a"] == 99


def test_run_config_to_yaml_dict_includes_required_keys_and_omits_nones(tmp_path: Path) -> None:
    yaml_path = tmp_path / "run.yaml"
    yaml_path.write_text(
        """
dataset_root: /tmp/ds
results_path: /tmp/results
run_name: test_run
image_size: [256, 256]
training:
  epochs: 10
""".strip()
        + "\n",
        encoding="utf-8",
    )

    config = RunConfig.from_yaml(yaml_path)
    data = config.to_yaml_dict()

    assert data["dataset_root"] == "/tmp/ds"
    assert data["run_name"] == "test_run"
    assert data["image_size"] == [256, 256]
    assert data["model"]["name"] == "pix2pix"
    assert data["model"]["generator"]["norm"] == "batch"
    assert data["model"]["generator"]["dropout"] is False
    assert data["model"]["discriminator"]["norm"] == "instance"
    assert data["model"]["gan_loss"] == "bce"
    assert data["training"]["epochs"] == 10
    assert "seed" not in data["training"]
    assert "resume" not in data["training"]


def test_model_bilinear_string_false_raises(tmp_path: Path) -> None:
    yaml_path = tmp_path / "run.yaml"
    yaml_path.write_text(
        """
dataset_root: /tmp/ds
results_path: /tmp/results
run_name: t
image_size: [256, 256]
model:
  generator:
    bilinear: "false"
training:
  epochs: 1
""".strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(TypeError, match="bilinear"):
        RunConfig.from_yaml(yaml_path)


def test_model_use_sigmoid_string_true_raises(tmp_path: Path) -> None:
    yaml_path = tmp_path / "run.yaml"
    yaml_path.write_text(
        """
dataset_root: /tmp/ds
results_path: /tmp/results
run_name: t
image_size: [256, 256]
model:
  discriminator:
    use_sigmoid: "true"
training:
  epochs: 1
""".strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(TypeError, match="use_sigmoid"):
        RunConfig.from_yaml(yaml_path)


def test_model_bilinear_yaml_bool_false_parses(tmp_path: Path) -> None:
    yaml_path = tmp_path / "run.yaml"
    yaml_path.write_text(
        """
dataset_root: /tmp/ds
results_path: /tmp/results
run_name: t
image_size: [256, 256]
model:
  generator:
    bilinear: false
training:
  epochs: 1
""".strip()
        + "\n",
        encoding="utf-8",
    )

    config = RunConfig.from_yaml(yaml_path)

    assert config.model.generator.bilinear is False


def test_model_generator_dropout_string_true_raises(tmp_path: Path) -> None:
    yaml_path = tmp_path / "run.yaml"
    yaml_path.write_text(
        """
dataset_root: /tmp/ds
results_path: /tmp/results
run_name: t
image_size: [256, 256]
model:
  generator:
    dropout: "true"
training:
  epochs: 1
""".strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(TypeError, match="dropout"):
        RunConfig.from_yaml(yaml_path)


def test_model_generator_dropout_yaml_bool_true_parses(tmp_path: Path) -> None:
    yaml_path = tmp_path / "run.yaml"
    yaml_path.write_text(
        """
dataset_root: /tmp/ds
results_path: /tmp/results
run_name: t
image_size: [256, 256]
model:
  generator:
    dropout: true
training:
  epochs: 1
""".strip()
        + "\n",
        encoding="utf-8",
    )

    config = RunConfig.from_yaml(yaml_path)

    assert config.model.generator.dropout is True


def test_model_bilinear_yaml_bool_true_raises(tmp_path: Path) -> None:
    yaml_path = tmp_path / "run.yaml"
    yaml_path.write_text(
        """
dataset_root: /tmp/ds
results_path: /tmp/results
run_name: t
image_size: [256, 256]
model:
  generator:
    bilinear: true
training:
  epochs: 1
""".strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="bilinear"):
        RunConfig.from_yaml(yaml_path)


def test_model_use_sigmoid_yaml_bool_true_raises(tmp_path: Path) -> None:
    yaml_path = tmp_path / "run.yaml"
    yaml_path.write_text(
        """
dataset_root: /tmp/ds
results_path: /tmp/results
run_name: t
image_size: [256, 256]
model:
  discriminator:
    use_sigmoid: true
training:
  epochs: 1
""".strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="BCEWithLogitsLoss"):
        RunConfig.from_yaml(yaml_path)


def test_model_use_sigmoid_yaml_bool_false_parses(tmp_path: Path) -> None:
    yaml_path = tmp_path / "run.yaml"
    yaml_path.write_text(
        """
dataset_root: /tmp/ds
results_path: /tmp/results
run_name: t
image_size: [256, 256]
model:
  discriminator:
    use_sigmoid: false
training:
  epochs: 1
""".strip()
        + "\n",
        encoding="utf-8",
    )

    config = RunConfig.from_yaml(yaml_path)

    assert config.model.discriminator.use_sigmoid is False


def test_model_norm_and_loss_round_trip_are_preserved(tmp_path: Path) -> None:
    yaml_path = tmp_path / "run.yaml"
    yaml_path.write_text(
        """
dataset_root: /tmp/ds
results_path: /tmp/results
run_name: t
image_size: [256, 256]
model:
  name: pix2pix
  generator:
    norm: instance
    dropout: true
    bilinear: false
  discriminator:
    norm: batch
    use_sigmoid: false
  gan_loss: bce
training:
  epochs: 1
""".strip()
        + "\n",
        encoding="utf-8",
    )

    config = RunConfig.from_yaml(yaml_path)
    data = config.to_yaml_dict()

    assert data["model"] == {
        "name": "pix2pix",
        "generator": {
            "name": "unet",
            "in_channels": 3,
            "out_channels": 3,
            "base_channels": 64,
            "norm": "instance",
            "dropout": True,
            "bilinear": False,
        },
        "discriminator": {
            "name": "patchgan",
            "in_channels": 6,
            "ndf": 64,
            "norm": "batch",
            "use_sigmoid": False,
        },
        "gan_loss": "bce",
    }


@pytest.mark.parametrize(
    ("block", "match"),
    [
        ("name: cyclegan", "model.name"),
        ("gan_loss: hinge", "model.gan_loss"),
    ],
)
def test_model_top_level_choice_validation_rejects_invalid_values(
    tmp_path: Path, block: str, match: str
) -> None:
    yaml_path = tmp_path / "run.yaml"
    yaml_path.write_text(
        f"""
dataset_root: /tmp/ds
results_path: /tmp/results
run_name: t
image_size: [256, 256]
model:
  {block}
training:
  epochs: 1
""".strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=match):
        RunConfig.from_yaml(yaml_path)


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("generator", "norm", "group"),
        ("discriminator", "norm", "layer"),
        ("generator", "name", "resnet"),
        ("discriminator", "name", "basic"),
    ],
)
def test_model_nested_choice_validation_rejects_invalid_values(
    tmp_path: Path, section: str, field: str, value: str
) -> None:
    yaml_path = tmp_path / "run.yaml"
    yaml_path.write_text(
        f"""
dataset_root: /tmp/ds
results_path: /tmp/results
run_name: t
image_size: [256, 256]
model:
  {section}:
    {field}: {value}
training:
  epochs: 1
""".strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=field):
        RunConfig.from_yaml(yaml_path)


def test_save_resolved_config_writes_valid_yaml(tmp_path: Path) -> None:
    dest = tmp_path / "config" / "resolved.yaml"

    save_resolved_config({"dataset_root": "/tmp/ds", "image_size": [256, 256]}, dest)

    assert dest.exists()
    loaded = yaml.safe_load(dest.read_text(encoding="utf-8"))
    assert loaded["dataset_root"] == "/tmp/ds"
    assert loaded["image_size"] == [256, 256]


def test_resolved_config_hash_is_stable_for_equivalent_mappings(tmp_path: Path) -> None:
    left = tmp_path / "left.yaml"
    right = tmp_path / "right.yaml"

    save_resolved_config(
        {
            "training": {"epochs": 10, "batch_size": 8},
            "dataset_root": "/tmp/ds",
            "image_size": [256, 256],
        },
        left,
    )
    save_resolved_config(
        {
            "image_size": [256, 256],
            "dataset_root": "/tmp/ds",
            "training": {"batch_size": 8, "epochs": 10},
        },
        right,
    )

    assert compute_config_hash(left) == compute_config_hash(right)
