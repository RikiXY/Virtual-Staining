from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from virtual_staining.config.data import PreprocessingConfig


def _mapping(**overrides: object) -> dict[str, object]:
    data: dict[str, object] = {
        "inputs": {
            "inventory": "inputs/slides.csv",
            "modalities": ["LF", "AF"],
            "reference": "LF",
            "target_modality": "stained",
        },
        "split": {"unit": "set", "train": 0.8, "val": 0.1, "test": 0.1},
    }
    data.update(overrides)
    return data


def test_from_mapping_uses_explicit_project_context() -> None:
    config = PreprocessingConfig.from_mapping(
        _mapping(), dataset_root=Path("/data"), default_image_size=(320, 256)
    )
    assert config.dataset_root == Path("/data")
    assert config.patching.patch_size == (320, 256)
    assert config.inputs.modalities == ("LF", "AF")


def test_patch_size_overrides_shared_image_size() -> None:
    config = PreprocessingConfig.from_mapping(
        _mapping(patching={"patch_size": [128, 64]}),
        dataset_root=Path("/data"),
        default_image_size=(256, 256),
    )
    assert config.patching.patch_size == (128, 64)


@pytest.mark.parametrize(
    "legacy_key", ["source_modality", "provided_layout", "source_strategy", "target_strategy"]
)
def test_from_mapping_rejects_removed_fields(legacy_key: str) -> None:
    inputs = cast(dict[str, object], _mapping()["inputs"])
    if legacy_key == "source_modality":
        inputs[legacy_key] = "LF"
        mapping = _mapping(inputs=inputs)
    else:
        mapping = _mapping(masks={legacy_key: "legacy"})
    with pytest.raises(ValueError, match=legacy_key):
        PreprocessingConfig.from_mapping(
            mapping, dataset_root=Path("/data"), default_image_size=(256, 256)
        )


def test_to_dict_round_trip_preserves_canonical_sections() -> None:
    config = PreprocessingConfig.from_mapping(
        _mapping(patching={"patch_size": [512, 256]}),
        dataset_root=Path("/data"),
        default_image_size=(256, 256),
    )
    assert (
        PreprocessingConfig.from_mapping(
            config.to_dict(), dataset_root=config.dataset_root, default_image_size=(256, 256)
        )
        == config
    )


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        (
            {
                "inputs": {
                    "modalities": ["LF", "LF"],
                    "reference": "LF",
                    "target_modality": "stained",
                    "inventory": "i.csv",
                }
            },
            "unique",
        ),
        (
            {
                "inputs": {
                    "modalities": ["LF"],
                    "reference": "AF",
                    "target_modality": "stained",
                    "inventory": "i.csv",
                }
            },
            "reference",
        ),
        (
            {
                "inputs": {
                    "modalities": ["LF"],
                    "reference": "LF",
                    "target_modality": "LF",
                    "inventory": "i.csv",
                }
            },
            "target",
        ),
        ({"split": {"unit": "pair", "train": 0.5, "val": 0.1, "test": 0.1}}, "unit"),
    ],
)
def test_invalid_mapping_is_rejected(overrides: dict[str, object], match: str) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        PreprocessingConfig.from_mapping(
            _mapping(**overrides), dataset_root=Path("/data"), default_image_size=(256, 256)
        )


def test_inventory_config_parses_nested_policies() -> None:
    config = PreprocessingConfig.from_mapping(
        _mapping(masks={"generation": "never"}, alignment={"mode": "never"}),
        dataset_root=Path("/data"),
        default_image_size=(256, 256),
    )
    assert config.inputs.inventory == Path("inputs/slides.csv")
    assert config.split.unit == "set"
    assert config.masks.generation == "never"


def test_inputs_and_split_are_required() -> None:
    with pytest.raises(ValueError, match="inputs"):
        PreprocessingConfig.from_mapping(
            {"split": {"unit": "set", "train": 0.8, "val": 0.1, "test": 0.1}},
            dataset_root=Path("/data"),
            default_image_size=(256, 256),
        )
