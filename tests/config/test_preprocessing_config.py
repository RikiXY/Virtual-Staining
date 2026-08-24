from __future__ import annotations

from pathlib import Path

import pytest

from virtual_staining.data.config import PreprocessingConfig


def _mapping(**overrides: object) -> dict[str, object]:
    data: dict[str, object] = {
        "inputs": {
            "inventory": "inputs/pairs.csv",
            "source_modality": "AF",
            "target_modality": "H&E",
        },
        "split": {"unit": "pair", "train": 0.8, "val": 0.1, "test": 0.1},
    }
    data.update(overrides)
    return data


def test_from_mapping_uses_explicit_project_context() -> None:
    config = PreprocessingConfig.from_mapping(
        _mapping(), dataset_root=Path("/data"), default_image_size=(320, 256)
    )
    assert config.dataset_root == Path("/data")
    assert config.patching.patch_size == (320, 256)
    assert config.patching.grid_movement == (320, 256)


def test_patch_size_overrides_shared_image_size() -> None:
    config = PreprocessingConfig.from_mapping(
        _mapping(patching={"patch_size": [128, 64]}),
        dataset_root=Path("/data"),
        default_image_size=(256, 256),
    )
    assert config.patching.patch_size == (128, 64)
    assert config.to_dict()["patching"]["patch_size"] == [128, 64]


@pytest.mark.parametrize(
    "legacy_key",
    ["source_name", "target_name", "save_masks", "train_ratio", "tiled_io"],
)
def test_from_mapping_rejects_legacy_preprocessing_fields(legacy_key: str) -> None:
    with pytest.raises(ValueError, match=legacy_key):
        PreprocessingConfig.from_mapping(
            _mapping(**{legacy_key: "legacy"}),
            dataset_root=Path("/data"),
            default_image_size=(256, 256),
        )


def test_to_dict_round_trip_preserves_canonical_sections() -> None:
    config = PreprocessingConfig.from_mapping(
        _mapping(
            patching={"patch_size": [512, 256], "grid_movement": [128, 64]},
            masks={"save_patch_masks": True, "scale": 0.25},
            split={"unit": "pair", "train": 0.7, "val": 0.1, "test": 0.2},
        ),
        dataset_root=Path("/data"),
        default_image_size=(256, 256),
    )
    assert (
        PreprocessingConfig.from_mapping(
            config.to_dict(),
            dataset_root=config.dataset_root,
            default_image_size=(256, 256),
        )
        == config
    )


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"patching": {"patch_size": [0, 64]}}, "patching.patch_size"),
        ({"split": {"unit": "pair", "train": 0.5, "val": 0.1, "test": 0.1}}, "sum"),
        ({"masks": {"scale": 0.0}}, "masks.scale"),
        ({"masks": {"save_patch_masks": "false"}}, "YAML boolean"),
        ({"unknown": 1}, "unknown"),
    ],
)
def test_invalid_mapping_is_rejected(overrides: dict[str, object], match: str) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        PreprocessingConfig.from_mapping(
            _mapping(**overrides),
            dataset_root=Path("/data"),
            default_image_size=(256, 256),
        )


def test_inventory_config_parses_nested_policies() -> None:
    config = PreprocessingConfig.from_mapping(
        _mapping(
            patching={"patch_size": [64, 32], "margin": 10},
            masks={"generation": "never"},
            alignment={"mode": "never"},
            filtering={"foreground": {"enabled": False}},
            io={"tiled": True, "backend": "pillow"},
        ),
        dataset_root=Path("/data"),
        default_image_size=(256, 256),
    )
    assert config.inputs.inventory == Path("inputs/pairs.csv")
    assert config.patching.patch_size == (64, 32)
    assert config.split.unit == "pair"
    assert config.to_dict()["inputs"]["source_modality"] == "AF"


def test_inputs_and_split_are_required() -> None:
    with pytest.raises(ValueError, match="inputs"):
        PreprocessingConfig.from_mapping(
            {"split": {"unit": "pair", "train": 0.8, "val": 0.1, "test": 0.1}},
            dataset_root=Path("/data"),
            default_image_size=(256, 256),
        )
