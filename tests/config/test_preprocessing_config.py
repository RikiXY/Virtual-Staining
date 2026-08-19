from __future__ import annotations

from pathlib import Path

import pytest

from virtual_staining.data.config import PreprocessingConfig


def _mapping(**overrides: object) -> dict[str, object]:
    data: dict[str, object] = {"source_name": "source.tif", "target_name": "target.tif"}
    data.update(overrides)
    return data


def test_from_mapping_uses_explicit_project_context() -> None:
    config = PreprocessingConfig.from_mapping(
        _mapping(), dataset_root=Path("/data"), default_image_size=(320, 256)
    )
    assert config.dataset_root == Path("/data")
    assert config.image_size == (320, 256)
    assert config.grid_movement == (320, 256)


def test_patch_size_overrides_shared_image_size() -> None:
    config = PreprocessingConfig.from_mapping(
        _mapping(patch_size=[128, 64]),
        dataset_root=Path("/data"),
        default_image_size=(256, 256),
    )
    assert config.image_size == (128, 64)
    assert config.to_dict()["patch_size"] == [128, 64]


@pytest.mark.parametrize("legacy_key", ["image_size", "dataset_root"])
def test_from_mapping_rejects_legacy_shared_fields(legacy_key: str) -> None:
    value: object = [256, 256] if legacy_key == "image_size" else "/data"
    with pytest.raises(ValueError, match=legacy_key):
        PreprocessingConfig.from_mapping(
            _mapping(**{legacy_key: value}),
            dataset_root=Path("/data"),
            default_image_size=(256, 256),
        )


def test_to_dict_round_trip_preserves_resolved_section() -> None:
    config = PreprocessingConfig.from_mapping(
        _mapping(
            patch_size=[512, 256],
            grid_movement=[128, 64],
            save_masks=True,
            mask_scale=0.25,
            train_ratio=0.7,
            val_ratio=0.1,
            test_ratio=0.2,
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
        ({"source_name": "same.tif", "target_name": "same.tif"}, "differ"),
        ({"train_ratio": 0.5}, "sum to 1"),
        ({"mask_scale": 0.0}, "mask_scale"),
        ({"save_masks": "false"}, "YAML boolean"),
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
