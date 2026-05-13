from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import pytest

from virtual_staining.data.config import PreprocessingConfig, load_preprocessing_config
from virtual_staining.utils.dimensions import to_torchvision_hw


def _make_namespace(**overrides: object) -> argparse.Namespace:
    defaults: dict[str, object] = dict(
        path="/data/samples",
        source_name="source.tif",
        target_name="target.tif",
        seed=None,
        save_masks=False,
        image_size=[256, 256],
        grid_movement=[256, 256],
        margin=200,
        min_foreground_ratio=0.25,
        max_white_ratio=0.7,
        white_threshold=250,
        max_largest_white_component_ratio=0.20,
        mask_scale=1.0,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_from_args_basic() -> None:
    config = PreprocessingConfig.from_args(_make_namespace())
    assert config.dataset_root == Path("/data/samples")
    assert config.source_name == "source.tif"
    assert config.target_name == "target.tif"
    assert config.image_size == (256, 256)
    assert config.grid_movement == (256, 256)
    assert config.margin == 200
    assert config.seed is None
    assert config.save_masks is False
    assert config.mask_scale == pytest.approx(1.0)


def test_from_args_thresholds() -> None:
    config = PreprocessingConfig.from_args(
        _make_namespace(
            min_foreground_ratio=0.3,
            max_white_ratio=0.6,
            white_threshold=240,
            max_largest_white_component_ratio=0.15,
        )
    )
    assert config.min_foreground_ratio == pytest.approx(0.3)
    assert config.max_white_ratio == pytest.approx(0.6)
    assert config.white_threshold == 240
    assert config.max_largest_white_component_ratio == pytest.approx(0.15)


def test_from_args_with_seed() -> None:
    config = PreprocessingConfig.from_args(_make_namespace(seed=99))
    assert config.seed == 99


def test_from_args_default_split_ratios() -> None:
    config = PreprocessingConfig.from_args(_make_namespace())
    assert config.train_ratio == pytest.approx(0.8)
    assert config.val_ratio == pytest.approx(0.05)
    assert config.test_ratio == pytest.approx(0.15)


def test_frozen() -> None:
    config = PreprocessingConfig.from_args(_make_namespace())
    with pytest.raises((AttributeError, TypeError)):
        config.margin = 999  # type: ignore[misc]


def test_from_yaml(tmp_path: Path) -> None:
    yaml_content = textwrap.dedent("""\
        dataset_root: /data/samples
        source_name: he.tif
        target_name: masson.tif
        image_size: [512, 512]
        grid_movement: [256, 256]
        margin: 100
        seed: 7
        save_masks: true
        mask_scale: 0.25
        train_ratio: 0.7
        val_ratio: 0.1
        test_ratio: 0.2
        min_foreground_ratio: 0.3
        max_white_ratio: 0.6
        white_threshold: 240
        max_largest_white_component_ratio: 0.15
    """)
    yaml_file = tmp_path / "preprocessing.yaml"
    yaml_file.write_text(yaml_content, encoding="utf-8")

    config = load_preprocessing_config(yaml_file)
    assert config.dataset_root == Path("/data/samples")
    assert config.source_name == "he.tif"
    assert config.target_name == "masson.tif"
    assert config.image_size == (512, 512)
    assert config.margin == 100
    assert config.seed == 7
    assert config.save_masks is True
    assert config.mask_scale == pytest.approx(0.25)
    assert config.train_ratio == pytest.approx(0.7)
    assert config.val_ratio == pytest.approx(0.1)
    assert config.test_ratio == pytest.approx(0.2)
    assert config.min_foreground_ratio == pytest.approx(0.3)
    assert config.white_threshold == 240


def test_from_args_partial_namespace() -> None:
    """from_args() falls back to dataclass defaults when optional fields are absent (SUPPRESS)."""
    args = argparse.Namespace(path="/data/samples", source_name="s.tif", target_name="t.tif")
    config = PreprocessingConfig.from_args(args)
    assert config.image_size == (256, 256)
    assert config.grid_movement == (256, 256)
    assert config.margin == 200
    assert config.seed is None
    assert config.save_masks is False
    assert config.mask_scale == pytest.approx(1.0)
    assert config.train_ratio == pytest.approx(0.8)
    assert config.min_foreground_ratio == pytest.approx(0.25)
    assert config.white_threshold == 250


def test_to_yaml_round_trip(tmp_path: Path) -> None:
    config = PreprocessingConfig(
        dataset_root=Path("/data/samples"),
        source_name="he.tif",
        target_name="masson.tif",
        image_size=(512, 512),
        grid_movement=(256, 256),
        margin=100,
        seed=7,
        save_masks=True,
        mask_scale=0.25,
        train_ratio=0.7,
        val_ratio=0.1,
        test_ratio=0.2,
        min_foreground_ratio=0.3,
        max_white_ratio=0.6,
        white_threshold=240,
        max_largest_white_component_ratio=0.15,
    )
    yaml_file = tmp_path / "config.yaml"
    config.to_yaml(yaml_file)
    assert yaml_file.exists()
    loaded = load_preprocessing_config(yaml_file)
    assert loaded == config


def test_mask_scale_invalid_direct_init_raises() -> None:
    with pytest.raises(ValueError, match="mask_scale"):
        PreprocessingConfig(
            dataset_root=Path("/data/samples"),
            source_name="he.tif",
            target_name="masson.tif",
            mask_scale=0.0,
        )


def test_from_yaml_defaults_for_optional_fields(tmp_path: Path) -> None:
    yaml_content = textwrap.dedent("""\
        dataset_root: /data/samples
        source_name: source.tif
        target_name: target.tif
    """)
    yaml_file = tmp_path / "minimal.yaml"
    yaml_file.write_text(yaml_content, encoding="utf-8")

    config = load_preprocessing_config(yaml_file)
    assert config.image_size == (256, 256)
    assert config.grid_movement == (256, 256)
    assert config.margin == 200
    assert config.seed is None
    assert config.save_masks is False
    assert config.mask_scale == pytest.approx(1.0)
    assert config.train_ratio == pytest.approx(0.8)
    assert config.val_ratio == pytest.approx(0.05)
    assert config.test_ratio == pytest.approx(0.15)
    assert config.min_foreground_ratio == pytest.approx(0.25)
    assert config.max_white_ratio == pytest.approx(0.7)
    assert config.white_threshold == 250
    assert config.max_largest_white_component_ratio == pytest.approx(0.20)


def test_from_run_yaml_section(tmp_path: Path) -> None:
    yaml_content = textwrap.dedent("""\
        dataset_root: /data/samples
        image_size: [512, 512]
        preprocessing:
          source_name: label_free.tif
          target_name: he.tif
          grid_movement: [512, 512]
          save_masks: true
          mask_scale: 0.5
    """)
    yaml_file = tmp_path / "run.yaml"
    yaml_file.write_text(yaml_content, encoding="utf-8")

    config = load_preprocessing_config(yaml_file)
    assert config.dataset_root == Path("/data/samples")
    assert config.source_name == "label_free.tif"
    assert config.target_name == "he.tif"
    assert config.image_size == (512, 512)
    assert config.grid_movement == (512, 512)
    assert config.save_masks is True
    assert config.mask_scale == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("overrides", "field"),
    [
        ({"source_name": ""}, "source_name"),
        ({"source_name": "same.tif", "target_name": "same.tif"}, "target_name"),
        ({"image_size": [0, 256]}, "image_size"),
        ({"grid_movement": [256, -1]}, "grid_movement"),
        ({"margin": -1}, "margin"),
        ({"mask_scale": 0.0}, "mask_scale"),
        ({"mask_scale": 1.5}, "mask_scale"),
        ({"train_ratio": 1.2}, "train_ratio"),
        ({"train_ratio": 0.6, "val_ratio": 0.2, "test_ratio": 0.3}, "split"),
        ({"min_foreground_ratio": -0.1}, "min_foreground_ratio"),
        ({"max_white_ratio": 1.1}, "max_white_ratio"),
        (
            {"max_largest_white_component_ratio": 1.1},
            "max_largest_white_component_ratio",
        ),
        ({"white_threshold": 256}, "white_threshold"),
    ],
)
def test_from_args_invalid_values_raise_value_error(
    overrides: dict[str, object], field: str
) -> None:
    with pytest.raises(ValueError, match=field):
        PreprocessingConfig.from_args(_make_namespace(**overrides))


def test_non_square_image_size_from_args() -> None:
    """image_size=[320, 256] must parse as (width=320, height=256)."""
    config = PreprocessingConfig.from_args(_make_namespace(image_size=[320, 256]))
    assert config.image_size == (320, 256)


def test_non_square_image_size_from_yaml(tmp_path: Path) -> None:
    """image_size: [320, 256] in YAML must parse as (width=320, height=256)."""
    yaml_content = textwrap.dedent("""\
        dataset_root: /data/samples
        source_name: source.tif
        target_name: target.tif
        image_size: [320, 256]
    """)
    yaml_file = tmp_path / "ns.yaml"
    yaml_file.write_text(yaml_content, encoding="utf-8")
    config = load_preprocessing_config(yaml_file)
    assert config.image_size == (320, 256)


def test_to_torchvision_hw_swaps_for_non_square() -> None:
    """to_torchvision_hw must return (height, width) from a (width, height) pair."""
    assert to_torchvision_hw((320, 256)) == (256, 320)


def test_to_torchvision_hw_preserves_square() -> None:
    """to_torchvision_hw must be a no-op for square sizes."""
    assert to_torchvision_hw((256, 256)) == (256, 256)


def test_from_yaml_invalid_margin_raises_value_error(tmp_path: Path) -> None:
    yaml_content = textwrap.dedent("""\
        dataset_root: /data/samples
        source_name: source.tif
        target_name: target.tif
        margin: -1
    """)
    yaml_file = tmp_path / "invalid.yaml"
    yaml_file.write_text(yaml_content, encoding="utf-8")

    with pytest.raises(ValueError, match="margin"):
        load_preprocessing_config(yaml_file)


def test_from_yaml_unknown_flat_key_raises(tmp_path: Path) -> None:
    yaml_content = textwrap.dedent("""\
        dataset_root: /data/samples
        source_name: source.tif
        target_name: target.tif
        sav_masks: true
    """)
    yaml_file = tmp_path / "typo.yaml"
    yaml_file.write_text(yaml_content, encoding="utf-8")
    with pytest.raises(ValueError, match="sav_masks"):
        load_preprocessing_config(yaml_file)


def test_from_yaml_unknown_section_key_raises(tmp_path: Path) -> None:
    yaml_content = textwrap.dedent("""\
        dataset_root: /data/samples
        preprocessing:
          source_name: source.tif
          target_name: target.tif
          sav_masks: true
    """)
    yaml_file = tmp_path / "typo_section.yaml"
    yaml_file.write_text(yaml_content, encoding="utf-8")
    with pytest.raises(ValueError, match="sav_masks"):
        load_preprocessing_config(yaml_file)


def test_from_yaml_unknown_top_level_key_raises(tmp_path: Path) -> None:
    yaml_content = textwrap.dedent("""\
        dataset_root: /data/samples
        typo_field: oops
        preprocessing:
          source_name: source.tif
          target_name: target.tif
    """)
    yaml_file = tmp_path / "typo_top.yaml"
    yaml_file.write_text(yaml_content, encoding="utf-8")
    with pytest.raises(ValueError, match="typo_field"):
        load_preprocessing_config(yaml_file)


def test_from_yaml_string_bool_save_masks_raises(tmp_path: Path) -> None:
    yaml_content = textwrap.dedent("""\
        dataset_root: /data/samples
        source_name: source.tif
        target_name: target.tif
        save_masks: "false"
    """)
    yaml_file = tmp_path / "str_bool.yaml"
    yaml_file.write_text(yaml_content, encoding="utf-8")
    with pytest.raises(TypeError, match="save_masks"):
        load_preprocessing_config(yaml_file)
