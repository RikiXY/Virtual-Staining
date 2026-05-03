from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from virtual_staining.data.builder import DatasetBuilder
from virtual_staining.data.config import PreprocessingConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_image(seed: int = 0) -> np.ndarray:
    """Return a 600x600 random-noise BGR image with no near-white pixels."""
    rng = np.random.default_rng(seed)
    return rng.integers(10, 200, (600, 600, 3), dtype=np.uint8)


def _white_mask(img: np.ndarray, _params) -> np.ndarray:
    """Full-white mask - every pixel is foreground."""
    return np.full((img.shape[0], img.shape[1]), 255, dtype=np.uint8)


def _identity_align(
    src: np.ndarray,
    tgt: np.ndarray,
    mask1: np.ndarray | None = None,
    mask2: np.ndarray | None = None,
    scale: float = 0.5,
    **_kwargs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return target unchanged - simulates a perfect identity alignment."""
    aligned_mask = (
        mask2.copy()
        if mask2 is not None
        else np.full(tgt.shape[:2], 255, dtype=np.uint8)
    )
    return tgt.copy(), aligned_mask, np.eye(2, 3, dtype=np.float64)


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def builder_config(tmp_path: Path) -> PreprocessingConfig:
    root = tmp_path / "data"
    root.mkdir()

    cv2.imwrite(str(root / "source.png"), _make_synthetic_image(seed=0))
    cv2.imwrite(str(root / "target.png"), _make_synthetic_image(seed=1))

    return PreprocessingConfig(
        dataset_root=root,
        source_name="source.png",
        target_name="target.png",
        image_size=(64, 64),
        grid_movement=(64, 64),
        margin=0,
        seed=42,
        save_masks=False,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        min_foreground_ratio=0.0,
        max_white_ratio=1.0,
        white_threshold=250,
        max_largest_white_component_ratio=1.0,
    )


_PATCHES = [
    patch("virtual_staining.data.builder.calculate_mask_with_multiple_parameters", side_effect=_white_mask),
    patch("virtual_staining.data.builder.align_from_scaled", side_effect=_identity_align),
]


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------

def test_run_all_creates_split_directories(builder_config: PreprocessingConfig) -> None:
    with _PATCHES[0], _PATCHES[1]:
        DatasetBuilder(builder_config).run_all()

    root = builder_config.dataset_root
    assert (root / "dataset_train").exists()
    assert (root / "dataset_val").exists()
    assert (root / "dataset_test").exists()


def test_run_all_result_counts_match_saved_files(builder_config: PreprocessingConfig) -> None:
    with _PATCHES[0], _PATCHES[1]:
        result = DatasetBuilder(builder_config).run_all()

    root = builder_config.dataset_root
    assert result.output_root == root

    total_valid = result.train_count + result.val_count + result.test_count
    assert total_valid > 0

    # Each split dir must contain exactly 2 files per pair (source + target).
    for split_name, count in [
        ("dataset_train", result.train_count),
        ("dataset_val", result.val_count),
        ("dataset_test", result.test_count),
    ]:
        files = list((root / split_name).iterdir())
        assert len(files) == count * 2, f"{split_name}: expected {count * 2} files, got {len(files)}"


def test_run_all_discarded_log_is_written(builder_config: PreprocessingConfig) -> None:
    with _PATCHES[0], _PATCHES[1]:
        result = DatasetBuilder(builder_config).run_all()

    log = builder_config.dataset_root / "discarded_patches" / "discarded_log.csv"
    assert log.exists()
    lines = log.read_text().splitlines()
    # Header + one row per discarded patch
    assert len(lines) == result.skipped_count + 1


def test_missing_dataset_root_raises(tmp_path: Path) -> None:
    config = PreprocessingConfig(
        dataset_root=tmp_path / "nonexistent",
        source_name="source.png",
        target_name="target.png",
    )
    with pytest.raises(FileNotFoundError):
        DatasetBuilder(config).compute_masks()


def test_align_requires_masks(builder_config: PreprocessingConfig) -> None:
    builder = DatasetBuilder(builder_config)
    with pytest.raises(RuntimeError, match="compute_masks"):
        builder.align()


def test_extract_patches_requires_align(builder_config: PreprocessingConfig) -> None:
    builder = DatasetBuilder(builder_config)
    with pytest.raises(RuntimeError, match="align"):
        builder.extract_patches()


def test_filter_patches_requires_extract(builder_config: PreprocessingConfig) -> None:
    builder = DatasetBuilder(builder_config)
    with pytest.raises(RuntimeError, match="extract_patches"):
        builder.filter_patches()


def test_split_and_save_requires_filter(builder_config: PreprocessingConfig) -> None:
    builder = DatasetBuilder(builder_config)
    with pytest.raises(RuntimeError, match="filter_patches"):
        builder.split_and_save()
