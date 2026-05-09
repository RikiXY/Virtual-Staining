from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from virtual_staining.data.builder import DatasetBuilder
from virtual_staining.data.config import PreprocessingConfig
from virtual_staining.data.preprocessing import AlignmentMetadata

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_synthetic_image(seed: int = 0) -> np.ndarray:
    """Return a 600x600 random-noise BGR image with no near-white pixels."""
    rng = np.random.default_rng(seed)
    return rng.integers(10, 200, (600, 600, 3), dtype=np.uint8)


def _white_mask(img: np.ndarray, _params: object) -> np.ndarray:
    """Full-white mask - every pixel is foreground."""
    return np.full((img.shape[0], img.shape[1]), 255, dtype=np.uint8)


def _identity_align(
    src: np.ndarray,
    tgt: np.ndarray,
    mask1: np.ndarray | None = None,
    mask2: np.ndarray | None = None,
    scale: float = 0.5,
    **_kwargs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, AlignmentMetadata]:
    """Return target unchanged - simulates a perfect identity alignment."""
    aligned_mask = (
        mask2.copy() if mask2 is not None else np.full(tgt.shape[:2], 255, dtype=np.uint8)
    )
    eye = np.eye(2, 3, dtype=np.float64)
    metadata = AlignmentMetadata(
        n_keypoints_src=100,
        n_keypoints_tgt=100,
        n_matches=50,
        n_inliers=45,
        warp_matrix=eye.tolist(),
    )
    return tgt.copy(), aligned_mask, eye, metadata


@contextmanager
def _patched_builder_dependencies() -> Iterator[None]:
    with (
        patch(
            "virtual_staining.data.builder.calculate_mask_with_multiple_parameters",
            side_effect=_white_mask,
        ),
        patch(
            "virtual_staining.data.builder.align_from_scaled",
            side_effect=_identity_align,
        ),
    ):
        yield


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


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------


def test_run_all_creates_split_directories(builder_config: PreprocessingConfig) -> None:
    with _patched_builder_dependencies():
        DatasetBuilder(builder_config).run_all()

    root = builder_config.dataset_root
    assert (root / "dataset_train").exists()
    assert (root / "dataset_val").exists()
    assert (root / "dataset_test").exists()


def test_run_all_result_counts_match_saved_files(
    builder_config: PreprocessingConfig,
) -> None:
    with _patched_builder_dependencies():
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
        expected_file_count = count * 2
        assert len(files) == expected_file_count, (
            f"{split_name}: expected {expected_file_count} files, got {len(files)}"
        )


def test_run_all_discarded_log_is_written(builder_config: PreprocessingConfig) -> None:
    with _patched_builder_dependencies():
        result = DatasetBuilder(builder_config).run_all()

    log = builder_config.dataset_root / "discarded_patches" / "discarded_log.csv"
    assert log.exists()
    lines = log.read_text(encoding="utf-8").splitlines()
    # Header + one row per discarded patch
    assert len(lines) == result.skipped_count + 1


def test_run_all_saves_config_and_environment(
    builder_config: PreprocessingConfig,
) -> None:
    with _patched_builder_dependencies():
        DatasetBuilder(builder_config).run_all()

    root = builder_config.dataset_root
    assert (root / "config.yaml").exists()
    assert (root / "environment.json").exists()

    loaded = PreprocessingConfig.from_yaml(root / "config.yaml")
    assert loaded == builder_config


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


# ---------------------------------------------------------------------------
# split_manifest.csv tests
# ---------------------------------------------------------------------------


def test_split_manifest_is_written(builder_config: PreprocessingConfig) -> None:
    with _patched_builder_dependencies():
        DatasetBuilder(builder_config).run_all()

    manifest = builder_config.dataset_root / "split_manifest.csv"
    assert manifest.exists()


def test_split_manifest_row_count_equals_valid_pairs(
    builder_config: PreprocessingConfig,
) -> None:
    import csv as csv_module

    with _patched_builder_dependencies():
        result = DatasetBuilder(builder_config).run_all()

    manifest = builder_config.dataset_root / "split_manifest.csv"
    with manifest.open(encoding="utf-8", newline="") as f:
        rows = list(csv_module.DictReader(f))

    expected = result.train_count + result.val_count + result.test_count
    assert len(rows) == expected


def test_split_manifest_split_values(builder_config: PreprocessingConfig) -> None:
    import csv as csv_module

    with _patched_builder_dependencies():
        DatasetBuilder(builder_config).run_all()

    manifest = builder_config.dataset_root / "split_manifest.csv"
    with manifest.open(encoding="utf-8", newline="") as f:
        rows = list(csv_module.DictReader(f))

    allowed = {"train", "val", "test"}
    assert all(r["split"] in allowed for r in rows)


def test_split_manifest_columns_and_coordinates(
    builder_config: PreprocessingConfig,
) -> None:
    import csv as csv_module

    with _patched_builder_dependencies():
        DatasetBuilder(builder_config).run_all()

    manifest = builder_config.dataset_root / "split_manifest.csv"
    with manifest.open(encoding="utf-8", newline="") as f:
        reader = csv_module.DictReader(f)
        rows = list(reader)
        assert reader.fieldnames is not None
        assert set(reader.fieldnames) >= {
            "sample_id",
            "split",
            "source_name",
            "target_name",
            "x",
            "y",
        }

    for row in rows:
        x, y = int(row["x"]), int(row["y"])
        assert row["sample_id"] == f"{x:05}_{y:05}"
        assert row["source_name"].startswith(f"{x:05}_{y:05}_source")
        assert row["target_name"].startswith(f"{x:05}_{y:05}_target")


# ---------------------------------------------------------------------------
# alignment_metadata.json tests
# ---------------------------------------------------------------------------


def test_run_all_saves_alignment_metadata(builder_config: PreprocessingConfig) -> None:
    import json

    with _patched_builder_dependencies():
        DatasetBuilder(builder_config).run_all()

    metadata_file = builder_config.dataset_root / "alignment_metadata.json"
    assert metadata_file.exists()

    data = json.loads(metadata_file.read_text(encoding="utf-8"))
    assert set(data.keys()) >= {
        "n_keypoints_src",
        "n_keypoints_tgt",
        "n_matches",
        "n_inliers",
        "warp_matrix",
    }
