from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from virtual_staining.applications.prepare import prepare
from virtual_staining.config.run import RunConfig
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


def test_run_all_does_not_create_canonical_snapshot_files(
    builder_config: PreprocessingConfig,
) -> None:
    with _patched_builder_dependencies():
        DatasetBuilder(builder_config).run_all()

    root = builder_config.dataset_root
    assert not (root / "config" / "input.yaml").exists()
    assert not (root / "config" / "resolved.yaml").exists()
    assert not (root / "metadata" / "config_hash.txt").exists()
    assert not (root / "metadata" / "environment.json").exists()
    assert not (root / "config.yaml").exists()
    assert not (root / "environment.json").exists()


def test_run_all_preserves_bootstrapped_snapshot_files(
    builder_config: PreprocessingConfig,
) -> None:
    root = builder_config.dataset_root
    (root / "config").mkdir()
    (root / "metadata").mkdir()
    input_path = root / "config" / "input.yaml"
    resolved_path = root / "config" / "resolved.yaml"
    hash_path = root / "metadata" / "config_hash.txt"
    environment_path = root / "metadata" / "environment.json"
    input_path.write_text("input\n", encoding="utf-8")
    resolved_path.write_text("resolved\n", encoding="utf-8")
    hash_path.write_text("sha256:test\n", encoding="utf-8")
    environment_path.write_text('{"python":"3"}\n', encoding="utf-8")

    with _patched_builder_dependencies():
        DatasetBuilder(builder_config).run_all()

    assert input_path.read_text(encoding="utf-8") == "input\n"
    assert resolved_path.read_text(encoding="utf-8") == "resolved\n"
    assert hash_path.read_text(encoding="utf-8") == "sha256:test\n"
    assert environment_path.read_text(encoding="utf-8") == '{"python":"3"}\n'


def test_prepare_writes_canonical_snapshot_files(
    builder_config: PreprocessingConfig,
) -> None:
    config_path = builder_config.dataset_root.parent / "prepare.yaml"
    config_path.write_text(
        f"""
dataset_root: {builder_config.dataset_root}
results_path: {builder_config.dataset_root.parent / "results"}
run_name: prepare_run
image_size: [{builder_config.image_size[0]}, {builder_config.image_size[1]}]
preprocessing:
  source_name: {builder_config.source_name}
  target_name: {builder_config.target_name}
  grid_movement: [{builder_config.grid_movement[0]}, {builder_config.grid_movement[1]}]
  margin: {builder_config.margin}
  seed: {builder_config.seed}
  save_masks: false
  train_ratio: {builder_config.train_ratio}
  val_ratio: {builder_config.val_ratio}
  test_ratio: {builder_config.test_ratio}
  min_foreground_ratio: {builder_config.min_foreground_ratio}
  max_white_ratio: {builder_config.max_white_ratio}
  white_threshold: {builder_config.white_threshold}
  max_largest_white_component_ratio: {builder_config.max_largest_white_component_ratio}
""".strip()
        + "\n",
        encoding="utf-8",
    )
    run_config = RunConfig.from_yaml(config_path)

    with _patched_builder_dependencies():
        prepare(run_config, config_path)

    root = builder_config.dataset_root
    assert (root / "config" / "input.yaml").exists()
    assert (root / "config" / "resolved.yaml").exists()
    assert (root / "metadata" / "config_hash.txt").exists()
    assert (root / "metadata" / "environment.json").exists()
    assert not (root / "config.yaml").exists()
    assert not (root / "environment.json").exists()

    loaded = RunConfig.from_yaml(root / "config" / "resolved.yaml")
    assert loaded == run_config


def test_run_all_saves_manifest_layout_and_resolved_config(
    builder_config: PreprocessingConfig,
) -> None:
    from virtual_staining.data.manifest import DatasetManifest

    with _patched_builder_dependencies():
        result = DatasetBuilder(builder_config).run_all()

    root = builder_config.dataset_root
    assert (root / "processed").exists()
    assert (root / "splits").exists()
    assert (root / "manifests" / "manifest.csv").exists()
    assert (root / "manifests" / "discarded_manifest.csv").exists()
    assert (root / "metadata" / "dataset_build.json").exists()

    manifest = DatasetManifest.from_csv(root / "manifests" / "manifest.csv", root)
    assert len(manifest) == result.train_count + result.val_count + result.test_count
    assert len(manifest.filter_split("train")) == result.train_count


def test_run_all_writes_manifest_columns_and_relative_paths(
    builder_config: PreprocessingConfig,
) -> None:
    import csv as csv_module

    with _patched_builder_dependencies():
        DatasetBuilder(builder_config).run_all()

    manifest = builder_config.dataset_root / "manifests" / "manifest.csv"
    with manifest.open(encoding="utf-8", newline="") as f:
        reader = csv_module.DictReader(f)
        rows = list(reader)

    assert reader.fieldnames == [
        "sample_id",
        "split",
        "input_path",
        "target_path",
        "input_modality",
        "target_modality",
        "x",
        "y",
        "width",
        "height",
    ]
    assert rows
    for row in rows:
        assert row["split"] in {"train", "val", "test"}
        assert row["input_path"].startswith(f"dataset_{row['split']}/")
        assert row["target_path"].startswith(f"dataset_{row['split']}/")
        assert row["input_modality"] == "label_free"
        assert row["target_modality"] == "stained"
        assert int(row["width"]) == builder_config.image_size[0]
        assert int(row["height"]) == builder_config.image_size[1]


def test_run_all_writes_dataset_build_metadata(builder_config: PreprocessingConfig) -> None:
    import json

    with _patched_builder_dependencies():
        result = DatasetBuilder(builder_config).run_all()

    metadata_path = builder_config.dataset_root / "metadata" / "dataset_build.json"
    data = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert data["dataset_name"] == builder_config.dataset_root.name
    assert data["status"] == "completed"
    assert data["started_at"]
    assert data["completed_at"]
    assert data["num_patches_valid"] == result.train_count + result.val_count + result.test_count
    assert data["num_patches_discarded"] == result.skipped_count
    assert data["num_patches_total"] == data["num_patches_valid"] + data["num_patches_discarded"]
    assert data["num_train"] == result.train_count
    assert data["num_val"] == result.val_count
    assert data["num_test"] == result.test_count
    assert data["seed"] == builder_config.seed


def test_run_all_writes_manifest_metadata(builder_config: PreprocessingConfig) -> None:
    import json

    with _patched_builder_dependencies():
        result = DatasetBuilder(builder_config).run_all()

    metadata_path = builder_config.dataset_root / "manifests" / "manifest_metadata.json"
    data = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert data["schema_version"] == "1.0"
    assert data["created_at"]
    assert data["record_count"] == result.train_count + result.val_count + result.test_count
    assert data["splits"] == {
        "train": result.train_count,
        "val": result.val_count,
        "test": result.test_count,
    }


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


def test_split_manifest_csv_is_not_written(builder_config: PreprocessingConfig) -> None:
    with _patched_builder_dependencies():
        DatasetBuilder(builder_config).run_all()

    manifest = builder_config.dataset_root / "split_manifest.csv"
    assert not manifest.exists()


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


# ---------------------------------------------------------------------------
# Count invariant tests
# ---------------------------------------------------------------------------


def test_extract_patches_raises_on_count_mismatch(
    builder_config: PreprocessingConfig,
) -> None:
    builder = DatasetBuilder(builder_config)
    dummy = np.zeros((64, 64, 3), dtype=np.uint8)

    with _patched_builder_dependencies():
        builder.compute_masks()
        builder.align()

        with (
            patch(
                "virtual_staining.data.builder.divide_image_with_positions",
                return_value=[dummy],
            ),
            pytest.raises(RuntimeError, match="mismatch"),
        ):
            builder.extract_patches()
