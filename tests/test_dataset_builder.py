from __future__ import annotations

import builtins
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from virtual_staining.applications.prepare import prepare
from virtual_staining.config.run import RunConfig
from virtual_staining.data import builder as builder_module
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


@pytest.fixture()
def builder_scaled_config(tmp_path: Path) -> PreprocessingConfig:
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
        mask_scale=0.25,
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
    assert (root / "splits" / "train").exists()
    assert (root / "splits" / "val").exists()
    assert (root / "splits" / "test").exists()
    assert not (root / "dataset_train").exists()
    assert not (root / "dataset_val").exists()
    assert not (root / "dataset_test").exists()


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
        ("train", result.train_count),
        ("val", result.val_count),
        ("test", result.test_count),
    ]:
        files = list((root / "splits" / split_name).iterdir())
        expected_file_count = count * 2
        assert len(files) == expected_file_count, (
            f"{split_name}: expected {expected_file_count} files, got {len(files)}"
        )


def test_builder_logs_stage_progress(
    builder_config: PreprocessingConfig, caplog: pytest.LogCaptureFixture
) -> None:
    with (
        _patched_builder_dependencies(),
        caplog.at_level(logging.INFO, logger="virtual_staining.data.builder"),
    ):
        DatasetBuilder(builder_config).run_all()

    messages = [record.message for record in caplog.records]
    assert any(message == "Seed set to 42" for message in messages)
    assert any(message == "Calculating masks..." for message in messages)
    assert any(message == "Aligning images..." for message in messages)
    assert any("patch pairs" in message for message in messages)
    assert any(message.startswith("Saved: train=") for message in messages)


def test_compute_masks_with_mask_scale(
    builder_scaled_config: PreprocessingConfig,
) -> None:
    builder = DatasetBuilder(builder_scaled_config)
    source_calls: list[tuple[int, int, int]] = []
    target_calls: list[tuple[int, int, int]] = []

    def _record_mask_shape(img: np.ndarray, _params: object) -> np.ndarray:
        if not source_calls:
            source_calls.append(img.shape)
        else:
            target_calls.append(img.shape)
        return np.full(img.shape[:2], 255, dtype=np.uint8)

    with patch(
        "virtual_staining.data.builder.calculate_mask_with_multiple_parameters",
        side_effect=_record_mask_shape,
    ):
        builder.compute_masks()

    assert source_calls == [(150, 150, 3)]
    assert target_calls == [(150, 150, 3)]
    assert builder._source_image is not None
    assert builder._target_image is not None
    assert builder._source_mask is not None
    assert builder._target_mask is not None
    assert builder._source_mask.shape == builder._source_image.shape[:2]
    assert builder._target_mask.shape == builder._target_image.shape[:2]


def test_compute_masks_raises_if_estimated_memory_exceeds_limit(
    builder_config: PreprocessingConfig,
) -> None:
    config = PreprocessingConfig(
        dataset_root=builder_config.dataset_root,
        source_name=builder_config.source_name,
        target_name=builder_config.target_name,
        image_size=builder_config.image_size,
        grid_movement=builder_config.grid_movement,
        margin=builder_config.margin,
        seed=builder_config.seed,
        save_masks=builder_config.save_masks,
        mask_scale=builder_config.mask_scale,
        max_memory_gb=0.0001,
        train_ratio=builder_config.train_ratio,
        val_ratio=builder_config.val_ratio,
        test_ratio=builder_config.test_ratio,
        min_foreground_ratio=builder_config.min_foreground_ratio,
        max_white_ratio=builder_config.max_white_ratio,
        white_threshold=builder_config.white_threshold,
        max_largest_white_component_ratio=builder_config.max_largest_white_component_ratio,
    )
    builder = DatasetBuilder(config)

    with (
        patch(
            "virtual_staining.data.builder.cv2.imread",
            side_effect=AssertionError("compute_masks() should fail before image loading"),
        ),
        pytest.raises(MemoryError, match="max_memory_gb") as exc_info,
    ):
        builder.compute_masks()

    assert "mask_scale" in str(exc_info.value)


def test_compute_masks_warns_when_estimate_is_high_without_limit(
    builder_config: PreprocessingConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    builder = DatasetBuilder(builder_config)

    with (
        _patched_builder_dependencies(),
        patch("virtual_staining.data.builder._estimate_memory_gb", return_value=9.0),
        caplog.at_level(logging.WARNING, logger="virtual_staining.data.builder"),
    ):
        builder.compute_masks()

    assert any("mask_scale: 0.25" in message for message in caplog.messages)


def test_estimate_memory_gb_decreases_with_mask_scale() -> None:
    full_scale = builder_module._estimate_memory_gb(6000, 8000, mask_scale=1.0)
    quarter_scale = builder_module._estimate_memory_gb(6000, 8000, mask_scale=0.25)

    assert quarter_scale < full_scale


def test_read_image_size_disables_pillow_bomb_limit_temporarily() -> None:
    original_max_image_pixels = builder_module.Image.MAX_IMAGE_PIXELS

    class _DummyImage:
        size = (123, 456)

        def __enter__(self) -> _DummyImage:
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            return None

    def _open_asserting_limit_disabled(_path: Path) -> _DummyImage:
        assert builder_module.Image.MAX_IMAGE_PIXELS is None
        return _DummyImage()

    with patch(
        "virtual_staining.data.builder.Image.open",
        side_effect=_open_asserting_limit_disabled,
    ):
        size = builder_module._read_image_size(Path("/tmp/fake.tif"))

    assert size == (123, 456)
    assert original_max_image_pixels == builder_module.Image.MAX_IMAGE_PIXELS


def test_builder_logs_memory_after_stages(
    builder_config: PreprocessingConfig, caplog: pytest.LogCaptureFixture
) -> None:
    with (
        _patched_builder_dependencies(),
        caplog.at_level(logging.INFO, logger="virtual_staining.data.builder"),
    ):
        DatasetBuilder(builder_config).run_all()

    memory_messages = [
        record.message for record in caplog.records if "Memory after" in record.message
    ]
    assert len(memory_messages) == 5

    logged_stages = {
        message.removeprefix("Memory after ").split(":", maxsplit=1)[0]
        for message in memory_messages
    }
    assert logged_stages == {
        "compute_masks",
        "align",
        "extract_patches",
        "filter_patches",
        "split_and_save",
    }


def test_dataset_builder_does_not_expose_old_patch_stage_methods(
    builder_config: PreprocessingConfig,
) -> None:
    builder = DatasetBuilder(builder_config)
    assert not hasattr(builder, "extract_patches")
    assert not hasattr(builder, "filter_patches")
    assert not hasattr(builder, "split_and_save")


def test_run_all_produces_same_output_as_old_stages(tmp_path: Path) -> None:
    import csv as csv_module

    def _make_config(dataset_root: Path) -> PreprocessingConfig:
        return PreprocessingConfig(
            dataset_root=dataset_root,
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

    def _write_pair(dataset_root: Path) -> None:
        dataset_root.mkdir()
        cv2.imwrite(str(dataset_root / "source.png"), _make_synthetic_image(seed=0))
        cv2.imwrite(str(dataset_root / "target.png"), _make_synthetic_image(seed=1))

    def _manifest_rows(dataset_root: Path) -> list[dict[str, str]]:
        manifest_path = dataset_root / "manifests" / "manifest.csv"
        with manifest_path.open(encoding="utf-8", newline="") as f:
            reader = csv_module.DictReader(f)
            return sorted(
                list(reader),
                key=lambda row: (
                    row["sample_id"],
                    row["split"],
                    row["input_path"],
                    row["target_path"],
                ),
            )

    streaming_root = tmp_path / "streaming"
    legacy_root = tmp_path / "legacy"
    _write_pair(streaming_root)
    _write_pair(legacy_root)

    streaming_builder = DatasetBuilder(_make_config(streaming_root))
    legacy_builder = DatasetBuilder(_make_config(legacy_root))

    with _patched_builder_dependencies():
        streaming_result = streaming_builder.run_all()

    with _patched_builder_dependencies():
        legacy_builder._started_at = "2026-01-01T00:00:00+00:00"
        legacy_builder._effective_seed = legacy_builder.config.seed
        legacy_builder.compute_masks()
        legacy_builder.align()
        valid_rows, discarded_rows = legacy_builder._stream_patches_to_disk()
        legacy_result = legacy_builder._assign_splits_and_finalize(valid_rows, discarded_rows)

    assert streaming_result.train_count == legacy_result.train_count
    assert streaming_result.val_count == legacy_result.val_count
    assert streaming_result.test_count == legacy_result.test_count
    assert streaming_result.skipped_count == legacy_result.skipped_count
    assert _manifest_rows(streaming_root) == _manifest_rows(legacy_root)
    for split in ("train", "val", "test"):
        streaming_files = sorted(
            path.name for path in (streaming_root / f"dataset_{split}").iterdir()
        )
        legacy_files = sorted(path.name for path in (legacy_root / f"dataset_{split}").iterdir())
        assert streaming_files == legacy_files


def test_stream_patches_to_disk_writes_valid_patch_staging(
    builder_config: PreprocessingConfig,
) -> None:
    builder = DatasetBuilder(builder_config)

    with _patched_builder_dependencies():
        builder.compute_masks()
        builder.align()
        valid_rows, discarded_rows = builder._stream_patches_to_disk()

    assert len(valid_rows) == 81
    assert discarded_rows == []
    assert all(set(row) == {"x", "y", "source", "target"} for row in valid_rows)
    assert all(
        not any(isinstance(value, np.ndarray) for value in row.values()) for row in valid_rows
    )

    root = builder_config.dataset_root
    valid_source_files = list((root / "processed" / "valid" / "source").iterdir())
    valid_target_files = list((root / "processed" / "valid" / "target").iterdir())
    discarded_source_files = list((root / "discarded_patches" / "source").iterdir())
    discarded_target_files = list((root / "discarded_patches" / "target").iterdir())
    assert len(valid_source_files) == 81
    assert len(valid_target_files) == 81
    assert discarded_source_files == []
    assert discarded_target_files == []


def test_stream_patches_to_disk_writes_discarded_patch_staging(
    builder_config: PreprocessingConfig,
) -> None:
    builder = DatasetBuilder(builder_config)
    validation_results = [
        (
            index % 2 == 0,
            {
                "source_foreground_ratio": 1.0,
                "target_foreground_ratio": 1.0,
                "source_white_ratio": 0.0,
                "target_white_ratio": 0.0,
                "source_largest_white_component_ratio": 0.0,
                "target_largest_white_component_ratio": 0.0,
                "reasons": [] if index % 2 == 0 else ["synthetic_rejection"],
            },
        )
        for index in range(81)
    ]

    with (
        _patched_builder_dependencies(),
        patch(
            "virtual_staining.data.builder.is_valid_patch_pair",
            side_effect=validation_results,
        ),
    ):
        builder.compute_masks()
        builder.align()
        valid_rows, discarded_rows = builder._stream_patches_to_disk()

    assert len(valid_rows) + len(discarded_rows) == 81
    assert valid_rows
    assert discarded_rows
    assert all(set(row) == {"x", "y", "source", "target"} for row in valid_rows)
    assert all("reasons" in row for row in discarded_rows)
    assert all(
        {
            "sample_id",
            "source_name",
            "target_name",
            "source_foreground_ratio",
            "target_foreground_ratio",
            "source_white_ratio",
            "target_white_ratio",
            "reasons",
            "source_largest_white_component_ratio",
            "target_largest_white_component_ratio",
        }
        == set(row)
        for row in discarded_rows
    )

    root = builder_config.dataset_root
    valid_source_files = list((root / "processed" / "valid" / "source").iterdir())
    valid_target_files = list((root / "processed" / "valid" / "target").iterdir())
    discarded_source_files = list((root / "discarded_patches" / "source").iterdir())
    discarded_target_files = list((root / "discarded_patches" / "target").iterdir())
    assert len(valid_source_files) == len(valid_rows)
    assert len(valid_target_files) == len(valid_rows)
    assert len(discarded_source_files) == len(discarded_rows)
    assert len(discarded_target_files) == len(discarded_rows)


def test_assign_splits_and_finalize_moves_staged_files_and_writes_manifest(
    builder_config: PreprocessingConfig,
) -> None:
    builder = DatasetBuilder(builder_config)
    validation_results = [
        (
            index % 2 == 0,
            {
                "source_foreground_ratio": 1.0,
                "target_foreground_ratio": 1.0,
                "source_white_ratio": 0.0,
                "target_white_ratio": 0.0,
                "source_largest_white_component_ratio": 0.0,
                "target_largest_white_component_ratio": 0.0,
                "reasons": [] if index % 2 == 0 else ["synthetic_rejection"],
            },
        )
        for index in range(81)
    ]

    with (
        _patched_builder_dependencies(),
        patch(
            "virtual_staining.data.builder.is_valid_patch_pair",
            side_effect=validation_results,
        ),
    ):
        builder.compute_masks()
        builder.align()
        builder._started_at = "2026-01-01T00:00:00+00:00"
        builder._effective_seed = builder_config.seed
        valid_rows, discarded_rows = builder._stream_patches_to_disk()
        result = builder._assign_splits_and_finalize(valid_rows, discarded_rows)

    assert result.train_count + result.val_count + result.test_count == len(valid_rows)
    assert result.skipped_count == len(discarded_rows)

    root = builder_config.dataset_root
    assert list((root / "processed" / "valid" / "source").iterdir()) == []
    assert list((root / "processed" / "valid" / "target").iterdir()) == []

    train_files = list((root / "splits" / "train").iterdir())
    val_files = list((root / "splits" / "val").iterdir())
    test_files = list((root / "splits" / "test").iterdir())
    assert len(train_files) == result.train_count * 2
    assert len(val_files) == result.val_count * 2
    assert len(test_files) == result.test_count * 2

    manifest = builder_config.dataset_root / "manifests" / "manifest.csv"
    discarded_manifest = builder_config.dataset_root / "manifests" / "discarded_manifest.csv"
    discarded_log = builder_config.dataset_root / "discarded_patches" / "discarded_log.csv"
    assert manifest.exists()
    assert discarded_manifest.exists()
    assert discarded_log.exists()

    import csv as csv_module

    with manifest.open(encoding="utf-8", newline="") as f:
        reader = csv_module.DictReader(f)
        rows = list(reader)
    assert len(rows) == len(valid_rows)
    assert {row["split"] for row in rows} <= {"train", "val", "test"}
    assert {row["input_path"].split("/")[0] for row in rows} == {"splits"}


def test_assign_splits_and_finalize_is_deterministic_for_fixed_seed(
    tmp_path: Path,
) -> None:
    import csv as csv_module

    def _run_finalize(dataset_root: Path) -> dict[str, str]:
        config = PreprocessingConfig(
            dataset_root=dataset_root,
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
        dataset_root.mkdir()
        cv2.imwrite(str(dataset_root / "source.png"), _make_synthetic_image(seed=0))
        cv2.imwrite(str(dataset_root / "target.png"), _make_synthetic_image(seed=1))

        builder = DatasetBuilder(config)
        with _patched_builder_dependencies():
            builder.compute_masks()
            builder.align()
            builder._started_at = "2026-01-01T00:00:00+00:00"
            builder._effective_seed = config.seed
            valid_rows, discarded_rows = builder._stream_patches_to_disk()
            builder._assign_splits_and_finalize(valid_rows, discarded_rows)

        manifest = dataset_root / "manifests" / "manifest.csv"
        with manifest.open(encoding="utf-8", newline="") as f:
            reader = csv_module.DictReader(f)
            return {row["sample_id"]: row["split"] for row in reader}

    first_assignment = _run_finalize(tmp_path / "run_one")
    second_assignment = _run_finalize(tmp_path / "run_two")

    assert first_assignment == second_assignment


def test_log_memory_handles_missing_resource(caplog: pytest.LogCaptureFixture) -> None:
    original_import = builtins.__import__

    def _import_with_missing_resource(
        name: str,
        globals: dict[str, object] | None = None,
        locals: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if name == "resource":
            raise ImportError("resource unavailable")
        return original_import(name, globals, locals, fromlist, level)

    with (
        patch("builtins.__import__", side_effect=_import_with_missing_resource),
        caplog.at_level(logging.INFO, logger="virtual_staining.data.builder"),
    ):
        builder_module._log_memory("compute_masks")

    assert "Memory after compute_masks: (not available on this platform)" in caplog.messages


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
        assert row["input_path"].startswith(f"splits/{row['split']}/")
        assert row["target_path"].startswith(f"splits/{row['split']}/")
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


def test_stream_patches_to_disk_requires_align(builder_config: PreprocessingConfig) -> None:
    builder = DatasetBuilder(builder_config)
    with pytest.raises(RuntimeError, match="align"):
        builder._stream_patches_to_disk()


def test_assign_splits_and_finalize_accepts_empty_rows(
    builder_config: PreprocessingConfig,
) -> None:
    builder = DatasetBuilder(builder_config)
    builder._started_at = "2026-01-01T00:00:00+00:00"
    builder._effective_seed = builder_config.seed
    result = builder._assign_splits_and_finalize([], [])

    assert result.train_count == 0
    assert result.val_count == 0
    assert result.test_count == 0
    assert result.skipped_count == 0


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


def test_stream_patches_to_disk_raises_on_count_mismatch(
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
            builder._stream_patches_to_disk()
