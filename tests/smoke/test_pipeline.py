from __future__ import annotations

import csv
import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from tests.config_helpers import write_queue_config, write_run_config
from virtual_staining.applications.complete_run import complete_run
from virtual_staining.applications.evaluate import evaluate
from virtual_staining.applications.infer import infer
from virtual_staining.applications.prepare import prepare
from virtual_staining.applications.run_queue import run_queue
from virtual_staining.applications.train import train
from virtual_staining.config.run import RunConfig
from virtual_staining.data.preprocessing import AlignmentMetadata


def _make_synthetic_dataset(dataset_root: Path, size: int = 192) -> Path:
    """Write a deterministic paired source/target TIFF dataset for smoke testing."""
    dataset_root.mkdir(parents=True, exist_ok=True)

    y, x = np.indices((size, size), dtype=np.uint16)
    base = ((x * 3 + y * 5) % 180).astype(np.uint8)
    source = np.stack(
        [
            base,
            ((base.astype(np.uint16) + 20) % 200).astype(np.uint8),
            ((x + y) % 170).astype(np.uint8),
        ],
        axis=-1,
    )
    target = np.clip(source.astype(np.int16) + np.array([12, 6, 18], dtype=np.int16), 0, 255)
    cv2.imwrite(str(dataset_root / "source.tif"), source)
    cv2.imwrite(str(dataset_root / "target.tif"), target.astype(np.uint8))
    return dataset_root


def _white_mask(img: np.ndarray, _params: object) -> np.ndarray:
    """Treat the full synthetic image as foreground to keep the smoke test deterministic."""
    return np.full((img.shape[0], img.shape[1]), 255, dtype=np.uint8)


def _identity_align(
    _src: np.ndarray,
    tgt: np.ndarray,
    mask_1: np.ndarray | None = None,
    mask_2: np.ndarray | None = None,
    scale: float = 0.5,
    **_kwargs: object,
) -> tuple[np.ndarray, AlignmentMetadata]:
    """Return an identity matrix with valid alignment metadata."""
    del _src, tgt, mask_1, mask_2, scale
    eye = np.eye(2, 3, dtype=np.float64)
    metadata = AlignmentMetadata(
        n_keypoints_src=100,
        n_keypoints_tgt=100,
        n_matches=50,
        n_inliers=45,
        inlier_ratio=0.9,
        scale_x=1.0,
        scale_y=1.0,
        rotation_deg=0.0,
        translation_x=0.0,
        translation_y=0.0,
        warp_matrix=eye.tolist(),
    )
    return eye, metadata


@contextmanager
def _patched_prepare_dependencies() -> Iterator[None]:
    with (
        patch(
            "virtual_staining.data.builder.calculate_mask_with_multiple_parameters",
            side_effect=_white_mask,
        ),
        patch(
            "virtual_staining.data.builder.estimate_affine_from_scaled",
            side_effect=_identity_align,
        ),
    ):
        yield


def _write_smoke_config(tmp_path: Path, dataset_root: Path, *, run_name: str = "smoke_run") -> Path:
    return write_run_config(
        tmp_path,
        """\
        image_size: [64, 64]

        preprocessing:
          source_name: source.tif
          target_name: target.tif
          image_size: [64, 64]
          grid_movement: [64, 64]
          margin: 0
          seed: 42
          train_ratio: 0.6
          val_ratio: 0.2
          test_ratio: 0.2
          min_foreground_ratio: 0.0
          max_white_ratio: 1.0
          white_threshold: 250
          max_largest_white_component_ratio: 1.0

        model:
          generator:
            base_channels: 16
          discriminator:
            ndf: 16

        training:
          batch_size: 4
          epochs: 1
          seed: 42
          num_workers: 0
          validate_rate: 1
          checkpoint_rate: 1
          log_rate: 1

        inference:
          checkpoint_policy: latest

        evaluation:
          save_graphs: false
        """,
        filename=f"{run_name}.yaml",
        dataset_root=dataset_root,
        results_path=tmp_path / "runs",
        run_name=run_name,
    )


def _write_queue_file(
    tmp_path: Path,
    config_paths: list[Path],
    *,
    continue_on_failure: bool,
) -> Path:
    jobs = "\n".join(f"  - config_path: {config_path}" for config_path in config_paths)
    return write_queue_config(
        tmp_path,
        jobs,
        continue_on_failure=continue_on_failure,
    )


@pytest.mark.slow
def test_full_pipeline_smoke(tmp_path: Path) -> None:
    dataset_root = _make_synthetic_dataset(tmp_path / "dataset")
    config_path = _write_smoke_config(tmp_path, dataset_root)
    config = RunConfig.from_yaml(config_path)

    with _patched_prepare_dependencies():
        prepared = prepare(config, config_path)

    assert prepared.train_count > 0
    assert prepared.val_count > 0
    assert prepared.test_count > 0

    train(config, config_path)
    infer(config, config_path)
    evaluate(config, config_path)

    metrics_csv = tmp_path / "runs" / "smoke_run" / "evaluation" / "per_image_metrics.csv"
    assert metrics_csv.exists()

    with metrics_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert rows
    assert rows[0]["sample_id"]


@pytest.mark.slow
def test_complete_run_smoke(tmp_path: Path) -> None:
    dataset_root = _make_synthetic_dataset(tmp_path / "dataset")
    config_path = _write_smoke_config(tmp_path, dataset_root)
    config = RunConfig.from_yaml(config_path)

    with _patched_prepare_dependencies():
        complete_run(config, config_path)

    metrics_csv = tmp_path / "runs" / "smoke_run" / "evaluation" / "per_image_metrics.csv"
    assert metrics_csv.exists()

    with metrics_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert rows
    assert rows[0]["sample_id"]


def test_prepare_smoke_reuses_cached_dataset(tmp_path: Path) -> None:
    dataset_root = _make_synthetic_dataset(tmp_path / "dataset")
    config_path = _write_smoke_config(tmp_path, dataset_root)
    config = RunConfig.from_yaml(config_path)

    with _patched_prepare_dependencies():
        first = prepare(config, config_path)

    with patch(
        "virtual_staining.data.builder.DatasetBuilder.run_all",
        side_effect=AssertionError("prepare should reuse the prepared dataset"),
    ):
        second = prepare(config, config_path)

    stage_data = json.loads(
        (dataset_root / "metadata" / "stages" / "prepare.json").read_text(encoding="utf-8")
    )

    assert first.reused is False
    assert second.reused is True
    assert second.train_count == first.train_count
    assert second.val_count == first.val_count
    assert second.test_count == first.test_count
    assert stage_data["reused"] is True


@pytest.mark.slow
def test_complete_run_smoke_reuses_cached_dataset(tmp_path: Path) -> None:
    dataset_root = _make_synthetic_dataset(tmp_path / "dataset")
    config_path = _write_smoke_config(tmp_path, dataset_root)
    config = RunConfig.from_yaml(config_path)

    with _patched_prepare_dependencies():
        complete_run(config, config_path)

    with patch(
        "virtual_staining.data.builder.DatasetBuilder.run_all",
        side_effect=AssertionError("complete_run should reuse the prepared dataset"),
    ):
        complete_run(config, config_path)

    stage_data = json.loads(
        (dataset_root / "metadata" / "stages" / "prepare.json").read_text(encoding="utf-8")
    )
    assert stage_data["reused"] is True


@pytest.mark.slow
def test_run_queue_smoke_executes_multiple_full_runs(tmp_path: Path) -> None:
    dataset_a = _make_synthetic_dataset(tmp_path / "dataset_a")
    dataset_b = _make_synthetic_dataset(tmp_path / "dataset_b")
    config_a = _write_smoke_config(tmp_path, dataset_a, run_name="queue_run_a")
    config_b = _write_smoke_config(tmp_path, dataset_b, run_name="queue_run_b")
    queue_path = _write_queue_file(tmp_path, [config_a, config_b], continue_on_failure=False)

    with _patched_prepare_dependencies():
        state = run_queue(queue_path)

    state_data = json.loads(
        (tmp_path / "local_workspace" / "queues" / "nightly.state.json").read_text(encoding="utf-8")
    )
    assert state.status == "completed"
    assert [job["status"] for job in state_data["jobs"]] == ["completed", "completed"]
    assert (tmp_path / "runs" / "queue_run_a" / "evaluation" / "per_image_metrics.csv").exists()
    assert (tmp_path / "runs" / "queue_run_b" / "evaluation" / "per_image_metrics.csv").exists()
