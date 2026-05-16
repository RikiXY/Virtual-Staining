from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from tests.config_helpers import write_run_config
from virtual_staining.applications.prepare import prepare
from virtual_staining.config.run import RunConfig
from virtual_staining.data.preprocessing import AlignmentMetadata
from virtual_staining.experiment.metadata import (
    append_run_event,
    ensure_run_metadata,
    save_stage_metadata,
)


def _white_mask(img: np.ndarray, _params: object) -> np.ndarray:
    return np.full((img.shape[0], img.shape[1]), 255, dtype=np.uint8)


def _identity_align(
    _src: np.ndarray,
    tgt: np.ndarray,
    mask_1: np.ndarray | None = None,
    mask_2: np.ndarray | None = None,
    scale: float = 0.5,
    **_kwargs: object,
) -> tuple[np.ndarray, AlignmentMetadata]:
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


def test_stage_metadata_overwrites_current_state_and_events_append(tmp_path: Path) -> None:
    metadata_dir = tmp_path / "metadata"
    ensure_run_metadata(metadata_dir / "run.json", run_name="demo", entrypoint="vs-train")

    save_stage_metadata(
        "infer",
        {"stage": "infer", "status": "running", "started_at": "2026-01-01T00:00:00+00:00"},
        metadata_dir,
    )
    append_run_event(
        {
            "timestamp": "2026-01-01T00:00:00+00:00",
            "run_name": "demo",
            "stage": "infer",
            "event_type": "stage_started",
            "status": "running",
            "config_hash": "sha256:a",
            "details": {"attempt": 1},
        },
        metadata_dir,
    )

    save_stage_metadata(
        "infer",
        {
            "stage": "infer",
            "status": "completed",
            "started_at": "2026-01-01T00:00:00+00:00",
            "completed_at": "2026-01-01T00:01:00+00:00",
        },
        metadata_dir,
    )
    append_run_event(
        {
            "timestamp": "2026-01-01T00:01:00+00:00",
            "run_name": "demo",
            "stage": "infer",
            "event_type": "stage_completed",
            "status": "completed",
            "config_hash": "sha256:a",
            "details": {"attempt": 1},
        },
        metadata_dir,
    )

    stage_data = json.loads((metadata_dir / "stages" / "infer.json").read_text(encoding="utf-8"))
    assert stage_data["status"] == "completed"

    events = [
        json.loads(line)
        for line in (metadata_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(events) == 2
    assert [event["event_type"] for event in events] == ["stage_started", "stage_completed"]

    run_data = json.loads((metadata_dir / "run.json").read_text(encoding="utf-8"))
    assert run_data["stages_present"] == ["infer"]
    assert run_data["last_completed_stage"] == "infer"
    assert run_data["last_event_at"] == "2026-01-01T00:01:00+00:00"


def test_prepare_writes_stage_metadata_and_events(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    image = np.full((192, 192, 3), 64, dtype=np.uint8)
    cv2.imwrite(str(dataset_root / "source.png"), image)
    cv2.imwrite(str(dataset_root / "target.png"), image + 8)

    config_path = write_run_config(
        tmp_path,
        """\
        image_size: [64, 64]
        preprocessing:
          source_name: source.png
          target_name: target.png
          image_size: [64, 64]
          grid_movement: [64, 64]
          margin: 0
          seed: 42
          train_ratio: 0.8
          val_ratio: 0.1
          test_ratio: 0.1
          min_foreground_ratio: 0.0
          max_white_ratio: 1.0
          white_threshold: 250
          max_largest_white_component_ratio: 1.0
        """,
        filename="prepare.yaml",
        dataset_root=dataset_root,
        results_path=tmp_path / "results",
        run_name="prepare_run",
    )
    config = RunConfig.from_yaml(config_path)

    with _patched_prepare_dependencies():
        prepare(config, config_path)

    metadata_dir = dataset_root / "metadata"
    run_data = json.loads((metadata_dir / "run.json").read_text(encoding="utf-8"))
    stage_data = json.loads((metadata_dir / "stages" / "prepare.json").read_text(encoding="utf-8"))
    fingerprint_data = json.loads(
        (metadata_dir / "dataset_fingerprint.json").read_text(encoding="utf-8")
    )
    events = [
        json.loads(line)
        for line in (metadata_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert run_data["stages_present"] == ["prepare"]
    assert run_data["last_completed_stage"] == "prepare"
    assert stage_data["stage"] == "prepare"
    assert stage_data["status"] == "completed"
    assert stage_data["reused"] is False
    assert stage_data["manifest_path"].endswith("manifests/manifest.csv")
    assert fingerprint_data["fingerprint"].startswith("sha256:")
    assert fingerprint_data["source"]["path"] == str((dataset_root / "source.png").resolve())
    assert fingerprint_data["target"]["path"] == str((dataset_root / "target.png").resolve())
    assert [event["event_type"] for event in events] == ["stage_started", "stage_completed"]
