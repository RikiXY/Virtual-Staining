from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from virtual_staining.config.run import RunConfig
from virtual_staining.data.builder import DatasetBuilder
from virtual_staining.data.results import DatasetBuildResult
from virtual_staining.experiment.metadata import (
    append_run_event,
    ensure_run_metadata,
    save_stage_metadata,
)
from virtual_staining.experiment.snapshots import (
    compute_manifest_hash,
    resolve_prepare_snapshot_paths,
    save_environment_snapshot,
    save_stage_config_snapshots,
)


def prepare(config: RunConfig, config_path: Path) -> DatasetBuildResult:
    """Application-level dataset preparation entry point."""
    if config.preprocessing is None:
        raise ValueError("RunConfig.preprocessing must be present for prepare().")

    dataset_root = config.preprocessing.dataset_root
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    snapshot_paths = resolve_prepare_snapshot_paths(dataset_root)
    metadata_dir = dataset_root / "metadata"

    config_hash = save_stage_config_snapshots(
        config,
        config_path,
        input_dest=snapshot_paths.input_config,
        resolved_dest=snapshot_paths.resolved_config,
        hash_dest=snapshot_paths.config_hash,
    )
    save_environment_snapshot(snapshot_paths.environment)
    ensure_run_metadata(
        metadata_dir / "run.json",
        run_name=config.project.run_name,
        entrypoint="vs-prepare",
        config_hash=config_hash,
    )

    started_at = datetime.now(UTC).isoformat()
    save_stage_metadata(
        "prepare",
        {
            "stage": "prepare",
            "status": "running",
            "started_at": started_at,
            "completed_at": None,
            "config_hash": config_hash,
            "dataset_root": str(dataset_root),
        },
        metadata_dir,
    )
    append_run_event(
        {
            "timestamp": started_at,
            "run_name": config.project.run_name,
            "stage": "prepare",
            "event_type": "stage_started",
            "status": "running",
            "config_hash": config_hash,
            "details": {"dataset_root": str(dataset_root)},
        },
        metadata_dir,
    )

    builder = DatasetBuilder(config.preprocessing)
    try:
        result = builder.run_all()
    except Exception as exc:
        completed_at = datetime.now(UTC).isoformat()
        save_stage_metadata(
            "prepare",
            {
                "stage": "prepare",
                "status": "failed",
                "started_at": started_at,
                "completed_at": completed_at,
                "config_hash": config_hash,
                "dataset_root": str(dataset_root),
                "error": str(exc),
            },
            metadata_dir,
        )
        append_run_event(
            {
                "timestamp": completed_at,
                "run_name": config.project.run_name,
                "stage": "prepare",
                "event_type": "stage_failed",
                "status": "failed",
                "config_hash": config_hash,
                "details": {"dataset_root": str(dataset_root), "error": str(exc)},
            },
            metadata_dir,
        )
        raise

    manifest_path = dataset_root / "manifests" / "manifest.csv"
    completed_at = datetime.now(UTC).isoformat()
    details = {
        "dataset_root": str(dataset_root),
        "manifest_path": str(manifest_path),
        "manifest_sha256": compute_manifest_hash(manifest_path),
        "train_count": result.train_count,
        "val_count": result.val_count,
        "test_count": result.test_count,
        "skipped_count": result.skipped_count,
    }
    save_stage_metadata(
        "prepare",
        {
            "stage": "prepare",
            "status": "completed",
            "started_at": started_at,
            "completed_at": completed_at,
            "config_hash": config_hash,
            **details,
        },
        metadata_dir,
    )
    append_run_event(
        {
            "timestamp": completed_at,
            "run_name": config.project.run_name,
            "stage": "prepare",
            "event_type": "stage_completed",
            "status": "completed",
            "config_hash": config_hash,
            "details": details,
        },
        metadata_dir,
    )
    return result
