from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from virtual_staining.config.run import RunConfig
from virtual_staining.data.builder import DatasetBuilder
from virtual_staining.data.results import DatasetBuildResult
from virtual_staining.experiment.metadata import (
    append_run_event,
    ensure_run_metadata,
    save_stage_metadata,
)
from virtual_staining.experiment.snapshots import (
    build_dataset_fingerprint_metadata,
    compute_manifest_hash,
    resolve_prepare_snapshot_paths,
    save_environment_snapshot,
    save_stage_config_snapshots,
    serialize_preprocessing_config,
)


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _build_current_fingerprint(config: RunConfig) -> dict[str, Any]:
    if config.preprocessing is None:
        raise ValueError("RunConfig.preprocessing must be present for prepare().")
    dataset_root = config.preprocessing.dataset_root
    preprocessing_payload = serialize_preprocessing_config(config.preprocessing)
    return build_dataset_fingerprint_metadata(
        dataset_root=dataset_root,
        preprocessing_config=preprocessing_payload,
        source_path=dataset_root / config.preprocessing.source_name,
        target_path=dataset_root / config.preprocessing.target_name,
    )


def _dataset_outputs_are_complete(dataset_root: Path) -> bool:
    required_files = (
        dataset_root / "manifests" / "manifest.csv",
        dataset_root / "manifests" / "discarded_manifest.csv",
        dataset_root / "metadata" / "dataset_build.json",
        dataset_root / "metadata" / "dataset_fingerprint.json",
    )
    required_dirs = (
        dataset_root / "splits" / "train",
        dataset_root / "splits" / "val",
        dataset_root / "splits" / "test",
    )
    return all(path.is_file() for path in required_files) and all(
        path.is_dir() for path in required_dirs
    )


def _build_reused_result(dataset_root: Path) -> DatasetBuildResult:
    metadata_path = dataset_root / "metadata" / "dataset_build.json"
    data = json.loads(metadata_path.read_text(encoding="utf-8"))
    return DatasetBuildResult(
        train_count=int(data["num_train"]),
        val_count=int(data["num_val"]),
        test_count=int(data["num_test"]),
        skipped_count=int(data["num_patches_discarded"]),
        output_root=dataset_root,
        reused=True,
    )


def _reuse_existing_dataset(config: RunConfig) -> DatasetBuildResult | None:
    if config.preprocessing is None:
        raise ValueError("RunConfig.preprocessing must be present for prepare().")

    dataset_root = config.preprocessing.dataset_root
    stored = _load_json(dataset_root / "metadata" / "dataset_fingerprint.json")
    build_metadata = _load_json(dataset_root / "metadata" / "dataset_build.json")
    if stored is None or build_metadata is None:
        return None
    if not _dataset_outputs_are_complete(dataset_root):
        return None

    current = _build_current_fingerprint(config)
    if stored.get("fingerprint") != current["fingerprint"]:
        return None
    return _build_reused_result(dataset_root)


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

    try:
        result = _reuse_existing_dataset(config)
        if result is None:
            builder = DatasetBuilder(config.preprocessing)
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
        "reused": result.reused,
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
