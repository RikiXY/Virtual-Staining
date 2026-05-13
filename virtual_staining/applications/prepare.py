from __future__ import annotations

from pathlib import Path

from virtual_staining.config.run import RunConfig
from virtual_staining.data.builder import DatasetBuilder
from virtual_staining.data.results import DatasetBuildResult
from virtual_staining.experiment.snapshots import (
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

    save_stage_config_snapshots(
        config,
        config_path,
        input_dest=snapshot_paths.input_config,
        resolved_dest=snapshot_paths.resolved_config,
        hash_dest=snapshot_paths.config_hash,
    )
    save_environment_snapshot(snapshot_paths.environment)

    builder = DatasetBuilder(config.preprocessing)
    return builder.run_all()
