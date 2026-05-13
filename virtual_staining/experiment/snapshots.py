from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

from virtual_staining.config.run import RunConfig
from virtual_staining.experiment.environment import collect_environment
from virtual_staining.experiment.run_paths import RunPaths


@dataclass(frozen=True)
class SnapshotPaths:
    input_config: Path
    resolved_config: Path
    config_hash: Path
    environment: Path


def resolve_run_snapshot_paths(
    *,
    stage: Literal["training", "inference", "evaluation"],
    run_paths: RunPaths,
) -> SnapshotPaths:
    """Return canonical snapshot destinations for a run-scoped stage."""
    if stage == "training":
        return SnapshotPaths(
            input_config=run_paths.input_config,
            resolved_config=run_paths.resolved_config,
            config_hash=run_paths.config_hash,
            environment=run_paths.environment_metadata,
        )
    if stage == "inference":
        return SnapshotPaths(
            input_config=run_paths.config_dir / "inference.input.yaml",
            resolved_config=run_paths.config_dir / "inference.resolved.yaml",
            config_hash=run_paths.metadata_dir / "inference_config_hash.txt",
            environment=run_paths.metadata_dir / "inference_environment.json",
        )
    if stage == "evaluation":
        return SnapshotPaths(
            input_config=run_paths.config_dir / "evaluation.input.yaml",
            resolved_config=run_paths.config_dir / "evaluation.resolved.yaml",
            config_hash=run_paths.metadata_dir / "evaluation_config_hash.txt",
            environment=run_paths.metadata_dir / "evaluation_environment.json",
        )
    raise ValueError(f"Unsupported run snapshot stage: {stage}")


def resolve_prepare_snapshot_paths(dataset_root: Path) -> SnapshotPaths:
    """Return canonical snapshot destinations for prepare-stage artifacts."""
    return SnapshotPaths(
        input_config=dataset_root / "config" / "input.yaml",
        resolved_config=dataset_root / "config" / "resolved.yaml",
        config_hash=dataset_root / "metadata" / "config_hash.txt",
        environment=dataset_root / "metadata" / "environment.json",
    )


def save_input_config(src_yaml: Path, dest: Path) -> None:
    """Copy the user-supplied YAML into config/input.yaml."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_yaml, dest)


def save_resolved_config(config_dict: dict[str, object], dest: Path) -> None:
    """Write the resolved config as canonical YAML."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            config_dict,
            handle,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=True,
        )


def compute_config_hash(yaml_path: Path) -> str:
    """Return sha256:<hex> of the YAML file content."""
    digest = hashlib.sha256(yaml_path.read_bytes()).hexdigest()
    return f"sha256:{digest}"


def compute_manifest_hash(manifest_path: Path) -> str:
    """Return sha256:<hex> of the manifest file content."""
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    return f"sha256:{digest}"


def save_config_hash(hash_str: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(hash_str, encoding="utf-8")


def save_stage_config_snapshots(
    config: RunConfig,
    config_path: Path,
    *,
    input_dest: Path,
    resolved_dest: Path,
    hash_dest: Path,
) -> str:
    """Write the canonical input/resolved/hash snapshot set for a stage."""
    save_input_config(config_path, input_dest)
    save_resolved_config(config.to_yaml_dict(), resolved_dest)
    config_hash = compute_config_hash(resolved_dest)
    save_config_hash(config_hash, hash_dest)
    return config_hash


def save_environment_snapshot(dest: Path) -> None:
    """Write a JSON snapshot of the active runtime environment."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as handle:
        json.dump(collect_environment(), handle, indent=2, default=str)
