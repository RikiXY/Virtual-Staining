from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

from virtual_staining.config.run import RunConfig
from virtual_staining.experiment.environment import collect_environment
from virtual_staining.experiment.run_paths import RunPaths


@dataclass(frozen=True)
class SnapshotPaths:
    input_config: Path
    resolved_config: Path
    environment: Path


def resolve_run_snapshot_paths(
    *,
    stage: Literal["train", "infer", "evaluate"],
    run_paths: RunPaths,
) -> SnapshotPaths:
    """Return canonical snapshot destinations for a run-scoped stage."""
    if stage not in {"train", "infer", "evaluate"}:
        raise ValueError(f"Unsupported run snapshot stage: {stage}")
    config_dir = run_paths.stage_config_dir(stage)
    return SnapshotPaths(
        input_config=config_dir / "input.yaml",
        resolved_config=config_dir / "resolved.yaml",
        environment=run_paths.stage_environment(stage),
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


def _hash_bytes(payload: bytes) -> str:
    digest = hashlib.sha256(payload).hexdigest()
    return f"sha256:{digest}"


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def compute_payload_hash(payload: Any) -> str:
    """Return sha256:<hex> for a canonical JSON payload."""
    return _hash_bytes(_canonical_json_bytes(payload))


def save_config_hash(hash_str: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(hash_str, encoding="utf-8")


def save_stage_config_snapshots(
    config: RunConfig,
    config_path: Path,
    *,
    input_dest: Path,
    resolved_dest: Path,
) -> str:
    """Write canonical input/resolved snapshots and return the config hash."""
    save_input_config(config_path, input_dest)
    save_resolved_config(config.to_dict(), resolved_dest)
    return compute_config_hash(resolved_dest)


def save_environment_snapshot(dest: Path) -> None:
    """Write a JSON snapshot of the active runtime environment."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as handle:
        json.dump(collect_environment(), handle, indent=2, default=str)
