from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import yaml

from virtual_staining.config.run import RunConfig
from virtual_staining.experiment.environment import collect_environment
from virtual_staining.experiment.run_paths import RunPaths

if TYPE_CHECKING:
    from virtual_staining.data.config import PreprocessingConfig


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


def compute_file_sha256(path: Path) -> str:
    """Return sha256:<hex> for a file's content."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def build_file_provenance(path: Path) -> dict[str, Any]:
    """Return canonical provenance for one source dataset file."""
    resolved = path.resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": compute_file_sha256(resolved),
    }


def serialize_preprocessing_config(config: PreprocessingConfig) -> dict[str, Any]:
    """Return the canonical preprocessing payload used for dataset fingerprints."""
    return {
        "dataset_root": str(config.dataset_root.resolve()),
        "source_name": config.source_name,
        "target_name": config.target_name,
        "image_size": list(config.image_size),
        "grid_movement": list(config.grid_movement),
        "margin": config.margin,
        "seed": config.seed,
        "save_masks": config.save_masks,
        "save_discarded_patches": config.save_discarded_patches,
        "mask_scale": config.mask_scale,
        "lowres_mask_filtering": config.lowres_mask_filtering,
        "max_memory_gb": config.max_memory_gb,
        "train_ratio": config.train_ratio,
        "val_ratio": config.val_ratio,
        "test_ratio": config.test_ratio,
        "min_foreground_ratio": config.min_foreground_ratio,
        "max_white_ratio": config.max_white_ratio,
        "white_threshold": config.white_threshold,
        "max_largest_white_component_ratio": config.max_largest_white_component_ratio,
    }


def build_dataset_fingerprint_metadata(
    *,
    dataset_root: Path,
    preprocessing_config: dict[str, Any],
    source_path: Path,
    target_path: Path,
    prepared_at: str | None = None,
) -> dict[str, Any]:
    """Build machine-readable dataset fingerprint metadata for prepare reuse checks."""
    source = build_file_provenance(source_path)
    target = build_file_provenance(target_path)
    dataset_root_resolved = str(dataset_root.resolve())
    preprocessing_hash = compute_payload_hash(preprocessing_config)
    fingerprint_payload = {
        "dataset_root": dataset_root_resolved,
        "preprocessing": preprocessing_config,
        "source": source,
        "target": target,
    }
    return {
        "schema_version": "1.0",
        "fingerprint": compute_payload_hash(fingerprint_payload),
        "prepared_at": prepared_at or datetime.now(UTC).isoformat(),
        "dataset_root": dataset_root_resolved,
        "preprocessing": preprocessing_config,
        "preprocessing_hash": preprocessing_hash,
        "source": source,
        "target": target,
    }


def save_dataset_fingerprint(metadata: dict[str, Any], dest: Path) -> None:
    """Persist dataset fingerprint metadata as canonical JSON."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


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
