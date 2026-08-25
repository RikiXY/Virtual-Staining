from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from virtual_staining.data.slide_sets import SlideAsset, SlideSet


def _hash_bytes(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


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


def _cached_file_provenance(
    path: Path,
    *,
    cache: dict[str, Any],
    force: bool,
) -> dict[str, Any]:
    resolved = path.resolve()
    stat = resolved.stat()
    key = str(resolved)
    cached = cache.get(key, {})
    if (
        not force
        and cached.get("size") == stat.st_size
        and cached.get("mtime_ns") == stat.st_mtime_ns
        and isinstance(cached.get("sha256"), str)
    ):
        digest = cached["sha256"]
    else:
        digest = compute_file_sha256(resolved)
    value = {
        "path": key,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": digest,
    }
    cache[key] = value
    return value


def _asset_payload(asset: SlideAsset) -> dict[str, Any]:
    return {
        "modality": asset.modality,
        "path": asset.path.as_posix(),
        "already_aligned": asset.already_aligned,
        "mask_path": asset.mask_path.as_posix() if asset.mask_path else None,
        "slide_id": asset.slide_id,
    }


def canonical_set_payload(slide_sets: tuple[SlideSet, ...]) -> list[dict[str, Any]]:
    return [
        {
            "set_id": item.set_id,
            "reference_modality": item.reference_modality,
            "inputs": [_asset_payload(asset) for asset in item.inputs],
            "target": _asset_payload(item.target),
            "patient_id": item.patient_id,
            "specimen_id": item.specimen_id,
        }
        for item in sorted(slide_sets, key=lambda value: value.set_id)
    ]


def build_dataset_fingerprint_metadata(
    *,
    dataset_root: Path,
    preprocessing_config: dict[str, Any],
    slide_sets: tuple[SlideSet, ...],
    inventory_path: Path | None = None,
    hash_cache_path: Path | None = None,
    force_hash_verification: bool = False,
    prepared_at: str | None = None,
) -> dict[str, Any]:
    cache: dict[str, Any] = {}
    if hash_cache_path is not None and hash_cache_path.exists():
        try:
            cache = json.loads(hash_cache_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            cache = {}
    files: list[dict[str, Any]] = []
    for item in sorted(slide_sets, key=lambda value: value.set_id):
        for asset in (*item.inputs, item.target):
            assets = (("mask", asset.mask_path),) if asset.mask_path is not None else ()
            for role, relative in ((asset.modality, asset.path), *assets):
                files.append(
                    {
                        "set_id": item.set_id,
                        "modality": asset.modality,
                        "role": role,
                        **_cached_file_provenance(
                            dataset_root / relative, cache=cache, force=force_hash_verification
                        ),
                    }
                )
    if hash_cache_path is not None:
        hash_cache_path.parent.mkdir(parents=True, exist_ok=True)
        hash_cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    canonical_sets = canonical_set_payload(slide_sets)
    canonical_inventory_hash = _hash_bytes(_canonical_json_bytes(canonical_sets))
    raw_inventory_sha256 = (
        compute_file_sha256(inventory_path) if inventory_path is not None else None
    )
    dataset_root_resolved = str(dataset_root.resolve())
    semantic_config = json.loads(json.dumps(preprocessing_config))
    preprocessing_hash = _hash_bytes(_canonical_json_bytes(semantic_config))
    fingerprint_payload = {
        "dataset_root": dataset_root_resolved,
        "preprocessing": semantic_config,
        "canonical_inventory": canonical_sets,
        "files": files,
        "schema_version": "3.0",
    }
    return {
        "schema_version": "3.0",
        "fingerprint": _hash_bytes(_canonical_json_bytes(fingerprint_payload)),
        "prepared_at": prepared_at or datetime.now(UTC).isoformat(),
        "dataset_root": dataset_root_resolved,
        "preprocessing": semantic_config,
        "preprocessing_sha256": preprocessing_hash,
        "canonical_inventory_sha256": canonical_inventory_hash,
        "raw_inventory_sha256": raw_inventory_sha256,
        "files": files,
    }


def save_dataset_fingerprint(metadata: dict[str, Any], dest: Path) -> None:
    """Persist dataset fingerprint metadata as canonical JSON."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
