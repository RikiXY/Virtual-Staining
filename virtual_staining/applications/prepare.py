from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from virtual_staining.config.run import RunConfig
from virtual_staining.data.builder import DatasetBuilder, DatasetBuildResult
from virtual_staining.data.config import PreprocessingConfig
from virtual_staining.data.slide_sets import SlideSet, resolve_slide_sets
from virtual_staining.experiment.metadata import RunProvenance, ensure_run_metadata
from virtual_staining.experiment.snapshots import (
    build_dataset_fingerprint_metadata,
    compute_manifest_hash,
    resolve_prepare_snapshot_paths,
    save_environment_snapshot,
    save_stage_config_snapshots,
)
from virtual_staining.utils.console import style
from virtual_staining.utils.image_io import detect_openslide_format

logger = logging.getLogger(__name__)


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _build_current_fingerprint(
    config: RunConfig, slide_sets: tuple[SlideSet, ...]
) -> dict[str, Any]:
    assert config.preprocessing is not None
    root = config.preprocessing.dataset_root
    return build_dataset_fingerprint_metadata(
        dataset_root=root,
        preprocessing_config=config.preprocessing.to_dict(),
        slide_sets=slide_sets,
        inventory_path=root / config.preprocessing.inputs.inventory,
        hash_cache_path=root / "metadata" / "input_hashes.json",
        force_hash_verification=config.preprocessing.inputs.hash_verification == "always",
    )


def _dataset_outputs_are_complete(dataset_root: Path) -> bool:
    required = (
        dataset_root / "manifests" / "manifest.csv",
        dataset_root / "manifests" / "discarded_manifest.csv",
        dataset_root / "manifests" / "slide_sets.csv",
        dataset_root / "manifests" / "manifest_metadata.json",
        dataset_root / "metadata" / "dataset_build.json",
        dataset_root / "metadata" / "dataset_fingerprint.json",
        dataset_root / "metadata" / "split_assignment.csv",
    )
    return all(path.is_file() for path in required) and all(
        (dataset_root / "splits" / name).is_dir() for name in ("train", "val", "test")
    )


def _build_reused_result(dataset_root: Path) -> DatasetBuildResult:
    data = _load_json(dataset_root / "metadata" / "dataset_build.json") or {}
    patches = data.get("patches", {})
    return DatasetBuildResult(
        int(patches.get("train", 0)),
        int(patches.get("val", 0)),
        int(patches.get("test", 0)),
        int(patches.get("discarded", 0)),
        dataset_root,
        reused=True,
    )


def _log_prepare_summary(
    preprocessing: PreprocessingConfig, slide_sets: tuple[SlideSet, ...], *, reused: bool
) -> None:
    logger.info(
        "Prepare summary | dataset=%s | sets=%d | action=%s",
        preprocessing.dataset_root,
        len(slide_sets),
        "reuse" if reused else "build",
    )
    for item in slide_sets:
        logger.info(
            "Set %s | inputs=%s | target=%s | reference=%s",
            item.set_id,
            ",".join(asset.modality for asset in item.inputs),
            item.target.modality,
            item.reference_modality,
        )


def _warn_image_backend(config: RunConfig, slide_sets: tuple[SlideSet, ...]) -> None:
    preprocessing = config.preprocessing
    if preprocessing is None or not preprocessing.io.tiled:
        return
    root = preprocessing.dataset_root
    paths = tuple(
        sorted(
            {
                root / asset.path
                for item in slide_sets
                for asset in (*item.inputs, item.target)
                if (root / asset.path).is_file()
            }
        )
    )
    if not paths:
        return
    try:
        incompatible = tuple(path for path in paths if detect_openslide_format(path) is None)
    except RuntimeError:
        message = (
            "OpenSlide is unavailable, so tiled preparation cannot start."
            if preprocessing.io.backend == "openslide"
            else "Tiled preparation is using Pillow because OpenSlide is unavailable."
        )
        logger.warning(style(message, "yellow"))
        return
    if incompatible and preprocessing.io.backend == "openslide":
        logger.warning(
            style(
                "Configured slides are not OpenSlide-compatible; tiled preparation "
                "cannot use the requested backend.",
                "yellow",
            )
        )


def prepare(config: RunConfig, config_path: Path) -> DatasetBuildResult:
    if config.preprocessing is None:
        raise ValueError("RunConfig.preprocessing must be present for prepare().")
    root = config.preprocessing.dataset_root
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")
    snapshot_paths = resolve_prepare_snapshot_paths(root)
    metadata_dir = root / "metadata"
    slide_sets = resolve_slide_sets(config.preprocessing)
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
        entrypoint="vs prepare",
        config_hash=config_hash,
    )
    run = RunProvenance(metadata_dir, config.project.run_name, config_hash)
    with run.stage("prepare", details={"dataset_root": str(root)}) as stage:
        fingerprint = _build_current_fingerprint(config, slide_sets)
        result = None
        stored = _load_json(root / "metadata" / "dataset_fingerprint.json")
        if (
            stored
            and stored.get("fingerprint") == fingerprint.get("fingerprint")
            and _dataset_outputs_are_complete(root)
        ):
            result = _build_reused_result(root)
        _log_prepare_summary(config.preprocessing, slide_sets, reused=result is not None)
        if result is None:
            _warn_image_backend(config, slide_sets)
            result = DatasetBuilder(
                config.preprocessing, slide_sets=slide_sets, fingerprint_metadata=fingerprint
            ).run_all()
        manifest_path = root / "manifests" / "manifest.csv"
        stage.result(
            manifest_path=str(manifest_path),
            manifest_sha256=compute_manifest_hash(manifest_path),
            train_count=result.train_count,
            val_count=result.val_count,
            test_count=result.test_count,
            skipped_count=result.skipped_count,
            reused=result.reused,
            set_inventory_count=len(slide_sets),
            canonical_inventory_sha256=fingerprint.get("canonical_inventory_sha256"),
        )
    return result
