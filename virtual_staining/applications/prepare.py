from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from virtual_staining.config.run import RunConfig
from virtual_staining.data.builder import DatasetBuilder, DatasetBuildResult
from virtual_staining.data.config import PreprocessingConfig
from virtual_staining.data.pairs import SlidePair, load_pair_manifest, resolve_slide_pairs
from virtual_staining.experiment.metadata import (
    RunProvenance,
    ensure_run_metadata,
)
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


def _build_current_fingerprint(config: RunConfig, pairs: tuple[SlidePair, ...]) -> dict[str, Any]:
    if config.preprocessing is None:
        raise ValueError("RunConfig.preprocessing must be present for prepare().")
    dataset_root = config.preprocessing.dataset_root
    preprocessing_payload = config.preprocessing.to_dict()
    inputs = config.preprocessing.inputs
    return build_dataset_fingerprint_metadata(
        dataset_root=dataset_root,
        preprocessing_config=preprocessing_payload,
        pairs=pairs,
        inventory_path=(dataset_root / inputs.inventory if inputs else None),
        hash_cache_path=dataset_root / "metadata" / "input_hashes.json",
        force_hash_verification=bool(inputs and inputs.hash_verification == "always"),
    )


def _dataset_outputs_are_complete(dataset_root: Path) -> bool:
    required_files = (
        dataset_root / "manifests" / "manifest.csv",
        dataset_root / "manifests" / "discarded_manifest.csv",
        dataset_root / "manifests" / "pairs.csv",
        dataset_root / "metadata" / "dataset_build.json",
        dataset_root / "metadata" / "dataset_fingerprint.json",
        dataset_root / "metadata" / "split_assignment.csv",
    )
    required_dirs = (
        dataset_root / "splits" / "train",
        dataset_root / "splits" / "val",
        dataset_root / "splits" / "test",
    )
    complete = all(path.is_file() for path in required_files) and all(
        path.is_dir() for path in required_dirs
    )
    if not complete:
        return False
    try:
        pair_rows = load_pair_manifest(dataset_root / "manifests" / "pairs.csv")
    except (OSError, ValueError):
        return False
    return all(
        (dataset_root / row["alignment_metadata_path"]).is_file() for row in pair_rows.values()
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


def _log_prepare_summary(
    preprocessing: PreprocessingConfig,
    pairs: tuple[SlidePair, ...],
    *,
    reused: bool,
) -> None:
    masks = preprocessing.effective_masks
    alignment = preprocessing.effective_alignment
    patch_w, patch_h = preprocessing.patch_size
    step_w, step_h = preprocessing.grid_movement
    io_mode = "tiled" if preprocessing.tiled_io else "full"
    logger.info(
        "Prepare summary | dataset=%s | pairs=%d | action=%s | io=%s/%s | "
        "patch=%dx%d | stride=%dx%d | margin=%d | masks=%s/%s@%gx",
        preprocessing.dataset_root,
        len(pairs),
        "reuse" if reused else "build",
        preprocessing.io_backend,
        io_mode,
        patch_w,
        patch_h,
        step_w,
        step_h,
        preprocessing.margin,
        masks.generation,
        masks.strategy,
        masks.scale,
    )

    for pair in pairs:
        declared = {True: "yes", False: "no", None: "unspecified"}[pair.already_aligned]
        estimate_alignment = alignment.mode == "always" or (
            alignment.mode == "auto" and pair.already_aligned is not True
        )
        alignment_action = alignment.method if estimate_alignment else "identity"

        supplied = any(
            path is not None
            for path in (pair.shared_mask_path, pair.source_mask_path, pair.target_mask_path)
        )
        if supplied and masks.generation == "always":
            mask_action = "conflict: supplied with generation=always"
        elif pair.shared_mask_path is not None:
            mask_action = f"shared {pair.shared_mask_path}"
        elif pair.source_mask_path is not None and pair.target_mask_path is not None:
            mask_action = f"separate source={pair.source_mask_path},target={pair.target_mask_path}"
        elif supplied:
            mask_action = "conflict: incomplete separate masks"
        elif masks.generation == "never":
            mask_action = "none"
        else:
            source_strategy = masks.source_strategy or masks.strategy
            target_strategy = masks.target_strategy or masks.strategy
            strategy = (
                source_strategy
                if source_strategy == target_strategy
                else f"source={source_strategy},target={target_strategy}"
            )
            mask_action = f"generate {strategy}@{masks.scale:g}x"

        logger.info(
            "Pair %s | source=%s | target=%s | aligned=%s | alignment=%s | masks=%s",
            pair.pair_id,
            pair.source_path,
            pair.target_path,
            declared,
            alignment_action,
            mask_action,
        )


def _warn_image_backend(config: RunConfig, pairs: tuple[SlidePair, ...]) -> None:
    preprocessing = config.preprocessing
    if preprocessing is None or not preprocessing.tiled_io:
        return

    root = preprocessing.dataset_root
    paths = tuple(
        sorted(
            {
                root / path
                for pair in pairs
                for path in (pair.source_path, pair.target_path)
                if (root / path).is_file()
            }
        )
    )
    if not paths:
        return
    try:
        incompatible = tuple(path for path in paths if detect_openslide_format(path) is None)
    except RuntimeError:
        if preprocessing.io_backend == "openslide":
            message = "OpenSlide is unavailable, so tiled preparation cannot start."
        else:
            message = "Tiled preparation is using Pillow because OpenSlide is unavailable."
        message += (
            " Enter 'nix develop' and install the Python dependency with 'uv sync --extra wsi'."
        )
        logger.warning(style(message, "yellow"))
        return

    if preprocessing.io_backend == "pillow":
        if incompatible:
            message = (
                "Tiled preparation is using Pillow, and "
                f"{len(incompatible)} configured slide image(s) are not OpenSlide-compatible."
            )
        else:
            message = (
                "Tiled preparation is using Pillow even though the configured slide images "
                "are OpenSlide-compatible. Set 'io.backend: openslide' or 'auto' to avoid "
                "memory-heavy reads."
            )
            logger.warning(style(message, "yellow"))
            return
    elif not incompatible:
        return
    elif preprocessing.io_backend == "auto":
        message = (
            f"{len(incompatible)} configured slide image(s) are not OpenSlide-compatible; "
            "automatic tiled I/O will use memory-heavy Pillow reads for them."
        )
    else:
        message = (
            f"{len(incompatible)} configured slide image(s) are not OpenSlide-compatible, "
            "so tiled preparation cannot read them with OpenSlide."
        )

    shown = ", ".join(str(path) for path in incompatible[:3])
    remaining = len(incompatible) - 3
    if remaining > 0:
        shown += f" (+{remaining} more)"
    message += (
        " Convert them first, for example: 'vs convert /path/to/slides --output-dir "
        "/path/to/converted-slides', then update the inventory paths."
    )
    if shown:
        message += f" Affected: {shown}."
    logger.warning(style(message, "yellow"))


def _reuse_existing_dataset(
    config: RunConfig, current: dict[str, Any]
) -> DatasetBuildResult | None:
    if config.preprocessing is None:
        raise ValueError("RunConfig.preprocessing must be present for prepare().")

    dataset_root = config.preprocessing.dataset_root
    stored = _load_json(dataset_root / "metadata" / "dataset_fingerprint.json")
    build_metadata = _load_json(dataset_root / "metadata" / "dataset_build.json")
    if stored is None or build_metadata is None:
        return None
    if not _dataset_outputs_are_complete(dataset_root):
        return None

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
    pairs = resolve_slide_pairs(config.preprocessing)

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
    with run.stage("prepare", details={"dataset_root": str(dataset_root)}) as stage:
        current_fingerprint = _build_current_fingerprint(config, pairs)
        result = _reuse_existing_dataset(config, current_fingerprint)
        _log_prepare_summary(config.preprocessing, pairs, reused=result is not None)
        if result is None:
            _warn_image_backend(config, pairs)
            builder = DatasetBuilder(
                config.preprocessing,
                pairs=pairs,
                fingerprint_metadata=current_fingerprint,
            )
            result = builder.run_all()
        manifest_path = dataset_root / "manifests" / "manifest.csv"
        stage.result(
            manifest_path=str(manifest_path),
            manifest_sha256=compute_manifest_hash(manifest_path),
            train_count=result.train_count,
            val_count=result.val_count,
            test_count=result.test_count,
            skipped_count=result.skipped_count,
            reused=result.reused,
            pair_inventory_count=len(pairs),
            canonical_inventory_sha256=current_fingerprint.get("canonical_inventory_sha256"),
        )
    return result
