from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from virtual_staining.config.data import PreprocessingConfig
from virtual_staining.config.run import RunConfig
from virtual_staining.data.builder import DatasetBuilder, DatasetBuildResult
from virtual_staining.data.layout import DatasetLayout
from virtual_staining.data.provenance import build_dataset_fingerprint_metadata
from virtual_staining.data.slide_sets import SlideSet, resolve_slide_sets
from virtual_staining.experiment.snapshots import (
    save_config_hash,
    save_environment_snapshot,
    save_stage_config_snapshots,
)
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
    layout = DatasetLayout(config.preprocessing.dataset_root)
    return build_dataset_fingerprint_metadata(
        dataset_root=layout.root,
        preprocessing_config=config.preprocessing.to_dict(),
        slide_sets=slide_sets,
        inventory_path=layout.root / config.preprocessing.inputs.inventory,
        hash_cache_path=layout.input_hashes_path,
        force_hash_verification=config.preprocessing.inputs.hash_verification == "always",
    )


def _dataset_outputs_are_complete(dataset_root: Path) -> bool:
    layout = DatasetLayout(dataset_root)
    required = (
        layout.manifest_path,
        layout.discarded_manifest_path,
        layout.slide_sets_path,
        layout.manifest_metadata_path,
        layout.dataset_build_path,
        layout.dataset_fingerprint_path,
        layout.split_assignment_path,
    )
    return all(path.is_file() for path in required) and all(
        layout.split_dir(name).is_dir() for name in ("train", "val", "test")
    )


def _build_reused_result(dataset_root: Path) -> DatasetBuildResult:
    layout = DatasetLayout(dataset_root)
    return DatasetBuildResult.load(layout.dataset_build_path, output_root=layout.root, reused=True)


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
        logger.warning(message)
        return
    if incompatible and preprocessing.io.backend == "openslide":
        logger.warning(
            "Configured slides are not OpenSlide-compatible; tiled preparation "
            "cannot use the requested backend."
        )


def prepare(config: RunConfig, config_path: Path) -> DatasetBuildResult:
    if config.preprocessing is None:
        raise ValueError("RunConfig.preprocessing must be present for prepare().")
    root = config.preprocessing.dataset_root
    layout = DatasetLayout(root)
    slide_sets = resolve_slide_sets(config.preprocessing)
    config_hash = save_stage_config_snapshots(
        config,
        config_path,
        input_dest=layout.input_config_path,
        resolved_dest=layout.resolved_config_path,
    )
    save_config_hash(config_hash, layout.config_hash_path)
    save_environment_snapshot(layout.environment_path)

    fingerprint = _build_current_fingerprint(config, slide_sets)
    stored = _load_json(layout.dataset_fingerprint_path)
    result = None
    if (
        stored
        and stored.get("fingerprint") == fingerprint.get("fingerprint")
        and _dataset_outputs_are_complete(layout.root)
    ):
        result = _build_reused_result(layout.root)
    _log_prepare_summary(config.preprocessing, slide_sets, reused=result is not None)
    if result is None:
        _warn_image_backend(config, slide_sets)
        result = DatasetBuilder(
            config.preprocessing, slide_sets=slide_sets, fingerprint_metadata=fingerprint
        ).run_all()
    return result
