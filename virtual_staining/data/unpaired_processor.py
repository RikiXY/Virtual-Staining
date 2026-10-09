"""Bounded native-geometry preparation of one independent image."""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from virtual_staining.config.data import PreprocessingConfig
from virtual_staining.data.filtering import _compute_white_stats
from virtual_staining.data.patching import iter_patch_origins, mask_window_for_patch
from virtual_staining.data.preprocessing import (
    MASK_PARAMETER_GRID,
    calculate_mask_by_strategy,
    calculate_mask_with_multiple_parameters,
)
from virtual_staining.data.splitting import assign_split_by_hash
from virtual_staining.data.unpaired_inventory import RawDomainImage
from virtual_staining.split_contract import DatasetSplit
from virtual_staining.utils.files import publish_file_no_replace
from virtual_staining.utils.hashing import sha256_json
from virtual_staining.utils.image_io import (
    PillowRegionImageReader,
    open_image_reader,
    verify_written_image,
)


def _write_image(path: Path, pixels: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(".png", pixels)
    if not success:
        raise OSError(f"Could not encode patch {path}")
    publish_file_no_replace(
        encoded.tobytes(), path, verify=lambda dest: verify_written_image(dest, pixels)
    )


def _mask(image: np.ndarray, strategy: str) -> np.ndarray:
    mask = (
        calculate_mask_with_multiple_parameters(image, MASK_PARAMETER_GRID)
        if strategy == "connected_components"
        else calculate_mask_by_strategy(image, strategy=strategy, parameters=MASK_PARAMETER_GRID)
    )
    mask[np.all(image == 0, axis=2)] = 0
    return mask


def process_domain_image(
    config: PreprocessingConfig,
    item: RawDomainImage,
    output: Path,
    assigned_split: DatasetSplit | None,
) -> dict[str, Any]:
    """Return accepted/excluded origins and geometry, closing readers on every exit."""
    source_id = sha256_json({"domain": item.domain, "path": item.path}).removeprefix("sha256:")
    patch_w, patch_h = config.patching.patch_size
    with ExitStack() as resources:
        reader = open_image_reader(config.dataset_root / item.path, backend=config.io.backend)
        resources.callback(reader.close)
        width, height = reader.size
        mask_reader = None
        if item.mask_path and config.masks.generation != "always":
            mask_reader = open_image_reader(
                config.dataset_root / item.mask_path, backend=config.io.backend
            )
            resources.callback(mask_reader.close)
            if mask_reader.size != reader.size:
                raise ValueError(f"Mask geometry must equal its source image: {item.mask_path}")
        maskless = mask_reader is None and config.masks.generation == "never"
        if maskless and config.filtering.foreground.enabled:
            raise ValueError("maskless processing requires foreground.enabled=false")
        # Account for image, mask, and filtering work arrays before allocating pixels.
        working_pixels = patch_w * patch_h
        if not config.io.tiled or (not maskless and mask_reader is None and config.masks.scale < 1):
            working_pixels += max(1, int(width * config.masks.scale)) * max(
                1, int(height * config.masks.scale)
            )
        if not config.io.tiled or isinstance(reader, PillowRegionImageReader):
            working_pixels += width * height
        if (
            config.io.max_memory_gb is not None
            and working_pixels * 64 > config.io.max_memory_gb * 1024**3
        ):
            raise MemoryError(
                "Unpaired preparation working arrays exceed io.max_memory_gb; "
                "select an explicit smaller mask scale or patch size"
            )
        full = reader.read_full() if not config.io.tiled else None
        overview_mask = None
        if not maskless and mask_reader is None and config.masks.scale < 1:
            overview_mask = _mask(reader.read_preview(config.masks.scale), config.masks.strategy)
            if config.masks.save_resolved_masks:
                _write_image(
                    output / "masks" / "resolved" / item.domain / f"{source_id}.png", overview_mask
                )
        accepted, excluded = [], []
        for x, y in iter_patch_origins(
            image_size=(width, height),
            patch_size=(patch_w, patch_h),
            grid_movement=config.patching.grid_movement,
            margin=config.patching.margin,
        ):
            image = (
                reader.read_region(x, y, patch_w, patch_h)
                if full is None
                else full[y : y + patch_h, x : x + patch_w]
            )
            if image.shape != (patch_h, patch_w, 3) or image.dtype != np.uint8:
                raise ValueError("Image reader must return the requested native 8-bit RGB patch")
            mask = None
            if mask_reader is not None:
                mask_rgb = mask_reader.read_region(x, y, patch_w, patch_h)
                if not np.all((mask_rgb == 0) | (mask_rgb == 255)) or not np.all(
                    mask_rgb == mask_rgb[:, :, :1]
                ):
                    raise ValueError(
                        f"Mask must contain only known binary 0/255 values: {item.mask_path}"
                    )
                mask = mask_rgb[:, :, 0]
            elif overview_mask is not None:
                mask = mask_window_for_patch(
                    overview_mask, (height, width), x=x, y=y, width=patch_w, height=patch_h
                )
            elif not maskless:
                mask = _mask(image, config.masks.strategy)
            ratio = float(np.count_nonzero(mask) / mask.size) if mask is not None else None
            if mask is not None:
                mask = cv2.resize(mask, (patch_w, patch_h), interpolation=cv2.INTER_NEAREST)
                if not config.masks.lowres_filtering:
                    ratio = float(np.count_nonzero(mask) / mask.size)
            white, largest = _compute_white_stats(
                image,
                config.filtering.white_threshold,
                largest_component_threshold=config.filtering.max_largest_white_component_ratio,
            )
            reasons = []
            if (
                config.filtering.foreground.enabled
                and ratio is not None
                and ratio < config.filtering.foreground.min_ratio
            ):
                reasons.append("low_foreground")
            if white > config.filtering.max_white_ratio:
                reasons.append("high_white_ratio")
            if largest > config.filtering.max_largest_white_component_ratio:
                reasons.append("high_largest_white_component_ratio")
            sample_id = f"{source_id}__x{x:08}_y{y:08}"
            row = {
                "sample_id": sample_id,
                "source": item.path,
                "domain": item.domain,
                "x": x,
                "y": y,
                "width": patch_w,
                "height": patch_h,
                "set_id": item.set_id,
                "specimen_id": item.specimen_id,
                "patient_id": item.patient_id,
            }
            if mask is not None and config.masks.save_resolved_masks and overview_mask is None:
                _write_image(output / "masks" / "resolved" / item.domain / f"{sample_id}.png", mask)
            if reasons:
                excluded.append({**row, "reasons": reasons})
                if config.patching.save_discarded_patches:
                    _write_image(
                        output / "discarded_patches" / item.domain / f"{sample_id}.png", image
                    )
                continue
            split = assigned_split or assign_split_by_hash(
                seed=config.split.seed,
                sample_id=sample_id,
                ratios=(config.split.train, config.split.val, config.split.test),
            )
            locator = f"splits/{split}/{item.domain}/{sample_id}.png"
            _write_image(output / locator, image)
            if mask is not None and config.masks.save_patch_masks:
                _write_image(output / "masks" / split / item.domain / f"{sample_id}.png", mask)
            accepted.append({**row, "path": locator, "split": split})
        return {
            "source": asdict(item),
            "geometry": asdict(reader.metadata),
            "accepted": accepted,
            "excluded": excluded,
        }
