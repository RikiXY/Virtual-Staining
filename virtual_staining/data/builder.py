from __future__ import annotations

import csv
import dataclasses
import datetime
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
from PIL import Image

from virtual_staining.data.config import PreprocessingConfig
from virtual_staining.data.manifest import DatasetManifest, ManifestRecord, Split
from virtual_staining.data.pairs import SlidePair
from virtual_staining.data.preprocessing import (
    MASK_PARAMETER_GRID,
    assign_split_by_hash,
    calculate_mask_by_strategy,
    calculate_mask_with_multiple_parameters,
    ensure_clean_directory,
    estimate_affine_from_scaled,
    foreground_ratio_for_patch,
    is_valid_patch_pair,
    iter_image_with_grid,
    mask_window_for_patch,
    validate_image_filename,
    warp_aligned_mask_patch_from_mask_space,
    warp_aligned_patch,
)
from virtual_staining.data.splitting import (
    assign_group_splits,
    group_id_for_pair,
    write_split_assignment,
)
from virtual_staining.experiment.snapshots import (
    build_dataset_fingerprint_metadata,
    save_dataset_fingerprint,
)
from virtual_staining.utils.image_io import RegionImageReader, open_image_reader

logger = logging.getLogger(__name__)
_MEMORY_WARNING_THRESHOLD_GB = 8.0


@dataclasses.dataclass(frozen=True)
class DatasetBuildResult:
    train_count: int
    val_count: int
    test_count: int
    skipped_count: int
    output_root: Path
    reused: bool = False


@dataclasses.dataclass(frozen=True)
class AlignmentResult:
    method: str
    warp_matrix: np.ndarray
    metadata: dict[str, Any]


@dataclasses.dataclass(frozen=True)
class PairBuildResult:
    pair_id: str
    split: Split | None
    records: tuple[ManifestRecord, ...]
    discarded_records: tuple[ManifestRecord, ...]
    metadata: dict[str, Any]


def _log_memory(stage: str) -> None:
    try:
        import resource
    except ImportError:
        logger.info("Memory after %s: (not available on this platform)", stage)
        return

    max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_mb = max_rss / 1024 if sys.platform == "linux" else max_rss / (1024 * 1024)
    logger.info("Memory after %s: max_rss=%.1f MB", stage, rss_mb)


def _build_manifest_metadata(records: list[ManifestRecord]) -> dict[str, Any]:
    return {
        "schema_version": DatasetManifest.SCHEMA_VERSION,
        "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "record_count": len(records),
        "splits": {
            split: sum(1 for record in records if record.split == split)
            for split in ("train", "val", "test")
        },
    }


def _estimate_memory_gb(
    h: int,
    w: int,
    *,
    mask_scale: float = 1.0,
    lowres_mask_filtering: bool = False,
    tiled_io: bool = False,
) -> float:
    """Rough working-set estimate for processing a single image pair."""
    pixels = h * w
    scaled_h = max(1, int(h * mask_scale))
    scaled_w = max(1, int(w * mask_scale))
    scaled_pixels = scaled_h * scaled_w

    # Full-resolution resident state kept across the pipeline.
    resident_mask_pixels = scaled_pixels if lowres_mask_filtering else pixels
    image_pixels = scaled_pixels if tiled_io else pixels
    full_res_bytes = image_pixels * (2 * 3 + (3 + 1)) + resident_mask_pixels * 2
    # Mask computation scales with the downsampled image area when mask_scale < 1.0.
    scaled_mask_bytes = scaled_pixels * (2 * 3 + 2) * 3
    estimated_bytes = full_res_bytes + scaled_mask_bytes
    return estimated_bytes / (1024**3)


def _modality_from_filename(path: Path) -> str:
    """Derive a human-readable modality label from a configured image filename."""
    return path.stem


def _foreground_mask_patch_name(sample_id: str, suffix: str) -> str:
    return f"{sample_id}_foreground_mask{suffix}"


def _read_image_size(path: Path) -> tuple[int, int]:
    """
    Read image size from metadata only.

    Pillow's decompression-bomb guard can fire on legitimate whole-slide or
    microscopy images even when reading only headers, so disable it briefly
    for this metadata-only check and restore it immediately afterward.
    """
    original_max_image_pixels = Image.MAX_IMAGE_PIXELS
    try:
        Image.MAX_IMAGE_PIXELS = None
        with Image.open(path) as img:
            return img.size
    finally:
        Image.MAX_IMAGE_PIXELS = original_max_image_pixels


class PairProcessor:
    """
    Orchestrates the full dataset preparation pipeline.

    Owns the config, loaded images, and all stage-to-stage intermediate results.
    Low-level image operations are delegated to pure functions in preprocessing.py.

    Call run_all() to execute the complete pipeline, or call individual stage
    methods in order (compute_masks -> align) to run incrementally.
    """

    def __init__(
        self,
        config: PreprocessingConfig,
        pair: SlidePair | None = None,
        assigned_split: Split | None = None,
    ) -> None:
        self.config = config
        self.pair = pair or SlidePair(
            "pair_0000", Path(config.source_name or ""), Path(config.target_name or "")
        )
        self.assigned_split = assigned_split
        self._source_file = validate_image_filename(str(self.pair.source_path), "Source")
        self._target_file = validate_image_filename(str(self.pair.target_path), "Target")
        self._source_suffix = self._source_file.suffix.lower()
        self._target_suffix = self._target_file.suffix.lower()

        self._source_image: np.ndarray | None = None
        self._target_image: np.ndarray | None = None
        self._source_mask: np.ndarray | None = None
        self._target_mask: np.ndarray | None = None
        self._warp_matrix: np.ndarray | None = None
        self._alignment_metadata: dict[str, Any] | None = None
        self._alignment_result: AlignmentResult | None = None
        self._source_reader: RegionImageReader | None = None
        self._target_reader: RegionImageReader | None = None
        self._source_shape: tuple[int, int] | None = None
        self._target_shape: tuple[int, int] | None = None
        self._alignment_preview_scale: float = 1.0
        self._started_at: str | None = None
        self._effective_seed: int | None = None
        self._maskless = False

    def _uses_lowres_mask_filtering(self) -> bool:
        masks = self.config.effective_masks
        return masks.lowres_filtering and masks.scale < 1.0

    def _uses_mask_space_filtering(self) -> bool:
        return self._uses_lowres_mask_filtering() or self.config.tiled_io

    @staticmethod
    def _mask_in_image_space(mask: np.ndarray, image: np.ndarray) -> np.ndarray:
        image_h, image_w = image.shape[:2]
        if mask.shape[:2] == (image_h, image_w):
            return mask
        return cv2.resize(mask, (image_w, image_h), interpolation=cv2.INTER_NEAREST)

    def _source_mask_strategy(self) -> str:
        masks = self.config.effective_masks
        return masks.source_strategy or masks.strategy

    def _target_mask_strategy(self) -> str:
        masks = self.config.effective_masks
        return masks.target_strategy or masks.strategy

    @staticmethod
    def _calculate_mask(img: np.ndarray, *, strategy: str) -> np.ndarray:
        if strategy == "connected_components":
            mask = calculate_mask_with_multiple_parameters(img, MASK_PARAMETER_GRID)
        else:
            mask = calculate_mask_by_strategy(
                img, strategy=strategy, parameters=MASK_PARAMETER_GRID
            )
        mask[np.all(img == 0, axis=2)] = 0
        return mask

    @staticmethod
    def _shape_from_size(size: tuple[int, int]) -> tuple[int, int]:
        width, height = size
        return height, width

    @staticmethod
    def _fullres_warp_from_preview(warp_matrix: np.ndarray, preview_scale: float) -> np.ndarray:
        fullres_matrix = np.asarray(warp_matrix, dtype=np.float64).copy()
        if preview_scale != 1.0:
            fullres_matrix[:, 2] /= preview_scale
        return fullres_matrix

    def _supplied_mask_paths(self) -> tuple[Path, Path] | None:
        masks = self.config.effective_masks
        shared = self.pair.shared_mask_path
        source = self.pair.source_mask_path
        target = self.pair.target_mask_path
        supplied = [path for path in (shared, source, target) if path is not None]
        if masks.generation == "always" and supplied:
            raise ValueError(
                f"Pair {self.pair.pair_id}: generation=always cannot be combined with masks"
            )
        if shared is not None and (source is not None or target is not None):
            raise ValueError(f"Pair {self.pair.pair_id}: shared and separate masks cannot be mixed")
        if (source is None) != (target is None):
            raise ValueError(f"Pair {self.pair.pair_id}: separate masks require both paths")
        actual = "shared" if shared else "separate" if source and target else "none"
        if masks.provided_layout != "auto" and masks.provided_layout != actual:
            raise ValueError(
                f"Pair {self.pair.pair_id}: masks.provided_layout={masks.provided_layout!r} "
                f"does not match {actual!r}"
            )
        if shared is not None:
            if self.pair.already_aligned is not True:
                raise ValueError(
                    f"Pair {self.pair.pair_id}: shared masks require already_aligned=true"
                )
            return shared, shared
        if source is not None and target is not None:
            return source, target
        return None

    def _load_supplied_masks(self) -> bool:
        paths = self._supplied_mask_paths()
        masks = self.config.effective_masks
        if paths is None:
            if masks.generation != "never":
                return False
            if self.config.foreground_enabled:
                raise ValueError(
                    f"Pair {self.pair.pair_id}: maskless processing requires "
                    "foreground.enabled=false"
                )
            assert self._source_image is not None and self._target_image is not None
            self._source_mask = np.full(self._source_image.shape[:2], 255, dtype=np.uint8)
            self._target_mask = np.full(self._target_image.shape[:2], 255, dtype=np.uint8)
            self._maskless = True
            return True

        assert self._source_image is not None and self._target_image is not None

        def load(path: Path, expected_size: tuple[int, int]) -> np.ndarray:
            full_path = self.config.dataset_root / path
            if not self.config.tiled_io:
                mask = cv2.imread(str(full_path), cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    raise ValueError(f"Pair {self.pair.pair_id}: could not read {path}")
                return mask

            reader = open_image_reader(full_path, backend=self.config.io_backend)
            try:
                image = (
                    reader.read_preview(masks.scale)
                    if reader.size == expected_size
                    else reader.read_full()
                )
            finally:
                reader.close()
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return np.where(gray > 0, 255, 0).astype(np.uint8)

        source_size = (self._source_image.shape[1], self._source_image.shape[0])
        target_size = (self._target_image.shape[1], self._target_image.shape[0])
        if self._source_shape is not None:
            source_size = (self._source_shape[1], self._source_shape[0])
        if self._target_shape is not None:
            target_size = (self._target_shape[1], self._target_shape[0])
        source_mask = load(paths[0], source_size)
        target_mask = source_mask if paths[0] == paths[1] else load(paths[1], target_size)
        if self.config.tiled_io:
            source_mask = cv2.resize(
                source_mask,
                (self._source_image.shape[1], self._source_image.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            target_mask = cv2.resize(
                target_mask,
                (self._target_image.shape[1], self._target_image.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        self._source_mask, self._target_mask = source_mask, target_mask
        return True

    @staticmethod
    def _warp_target_region_patch(
        reader: RegionImageReader,
        warp_matrix: np.ndarray,
        *,
        x: int,
        y: int,
        output_size: tuple[int, int],
        border: int = 2,
    ) -> np.ndarray:
        patch_w, patch_h = output_size
        inverse_matrix = cv2.invertAffineTransform(np.asarray(warp_matrix, dtype=np.float64))
        corners = np.array(
            [
                [x, y],
                [x + patch_w, y],
                [x, y + patch_h],
                [x + patch_w, y + patch_h],
            ],
            dtype=np.float64,
        )
        target_corners = cv2.transform(corners[None, :, :], inverse_matrix)[0]
        region_x = int(np.floor(target_corners[:, 0].min())) - border
        region_y = int(np.floor(target_corners[:, 1].min())) - border
        region_right = int(np.ceil(target_corners[:, 0].max())) + border
        region_bottom = int(np.ceil(target_corners[:, 1].max())) + border
        region_w = max(1, region_right - region_x)
        region_h = max(1, region_bottom - region_y)

        region = reader.read_region(region_x, region_y, region_w, region_h)
        region_matrix = np.asarray(warp_matrix, dtype=np.float64).copy()
        region_matrix[:, 2] += region_matrix[:, :2] @ np.array([region_x, region_y])
        return warp_aligned_patch(
            region,
            region_matrix,
            x=x,
            y=y,
            output_size=output_size,
            is_mask=False,
        )

    # ------------------------------------------------------------------
    # Pipeline stages
    # ------------------------------------------------------------------

    def compute_masks(self) -> None:
        """Load source and target images and compute tissue masks for both."""
        root = self.config.dataset_root
        if not root.exists():
            raise FileNotFoundError(f"Dataset root not found: {root}")

        source_path = root / self._source_file
        target_path = root / self._target_file
        if self.config.tiled_io:
            self._source_reader = open_image_reader(source_path, backend=self.config.io_backend)
            self._target_reader = open_image_reader(target_path, backend=self.config.io_backend)
            source_w, source_h = self._source_reader.size
            target_w, target_h = self._target_reader.size
            self._source_shape = (source_h, source_w)
            self._target_shape = (target_h, target_w)
        else:
            source_w, source_h = _read_image_size(source_path)

        estimated_gb = _estimate_memory_gb(
            source_h,
            source_w,
            mask_scale=self.config.effective_masks.scale,
            lowres_mask_filtering=self._uses_mask_space_filtering(),
            tiled_io=self.config.tiled_io,
        )
        if self.config.max_memory_gb is not None and estimated_gb > self.config.max_memory_gb:
            raise MemoryError(
                f"Estimated working-set size {estimated_gb:.1f} GB exceeds configured "
                f"max_memory_gb={self.config.max_memory_gb}. Consider using 'mask_scale: 0.25' "
                "in your preprocessing config, splitting large images, or increasing "
                "max_memory_gb."
            )
        if self.config.max_memory_gb is None and estimated_gb > _MEMORY_WARNING_THRESHOLD_GB:
            logger.warning(
                "Estimated working-set size %.1f GB is large. If the process is killed, "
                "use 'mask_scale: 0.25' to reduce memory usage.",
                estimated_gb,
            )

        if self.config.tiled_io:
            assert self._source_reader is not None
            assert self._target_reader is not None
            preview_scale = self.config.effective_masks.scale
            self._alignment_preview_scale = preview_scale
            self._source_image = self._source_reader.read_preview(preview_scale)
            self._target_image = self._target_reader.read_preview(preview_scale)
        else:
            self._source_image = cv2.imread(str(source_path))
            self._target_image = cv2.imread(str(target_path))
            if self._source_image is None or self._target_image is None:
                raise FileNotFoundError(
                    f"Missing paired files for pair {self.pair.pair_id!r} inside: {root}"
                )
            self._source_shape = self._source_image.shape[:2]
            self._target_shape = self._target_image.shape[:2]

        if self._load_supplied_masks():
            self._save_resolved_masks()
            _log_memory("compute_masks")
            return

        logger.info("Calculating masks...")
        scale = self.config.effective_masks.scale
        if self.config.tiled_io:
            source_mask = self._calculate_mask(
                self._source_image,
                strategy=self._source_mask_strategy(),
            )
            target_mask = self._calculate_mask(
                self._target_image,
                strategy=self._target_mask_strategy(),
            )
            self._source_mask = source_mask
            self._target_mask = target_mask
        elif scale < 1.0:
            assert self._source_image is not None and self._target_image is not None
            source_h, source_w = self._source_image.shape[:2]
            target_h, target_w = self._target_image.shape[:2]
            scaled_source_size = (max(1, int(source_w * scale)), max(1, int(source_h * scale)))
            scaled_target_size = (max(1, int(target_w * scale)), max(1, int(target_h * scale)))

            small_source = cv2.resize(self._source_image, scaled_source_size)
            small_target = cv2.resize(self._target_image, scaled_target_size)
            source_mask = self._calculate_mask(
                small_source,
                strategy=self._source_mask_strategy(),
            )
            target_mask = self._calculate_mask(
                small_target,
                strategy=self._target_mask_strategy(),
            )
            if self._uses_mask_space_filtering():
                self._source_mask = source_mask
                self._target_mask = target_mask
            else:
                self._source_mask = cv2.resize(
                    source_mask,
                    (source_w, source_h),
                    interpolation=cv2.INTER_NEAREST,
                )
                self._target_mask = cv2.resize(
                    target_mask,
                    (target_w, target_h),
                    interpolation=cv2.INTER_NEAREST,
                )
        else:
            self._source_mask = self._calculate_mask(
                self._source_image,
                strategy=self._source_mask_strategy(),
            )
            self._target_mask = self._calculate_mask(
                self._target_image,
                strategy=self._target_mask_strategy(),
            )

        self._save_resolved_masks()

        _log_memory("compute_masks")

    def _save_resolved_masks(self) -> None:
        if not self.config.effective_masks.save_resolved_masks:
            return
        assert self._source_mask is not None and self._target_mask is not None
        assert self._source_image is not None and self._target_image is not None
        output = self.config.dataset_root / "resolved_masks" / self.pair.pair_id
        output.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(
            str(output / "source.tif"),
            self._mask_in_image_space(self._source_mask, self._source_image),
        )
        cv2.imwrite(
            str(output / "target.tif"),
            self._mask_in_image_space(self._target_mask, self._target_image),
        )

    def align(self) -> None:
        """Align the target image to the source reference frame."""
        if (
            self._source_image is None
            or self._target_image is None
            or self._source_mask is None
            or self._target_mask is None
        ):
            raise RuntimeError("compute_masks() must be called before align()")

        policy = self.config.effective_alignment
        if policy.mode == "never" and self.pair.already_aligned is False:
            raise ValueError(
                f"Pair {self.pair.pair_id}: alignment.mode=never contradicts already_aligned=false"
            )
        estimate = policy.mode == "always" or (
            policy.mode == "auto" and self.pair.already_aligned is not True
        )
        if self.pair.shared_mask_path is not None and estimate:
            raise ValueError(f"Pair {self.pair.pair_id}: a shared mask requires identity alignment")
        if self._maskless and estimate:
            raise ValueError(
                f"Pair {self.pair.pair_id}: affine registration requires source and target masks"
            )

        if not estimate:
            if policy.validate_declared:
                if self._source_shape != self._target_shape:
                    raise ValueError(
                        f"Pair {self.pair.pair_id}: identity alignment requires equal geometry"
                    )
                if self._source_reader is not None and self._target_reader is not None:
                    source_metadata = self._source_reader.metadata
                    target_metadata = self._target_reader.metadata
                    for name in ("mpp_x", "mpp_y"):
                        source_mpp = getattr(source_metadata, name, None)
                        target_mpp = getattr(target_metadata, name, None)
                        if (
                            source_mpp is not None
                            and target_mpp is not None
                            and not np.isclose(source_mpp, target_mpp, rtol=0.01)
                        ):
                            raise ValueError(
                                f"Pair {self.pair.pair_id}: identity alignment has "
                                f"incompatible {name}"
                            )
            identity = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
            self._warp_matrix = identity
            self._alignment_metadata = {
                "method": "identity",
                "reason": "declared_aligned" if self.pair.already_aligned else "policy_never",
                "warp_matrix": identity.tolist(),
            }
            self._alignment_result = AlignmentResult(
                method="identity", warp_matrix=identity, metadata=self._alignment_metadata
            )
            _log_memory("align")
            return

        logger.info("Aligning images...")
        source_alignment_mask = self._mask_in_image_space(self._source_mask, self._source_image)
        target_alignment_mask = self._mask_in_image_space(self._target_mask, self._target_image)
        warp_matrix, metadata = estimate_affine_from_scaled(
            self._source_image,
            self._target_image,
            mask_1=source_alignment_mask,
            mask_2=target_alignment_mask,
            scale=0.5,
        )
        if self.config.tiled_io:
            warp_matrix = self._fullres_warp_from_preview(
                warp_matrix,
                self._alignment_preview_scale,
            )
            metadata.warp_matrix = warp_matrix.tolist()
            metadata.translation_x = float(warp_matrix[0, 2])
            metadata.translation_y = float(warp_matrix[1, 2])
        self._warp_matrix = warp_matrix
        self._alignment_metadata = dataclasses.asdict(metadata)

        self._alignment_metadata["method"] = "affine_sift"
        self._alignment_result = AlignmentResult(
            method="affine_sift",
            warp_matrix=warp_matrix,
            metadata=self._alignment_metadata,
        )
        if self.config.inputs is None:
            with open(
                self.config.dataset_root / "alignment_metadata.json", "w", encoding="utf-8"
            ) as f:
                json.dump(self._alignment_metadata, f, indent=2)

        _log_memory("align")

    def _stream_patches_to_disk(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """
        Extract, filter, and write patches in a single pass.

        Returns metadata rows for valid and discarded patches without retaining
        patch arrays on the builder.
        """
        if (
            (self._source_image is None and not self.config.tiled_io)
            or self._source_mask is None
            or (self._target_image is None and not self.config.tiled_io)
            or self._target_mask is None
            or self._warp_matrix is None
            or self._source_shape is None
            or self._target_shape is None
        ):
            raise RuntimeError("align() must be called before _stream_patches_to_disk()")

        root = self.config.dataset_root
        splits_root = root / "splits"
        discarded_src_dir = root / "discarded_patches" / self.pair.pair_id / "source"
        discarded_tgt_dir = root / "discarded_patches" / self.pair.pair_id / "target"
        split_dirs: dict[Split, Path] = {
            "train": splits_root / "train" / self.pair.pair_id,
            "val": splits_root / "val" / self.pair.pair_id,
            "test": splits_root / "test" / self.pair.pair_id,
        }
        for split_name, path in split_dirs.items():
            if self.assigned_split is None or split_name == self.assigned_split:
                path.mkdir(parents=True, exist_ok=True)
        if self.config.save_discarded_patches:
            for path in [discarded_src_dir, discarded_tgt_dir]:
                path.mkdir(parents=True, exist_ok=True)

        m = self.config.margin

        def _crop(img: np.ndarray) -> np.ndarray:
            # img[0:-0] returns an empty array, so treat margin=0 as no crop.
            return img[m:-m, m:-m] if m > 0 else img

        use_mask_space_filtering = self._uses_mask_space_filtering()
        if self.config.tiled_io:
            source_h, source_w = self._source_shape
            cropped_w = max(0, source_w - 2 * m)
            cropped_h = max(0, source_h - 2 * m)

            def _source_iter() -> Any:
                if self._source_reader is None:
                    raise RuntimeError("Tiled source reader is not available")
                patch_w, patch_h = self.config.patch_size
                step_x, step_y = self.config.grid_movement
                for x in range(0, cropped_w, step_x):
                    for y in range(0, cropped_h, step_y):
                        if y + patch_h > cropped_h or x + patch_w > cropped_w:
                            continue
                        yield (
                            (x, y),
                            self._source_reader.read_region(
                                x + m,
                                y + m,
                                patch_w,
                                patch_h,
                            ),
                            None,
                        )

            source_iter = _source_iter()
        else:
            assert self._source_image is not None
            cropped_source = _crop(self._source_image)
            source_prefilter = self.config.foreground_enabled and self.config.foreground_policy in {
                "source",
                "both",
                "intersection",
            }
            cropped_source_mask = (
                None
                if use_mask_space_filtering or not source_prefilter
                else _crop(self._source_mask)
            )
            source_iter = iter_image_with_grid(
                cropped_source,
                self.config.patch_size,
                self.config.grid_movement,
                cropped_source_mask,
                max_mask_percentage=self.config.min_foreground_ratio,
            )

        extracted_pairs = 0
        _log_memory("extract_patches")
        valid_rows: list[dict[str, Any]] = []
        discarded_rows: list[dict[str, Any]] = []
        patch_w, patch_h = self.config.patch_size
        split_seed = self._effective_seed if self._effective_seed is not None else self.config.seed
        if split_seed is None:
            split_seed = 0
        split_ratios = (self.config.train_ratio, self.config.val_ratio, self.config.test_ratio)
        for (x, y), src, src_mask in source_iter:
            target_x = x + m
            target_y = y + m
            if use_mask_space_filtering:
                source_foreground_ratio = foreground_ratio_for_patch(
                    self._source_mask,
                    self._source_shape,
                    x=target_x,
                    y=target_y,
                    width=patch_w,
                    height=patch_h,
                )
                if (
                    self.config.foreground_enabled
                    and self.config.foreground_policy in {"source", "both", "intersection"}
                    and source_foreground_ratio < self.config.min_foreground_ratio
                ):
                    continue
                source_mask_window = mask_window_for_patch(
                    self._source_mask,
                    self._source_shape,
                    x=target_x,
                    y=target_y,
                    width=patch_w,
                    height=patch_h,
                )
                src_mask = cv2.resize(
                    source_mask_window,
                    (patch_w, patch_h),
                    interpolation=cv2.INTER_NEAREST,
                )
            elif src_mask is None:
                src_mask = self._source_mask[
                    target_y : target_y + patch_h,
                    target_x : target_x + patch_w,
                ]
            if self.config.tiled_io:
                if self._target_reader is None:
                    raise RuntimeError("Tiled target reader is not available")
                tgt = self._warp_target_region_patch(
                    self._target_reader,
                    self._warp_matrix,
                    x=target_x,
                    y=target_y,
                    output_size=(patch_w, patch_h),
                )
            else:
                assert self._target_image is not None
                tgt = warp_aligned_patch(
                    self._target_image,
                    self._warp_matrix,
                    x=target_x,
                    y=target_y,
                    output_size=(patch_w, patch_h),
                    is_mask=False,
                )
            if use_mask_space_filtering:
                tgt_mask = warp_aligned_mask_patch_from_mask_space(
                    self._target_mask,
                    self._warp_matrix,
                    self._target_shape,
                    x=target_x,
                    y=target_y,
                    output_size=(patch_w, patch_h),
                )
            else:
                tgt_mask = warp_aligned_patch(
                    self._target_mask,
                    self._warp_matrix,
                    x=target_x,
                    y=target_y,
                    output_size=(patch_w, patch_h),
                    is_mask=True,
                )
            if (
                tgt.shape[0] < patch_h
                or tgt.shape[1] < patch_w
                or tgt_mask.shape[0] < patch_h
                or tgt_mask.shape[1] < patch_w
            ):
                raise RuntimeError(
                    "Patch extraction mismatch after extraction: "
                    f"source=({x}, {y}), target_shape={tgt.shape}, "
                    f"target_mask_shape={tgt_mask.shape}"
                )

            extracted_pairs += 1
            canonical_x, canonical_y = x + m, y + m
            sample_id = f"{self.pair.pair_id}__x{canonical_x:08}_y{canonical_y:08}"
            patch_source_name = f"{sample_id}_source{self._source_suffix}"
            patch_target_name = f"{sample_id}_target{self._target_suffix}"
            patch_mask_name = _foreground_mask_patch_name(sample_id, self._target_suffix)

            is_valid, debug_info = is_valid_patch_pair(
                source_img=src,
                target_img=tgt,
                source_mask=src_mask,
                target_mask=tgt_mask,
                min_foreground_ratio=0.0,
                max_white_ratio=self.config.max_white_ratio,
                white_threshold=self.config.white_threshold,
                max_largest_white_component_ratio=self.config.max_largest_white_component_ratio,
            )
            if self.config.foreground_enabled:
                source_ratio = float(cv2.countNonZero(src_mask) / src_mask.size)
                target_ratio = float(cv2.countNonZero(tgt_mask) / tgt_mask.size)
                intersection_ratio = float(
                    cv2.countNonZero(cv2.bitwise_and(src_mask, tgt_mask)) / src_mask.size
                )
                union_ratio = float(
                    cv2.countNonZero(cv2.bitwise_or(src_mask, tgt_mask)) / src_mask.size
                )
                ratios = {
                    "source": source_ratio,
                    "target": target_ratio,
                    "both": min(source_ratio, target_ratio),
                    "intersection": intersection_ratio,
                    "union": union_ratio,
                }
                if ratios[self.config.foreground_policy] < self.config.min_foreground_ratio:
                    is_valid = False
                    cast(list[str], debug_info["reasons"]).append(
                        f"foreground_{self.config.foreground_policy}"
                    )
            if is_valid:
                split_sample_id = (
                    f"{canonical_x:05}_{canonical_y:05}"
                    if self.config.inputs is None
                    else sample_id
                )
                split_name = self.assigned_split or cast(
                    Split,
                    assign_split_by_hash(
                        seed=split_seed, sample_id=split_sample_id, ratios=split_ratios
                    ),
                )
                split_dir = split_dirs[cast(Split, split_name)]
                cv2.imwrite(str(split_dir / patch_source_name), src)
                cv2.imwrite(str(split_dir / patch_target_name), tgt)
                if self.config.effective_masks.save_patch_masks and not self._maskless:
                    cv2.imwrite(str(split_dir / patch_mask_name), tgt_mask)
                valid_rows.append(
                    {
                        "sample_id": sample_id,
                        "pair_id": self.pair.pair_id,
                        "split": split_name,
                        "x": canonical_x,
                        "y": canonical_y,
                        "source": patch_source_name,
                        "target": patch_target_name,
                        "foreground_mask": (
                            patch_mask_name
                            if self.config.effective_masks.save_patch_masks and not self._maskless
                            else None
                        ),
                    }
                )
            else:
                if self.config.save_discarded_patches:
                    cv2.imwrite(str(discarded_src_dir / patch_source_name), src)
                    cv2.imwrite(str(discarded_tgt_dir / patch_target_name), tgt)
                discarded_rows.append(
                    {
                        "sample_id": sample_id,
                        "pair_id": self.pair.pair_id,
                        "x": canonical_x,
                        "y": canonical_y,
                        "source_name": patch_source_name,
                        "target_name": patch_target_name,
                        "source_foreground_ratio": debug_info["source_foreground_ratio"],
                        "target_foreground_ratio": debug_info["target_foreground_ratio"],
                        "source_white_ratio": debug_info["source_white_ratio"],
                        "target_white_ratio": debug_info["target_white_ratio"],
                        "source_largest_white_component_ratio": debug_info[
                            "source_largest_white_component_ratio"
                        ],
                        "target_largest_white_component_ratio": debug_info[
                            "target_largest_white_component_ratio"
                        ],
                        "reasons": ";".join(cast(list[str], debug_info["reasons"])),
                    }
                )

        logger.info("Extracted %s patch pairs", extracted_pairs)
        _log_memory("filter_patches")
        return valid_rows, discarded_rows

    def _assign_splits_and_finalize(
        self,
        valid_rows: list[dict[str, Any]],
        discarded_rows: list[dict[str, Any]],
    ) -> DatasetBuildResult:
        """Write manifests and metadata for patches already streamed to disk."""
        root = self.config.dataset_root
        discarded_root = root / "discarded_patches"
        manifests_dir = root / "manifests"
        metadata_dir = root / "metadata"

        for path in [
            manifests_dir,
        ]:
            ensure_clean_directory(path)
        metadata_dir.mkdir(parents=True, exist_ok=True)
        discarded_root.mkdir(parents=True, exist_ok=True)

        input_modality = self.config.source_modality
        target_modality = self.config.target_modality

        manifest_records: list[ManifestRecord] = []
        for row in valid_rows:
            src_name = cast(str, row["source"])
            tgt_name = cast(str, row["target"])
            x = cast(int, row["x"])
            y = cast(int, row["y"])
            split_name = cast(Split, row["split"])
            sample_id = cast(str, row["sample_id"])

            manifest_records.append(
                ManifestRecord(
                    sample_id=sample_id,
                    split=split_name,
                    input_path=Path(f"splits/{split_name}/{src_name}"),
                    target_path=Path(f"splits/{split_name}/{tgt_name}"),
                    input_modality=input_modality,
                    target_modality=target_modality,
                    x=x,
                    y=y,
                    width=self.config.patch_size[0],
                    height=self.config.patch_size[1],
                    pair_id=self.pair.pair_id,
                    foreground_mask_path=(
                        Path(
                            f"splits/{split_name}/{self.pair.pair_id}/"
                            f"{cast(str, row['foreground_mask'])}"
                        )
                        if row.get("foreground_mask")
                        else None
                    ),
                )
            )

        discarded_manifest_records: list[ManifestRecord] = []
        for row in discarded_rows:
            src_name = cast(str, row["source_name"])
            tgt_name = cast(str, row["target_name"])
            sample_id = cast(str, row["sample_id"])
            x = cast(int, row["x"])
            y = cast(int, row["y"])
            discarded_manifest_records.append(
                ManifestRecord(
                    sample_id=sample_id,
                    split="discarded",
                    input_path=Path(f"discarded_patches/source/{src_name}"),
                    target_path=Path(f"discarded_patches/target/{tgt_name}"),
                    input_modality=input_modality,
                    target_modality=target_modality,
                    x=x,
                    y=y,
                    width=self.config.patch_size[0],
                    height=self.config.patch_size[1],
                    pair_id=self.pair.pair_id,
                )
            )

        # Paths written by PairProcessor live one directory below each split.
        manifest_records = [
            dataclasses.replace(
                record,
                input_path=Path(
                    f"splits/{record.split}/{self.pair.pair_id}/{record.input_path.name}"
                ),
                target_path=Path(
                    f"splits/{record.split}/{self.pair.pair_id}/{record.target_path.name}"
                ),
            )
            for record in manifest_records
        ]
        manifest = DatasetManifest(
            records=tuple(manifest_records), dataset_root=root, schema_version="2.0"
        )
        manifest.validate()
        manifest.to_csv(manifests_dir / "manifest.csv")
        (manifests_dir / "manifest_metadata.json").write_text(
            json.dumps(_build_manifest_metadata(manifest_records), indent=2),
            encoding="utf-8",
        )

        discarded_manifest = DatasetManifest(
            records=tuple(discarded_manifest_records),
            dataset_root=root,
            schema_version="2.0",
        )
        discarded_manifest.validate()
        discarded_manifest.to_csv(manifests_dir / "discarded_manifest.csv")

        with open(discarded_root / "discarded_log.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "sample_id",
                    "pair_id",
                    "x",
                    "y",
                    "source_name",
                    "target_name",
                    "source_foreground_ratio",
                    "target_foreground_ratio",
                    "source_white_ratio",
                    "target_white_ratio",
                    "reasons",
                    "source_largest_white_component_ratio",
                    "target_largest_white_component_ratio",
                ],
            )
            writer.writeheader()
            writer.writerows(discarded_rows)

        build_metadata = {
            "dataset_name": root.name,
            "status": "completed",
            "started_at": self._started_at,
            "completed_at": datetime.datetime.now(datetime.UTC).isoformat(),
            "num_patches_total": len(manifest_records) + len(discarded_manifest_records),
            "num_patches_valid": len(manifest_records),
            "num_patches_discarded": len(discarded_manifest_records),
            "num_train": sum(1 for record in manifest_records if record.split == "train"),
            "num_val": sum(1 for record in manifest_records if record.split == "val"),
            "num_test": sum(1 for record in manifest_records if record.split == "test"),
            "seed": self._effective_seed,
        }
        with open(metadata_dir / "dataset_build.json", "w", encoding="utf-8") as f:
            json.dump(build_metadata, f, indent=2, default=str)

        fingerprint_metadata = build_dataset_fingerprint_metadata(
            dataset_root=root,
            preprocessing_config=self.config.to_dict(),
            source_path=root / self._source_file,
            target_path=root / self._target_file,
            prepared_at=build_metadata["completed_at"],
        )
        save_dataset_fingerprint(fingerprint_metadata, metadata_dir / "dataset_fingerprint.json")

        logger.info(
            "Saved: train=%s, val=%s, test=%s, discarded=%s",
            build_metadata["num_train"],
            build_metadata["num_val"],
            build_metadata["num_test"],
            len(discarded_rows),
        )
        _log_memory("split_and_save")
        return DatasetBuildResult(
            train_count=cast(int, build_metadata["num_train"]),
            val_count=cast(int, build_metadata["num_val"]),
            test_count=cast(int, build_metadata["num_test"]),
            skipped_count=len(discarded_rows),
            output_root=root,
            reused=False,
        )

    def run_all(self) -> DatasetBuildResult:
        """Run all pipeline stages in sequence and return the build result."""
        self._started_at = datetime.datetime.now(datetime.UTC).isoformat()
        seed = self.config.seed if self.config.seed is not None else random.randint(0, 2**32 - 1)
        self._effective_seed = seed
        random.seed(seed)
        logger.info("Seed set to %s", seed)

        self.compute_masks()
        self.align()
        valid_rows, discarded_rows = self._stream_patches_to_disk()
        result = self._assign_splits_and_finalize(valid_rows, discarded_rows)

        return result


class DatasetBuilder:
    """Own one complete dataset build; PairProcessor owns each source/target pair."""

    def __init__(
        self,
        config: PreprocessingConfig,
        pairs: tuple[SlidePair, ...] | None = None,
        fingerprint_metadata: dict[str, Any] | None = None,
    ) -> None:
        self.config = config
        self.pairs: tuple[SlidePair, ...] = (
            pairs
            if pairs is not None
            else (SlidePair("pair_0000", Path(config.source_name), Path(config.target_name)),)
        )
        if not self.pairs:
            raise ValueError("DatasetBuilder requires at least one slide pair")
        self.fingerprint_metadata = fingerprint_metadata
        self._single = PairProcessor(config, self.pairs[0]) if len(self.pairs) == 1 else None

    def __getattr__(self, name: str) -> Any:
        if self._single is not None:
            return getattr(self._single, name)
        raise AttributeError(name)

    @property
    def _started_at(self) -> str | None:
        return self._single._started_at if self._single else None

    @_started_at.setter
    def _started_at(self, value: str | None) -> None:
        if self._single is not None:
            self._single._started_at = value

    @property
    def _effective_seed(self) -> int | None:
        return self._single._effective_seed if self._single else None

    @_effective_seed.setter
    def _effective_seed(self, value: int | None) -> None:
        if self._single is not None:
            self._single._effective_seed = value

    def compute_masks(self) -> None:
        if self._single is None:
            raise RuntimeError("compute_masks() is pair-scoped; use run_all() for multiple pairs")
        self._single.compute_masks()

    def align(self) -> None:
        if self._single is None:
            raise RuntimeError("align() is pair-scoped; use run_all() for multiple pairs")
        self._single.align()

    def _stream_patches_to_disk(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if self._single is None:
            raise RuntimeError("patch streaming is pair-scoped; use run_all() for multiple pairs")
        return self._single._stream_patches_to_disk()

    def _assign_splits_and_finalize(
        self, valid_rows: list[dict[str, Any]], discarded_rows: list[dict[str, Any]]
    ) -> DatasetBuildResult:
        if self._single is None:
            raise RuntimeError("finalization is dataset-scoped; use run_all() for multiple pairs")
        return self._single._assign_splits_and_finalize(valid_rows, discarded_rows)

    def _initialize_output_tree_once(self) -> None:
        root = self.config.dataset_root
        for path in (
            root / "splits" / "train",
            root / "splits" / "val",
            root / "splits" / "test",
            root / "manifests",
            root / "discarded_patches",
            root / "metadata" / "pairs",
            root / "resolved_masks",
        ):
            ensure_clean_directory(path)

    @staticmethod
    def _close_processor(processor: PairProcessor) -> None:
        for reader in (processor._source_reader, processor._target_reader):
            if reader is not None:
                close = getattr(reader, "close", None)
                if close is not None:
                    close()

    def _records(
        self, rows: list[dict[str, Any]], *, discarded: bool = False
    ) -> tuple[ManifestRecord, ...]:
        records: list[ManifestRecord] = []
        for row in rows:
            pair_id = cast(str, row["pair_id"])
            sample_id = cast(str, row["sample_id"])
            split = cast(Split, "discarded" if discarded else row["split"])
            if discarded:
                input_path = Path(
                    f"discarded_patches/{pair_id}/source/{cast(str, row['source_name'])}"
                )
                target_path = Path(
                    f"discarded_patches/{pair_id}/target/{cast(str, row['target_name'])}"
                )
                foreground_mask_path = None
            else:
                input_path = Path(f"splits/{split}/{pair_id}/{cast(str, row['source'])}")
                target_path = Path(f"splits/{split}/{pair_id}/{cast(str, row['target'])}")
                foreground_mask_path = (
                    Path(f"splits/{split}/{pair_id}/{cast(str, row['foreground_mask'])}")
                    if row.get("foreground_mask")
                    else None
                )
            records.append(
                ManifestRecord(
                    sample_id=sample_id,
                    pair_id=pair_id,
                    split=split,
                    input_path=input_path,
                    target_path=target_path,
                    foreground_mask_path=foreground_mask_path,
                    input_modality=self.config.source_modality,
                    target_modality=self.config.target_modality,
                    x=cast(int, row["x"]),
                    y=cast(int, row["y"]),
                    width=self.config.patch_size[0],
                    height=self.config.patch_size[1],
                )
            )
        return tuple(sorted(records, key=lambda record: (record.pair_id, record.y, record.x)))

    def run_all(self) -> DatasetBuildResult:
        root = self.config.dataset_root
        started_at = datetime.datetime.now(datetime.UTC).isoformat()
        split = self.config.effective_split
        logger.info("Seed set to %s", split.seed)
        ratios = (split.train, split.val, split.test)
        pair_assignments = assign_group_splits(
            self.pairs,
            unit=split.unit,
            ratios=ratios,
            seed=split.seed,
            assignment_file=split.assignment_file,
            dataset_root=root,
        )
        self._initialize_output_tree_once()

        valid_rows: list[dict[str, Any]] = []
        discarded_rows: list[dict[str, Any]] = []
        pair_rows: list[dict[str, Any]] = []
        excluded_rows: list[dict[str, str]] = []
        pair_results: list[PairBuildResult] = []
        for pair in sorted(self.pairs, key=lambda item: item.pair_id):
            assigned = pair_assignments.get(pair.pair_id)
            processor = (
                self._single
                if self._single is not None and pair.pair_id == self._single.pair.pair_id
                else PairProcessor(self.config, pair, assigned)
            )
            processor.assigned_split = assigned
            processor._started_at = started_at
            processor._effective_seed = split.seed
            error: Exception | None = None
            pair_valid: list[dict[str, Any]] = []
            pair_discarded: list[dict[str, Any]] = []
            source_details: dict[str, Any] = {"path": pair.source_path.as_posix()}
            target_details: dict[str, Any] = {"path": pair.target_path.as_posix()}
            try:
                processor.compute_masks()
                for details, reader, shape in (
                    (source_details, processor._source_reader, processor._source_shape),
                    (target_details, processor._target_reader, processor._target_shape),
                ):
                    if reader is not None:
                        details.update(dataclasses.asdict(reader.metadata))
                    elif shape is not None:
                        details.update({"width": shape[1], "height": shape[0]})
                try:
                    processor.align()
                except Exception as exc:
                    if self.config.effective_alignment.on_failure != "skip_pair":
                        raise
                    error = exc
                    excluded_rows.append(
                        {"pair_id": pair.pair_id, "split": assigned or "", "error": str(exc)}
                    )
                if error is None:
                    pair_valid, pair_discarded = processor._stream_patches_to_disk()
                    valid_rows.extend(pair_valid)
                    discarded_rows.extend(pair_discarded)
            finally:
                self._close_processor(processor)

            metadata_path = Path(f"metadata/pairs/{pair.pair_id}.json")
            metadata = {
                "pair_id": pair.pair_id,
                "split": assigned,
                "status": "excluded" if error else "processed",
                "source": source_details,
                "target": target_details,
                "masks": {
                    "layout": (
                        "shared"
                        if pair.shared_mask_path
                        else "separate"
                        if pair.source_mask_path and pair.target_mask_path
                        else "none"
                        if processor._maskless
                        else "generated"
                    ),
                    "source_origin": (
                        "none"
                        if processor._maskless
                        else "provided"
                        if pair.shared_mask_path or pair.source_mask_path
                        else "generated"
                    ),
                    "target_origin": (
                        "none"
                        if processor._maskless
                        else "provided"
                        if pair.shared_mask_path or pair.target_mask_path
                        else "generated"
                    ),
                },
                "alignment": processor._alignment_metadata,
                "patches": {
                    "candidate": len(pair_valid) + len(pair_discarded),
                    "accepted": len(pair_valid),
                    "discarded": len(pair_discarded),
                },
                **({"error": str(error)} if error else {}),
            }
            (root / metadata_path).write_text(
                json.dumps(metadata, indent=2, default=str), encoding="utf-8"
            )
            pair_results.append(
                PairBuildResult(
                    pair_id=pair.pair_id,
                    split=assigned,
                    records=self._records(pair_valid),
                    discarded_records=self._records(pair_discarded, discarded=True),
                    metadata=metadata,
                )
            )
            pair_rows.append(
                {
                    "pair_id": pair.pair_id,
                    "split": assigned or "",
                    "source_path": pair.source_path.as_posix(),
                    "target_path": pair.target_path.as_posix(),
                    "patient_id": pair.patient_id or "",
                    "specimen_id": pair.specimen_id or "",
                    "source_slide_id": pair.source_slide_id or "",
                    "target_slide_id": pair.target_slide_id or "",
                    "already_aligned": (
                        "" if pair.already_aligned is None else str(pair.already_aligned).lower()
                    ),
                    "shared_mask_path": pair.shared_mask_path.as_posix()
                    if pair.shared_mask_path
                    else "",
                    "source_mask_path": pair.source_mask_path.as_posix()
                    if pair.source_mask_path
                    else "",
                    "target_mask_path": pair.target_mask_path.as_posix()
                    if pair.target_mask_path
                    else "",
                    "status": "excluded" if error else "processed",
                    "alignment_method": (
                        processor._alignment_metadata.get("method", "")
                        if processor._alignment_metadata
                        else ""
                    ),
                    "alignment_metadata_path": metadata_path.as_posix(),
                }
            )

        records = tuple(
            sorted(
                (record for result in pair_results for record in result.records),
                key=lambda record: (record.pair_id, record.y, record.x),
            )
        )
        discarded_records = tuple(
            sorted(
                (record for result in pair_results for record in result.discarded_records),
                key=lambda record: (record.pair_id, record.y, record.x),
            )
        )
        manifest = DatasetManifest(records, root, schema_version="2.0")
        manifest.validate()
        if self.config.inputs is not None:
            required = {
                name
                for name, ratio in zip(("train", "val", "test"), ratios, strict=True)
                if ratio > 0
            }
            manifest.validate(require_splits=required)
        manifests = root / "manifests"
        manifest.to_csv(manifests / "manifest.csv")
        DatasetManifest(discarded_records, root, schema_version="2.0").to_csv(
            manifests / "discarded_manifest.csv"
        )
        (manifests / "manifest_metadata.json").write_text(
            json.dumps(_build_manifest_metadata(list(records)), indent=2), encoding="utf-8"
        )

        pair_fields = [
            "pair_id",
            "split",
            "source_path",
            "target_path",
            "patient_id",
            "specimen_id",
            "source_slide_id",
            "target_slide_id",
            "already_aligned",
            "shared_mask_path",
            "source_mask_path",
            "target_mask_path",
            "status",
            "alignment_method",
            "alignment_metadata_path",
        ]
        with (manifests / "pairs.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=pair_fields)
            writer.writeheader()
            writer.writerows(pair_rows)

        assignment_rows: dict[str, Split]
        if split.unit == "patch":
            assignment_rows = {record.sample_id: cast(Split, record.split) for record in records}
        else:
            assignment_rows = {
                group_id_for_pair(pair, split.unit): pair_assignments[pair.pair_id]
                for pair in self.pairs
            }
        write_split_assignment(
            root / "metadata" / "split_assignment.csv",
            unit=split.unit,
            assignments=assignment_rows,
        )
        with (root / "metadata" / "excluded_pairs.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=["pair_id", "split", "error"])
            writer.writeheader()
            writer.writerows(excluded_rows)

        with (root / "discarded_patches" / "discarded_log.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            fields = [
                "sample_id",
                "pair_id",
                "x",
                "y",
                "source_name",
                "target_name",
                "source_foreground_ratio",
                "target_foreground_ratio",
                "source_white_ratio",
                "target_white_ratio",
                "reasons",
                "source_largest_white_component_ratio",
                "target_largest_white_component_ratio",
            ]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(discarded_rows)

        counts = {
            name: sum(record.split == name for record in records)
            for name in ("train", "val", "test")
        }
        processed_pairs = [row for row in pair_rows if row["status"] == "processed"]
        groups_by_split = {
            name: (
                len({record.sample_id for record in records if record.split == name})
                if split.unit == "patch"
                else len(
                    {
                        group_id_for_pair(pair, split.unit)
                        for pair in self.pairs
                        if pair_assignments[pair.pair_id] == name
                    }
                )
            )
            for name in ("train", "val", "test")
        }
        pairs_by_split = {
            name: len({record.pair_id for record in records if record.split == name})
            for name in ("train", "val", "test")
        }
        build_metadata = {
            "dataset_name": root.name,
            "status": "completed",
            "started_at": started_at,
            "completed_at": datetime.datetime.now(datetime.UTC).isoformat(),
            "num_pairs": len(self.pairs),
            "num_pairs_processed": len(processed_pairs),
            "num_pairs_excluded": len(excluded_rows),
            "num_patients": len({pair.patient_id for pair in self.pairs if pair.patient_id}),
            "num_specimens": len({pair.specimen_id for pair in self.pairs if pair.specimen_id}),
            "splitting": {"unit": split.unit, "seed": split.seed},
            "groups": groups_by_split,
            "pairs": pairs_by_split,
            "patches": {**counts, "discarded": len(discarded_records)},
            # Compatibility counters consumed by prepare reuse and existing callers.
            "num_train": counts["train"],
            "num_val": counts["val"],
            "num_test": counts["test"],
            "num_patches_discarded": len(discarded_records),
            "num_patches_total": len(records) + len(discarded_records),
            "num_patches_valid": len(records),
            "seed": split.seed,
            "canonical_inventory_sha256": (
                self.fingerprint_metadata.get("canonical_inventory_sha256")
                if self.fingerprint_metadata
                else None
            ),
        }
        metadata_dir = root / "metadata"
        (metadata_dir / "dataset_build.json").write_text(
            json.dumps(build_metadata, indent=2), encoding="utf-8"
        )
        if self.fingerprint_metadata is not None:
            fingerprint = dict(self.fingerprint_metadata)
            fingerprint["prepared_at"] = build_metadata["completed_at"]
        elif self.config.inputs is not None or len(self.pairs) > 1:
            fingerprint = build_dataset_fingerprint_metadata(
                dataset_root=root,
                preprocessing_config=self.config.to_dict(),
                pairs=self.pairs,
                prepared_at=build_metadata["completed_at"],
            )
        else:
            pair = next(iter(self.pairs))
            fingerprint = build_dataset_fingerprint_metadata(
                dataset_root=root,
                preprocessing_config=self.config.to_dict(),
                source_path=root / pair.source_path,
                target_path=root / pair.target_path,
                prepared_at=build_metadata["completed_at"],
            )
        save_dataset_fingerprint(fingerprint, metadata_dir / "dataset_fingerprint.json")
        logger.info(
            "Saved: train=%s, val=%s, test=%s, discarded=%s",
            counts["train"],
            counts["val"],
            counts["test"],
            len(discarded_records),
        )
        _log_memory("split_and_save")
        return DatasetBuildResult(
            train_count=counts["train"],
            val_count=counts["val"],
            test_count=counts["test"],
            skipped_count=len(discarded_records),
            output_root=root,
        )
