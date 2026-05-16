from __future__ import annotations

import csv
import dataclasses
import datetime
import json
import logging
import random
import shutil
import sys
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
from PIL import Image

from virtual_staining.data.config import PreprocessingConfig
from virtual_staining.data.manifest import DatasetManifest, ManifestRecord, Split
from virtual_staining.data.preprocessing import (
    MASK_PARAMETER_GRID,
    calculate_mask_with_multiple_parameters,
    ensure_clean_directory,
    estimate_affine_from_scaled,
    is_valid_patch_pair,
    iter_image_with_grid,
    split_items,
    validate_image_filename,
    warp_aligned_patch,
)
from virtual_staining.data.results import DatasetBuildResult
from virtual_staining.experiment.snapshots import (
    build_dataset_fingerprint_metadata,
    save_dataset_fingerprint,
    serialize_preprocessing_config,
)

logger = logging.getLogger(__name__)
_MEMORY_WARNING_THRESHOLD_GB = 8.0


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


def _estimate_memory_gb(h: int, w: int, *, mask_scale: float = 1.0) -> float:
    """Rough working-set estimate for processing a single image pair."""
    pixels = h * w
    scaled_h = max(1, int(h * mask_scale))
    scaled_w = max(1, int(w * mask_scale))
    scaled_pixels = scaled_h * scaled_w

    # Full-resolution resident state kept across the pipeline.
    full_res_bytes = pixels * (2 * 3 + 2 + (3 + 1))
    # Mask computation scales with the downsampled image area when mask_scale < 1.0.
    scaled_mask_bytes = scaled_pixels * (2 * 3 + 2) * 3
    estimated_bytes = full_res_bytes + scaled_mask_bytes
    return estimated_bytes / (1024**3)


def _modality_from_filename(path: Path) -> str:
    """Derive a human-readable modality label from a configured image filename."""
    return path.stem


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


class DatasetBuilder:
    """
    Orchestrates the full dataset preparation pipeline.

    Owns the config, loaded images, and all stage-to-stage intermediate results.
    Low-level image operations are delegated to pure functions in preprocessing.py.

    Call run_all() to execute the complete pipeline, or call individual stage
    methods in order (compute_masks -> align) to run incrementally.
    """

    def __init__(self, config: PreprocessingConfig) -> None:
        self.config = config
        self._source_file = validate_image_filename(config.source_name, "Source")
        self._target_file = validate_image_filename(config.target_name, "Target")
        self._source_suffix = self._source_file.suffix.lower()
        self._target_suffix = self._target_file.suffix.lower()

        self._source_image: np.ndarray | None = None
        self._target_image: np.ndarray | None = None
        self._source_mask: np.ndarray | None = None
        self._target_mask: np.ndarray | None = None
        self._warp_matrix: np.ndarray | None = None
        self._alignment_metadata: dict[str, Any] | None = None
        self._started_at: str | None = None
        self._effective_seed: int | None = None

    # ------------------------------------------------------------------
    # Pipeline stages
    # ------------------------------------------------------------------

    def compute_masks(self) -> None:
        """Load source and target images and compute tissue masks for both."""
        root = self.config.dataset_root
        if not root.exists():
            raise FileNotFoundError(f"Dataset root not found: {root}")

        source_w, source_h = _read_image_size(root / self._source_file.name)

        estimated_gb = _estimate_memory_gb(
            source_h,
            source_w,
            mask_scale=self.config.mask_scale,
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

        self._source_image = cv2.imread(str(root / self._source_file.name))
        self._target_image = cv2.imread(str(root / self._target_file.name))
        if self._source_image is None or self._target_image is None:
            raise FileNotFoundError(
                f"Missing paired files. Expected '{self.config.source_name}' and "
                f"'{self.config.target_name}' inside: {root}"
            )

        logger.info("Calculating masks...")
        scale = self.config.mask_scale
        if scale < 1.0:
            source_h, source_w = self._source_image.shape[:2]
            target_h, target_w = self._target_image.shape[:2]
            scaled_source_size = (max(1, int(source_w * scale)), max(1, int(source_h * scale)))
            scaled_target_size = (max(1, int(target_w * scale)), max(1, int(target_h * scale)))

            small_source = cv2.resize(self._source_image, scaled_source_size)
            small_target = cv2.resize(self._target_image, scaled_target_size)
            source_mask = calculate_mask_with_multiple_parameters(small_source, MASK_PARAMETER_GRID)
            target_mask = calculate_mask_with_multiple_parameters(small_target, MASK_PARAMETER_GRID)
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
            self._source_mask = calculate_mask_with_multiple_parameters(
                self._source_image, MASK_PARAMETER_GRID
            )
            self._target_mask = calculate_mask_with_multiple_parameters(
                self._target_image, MASK_PARAMETER_GRID
            )

        if self.config.save_masks:
            cv2.imwrite(
                str(root / f"mask_{self._source_file.stem}{self._source_suffix}"),
                self._source_mask,
            )
            cv2.imwrite(
                str(root / f"mask_{self._target_file.stem}{self._target_suffix}"),
                self._target_mask,
            )

        _log_memory("compute_masks")

    def align(self) -> None:
        """Align the target image to the source reference frame."""
        if (
            self._source_image is None
            or self._target_image is None
            or self._source_mask is None
            or self._target_mask is None
        ):
            raise RuntimeError("compute_masks() must be called before align()")

        logger.info("Aligning images...")
        warp_matrix, metadata = estimate_affine_from_scaled(
            self._source_image,
            self._target_image,
            mask1=self._source_mask,
            mask2=self._target_mask,
            scale=0.5,
        )
        self._warp_matrix = warp_matrix
        self._alignment_metadata = dataclasses.asdict(metadata)

        root = self.config.dataset_root
        with open(root / "alignment_metadata.json", "w", encoding="utf-8") as f:
            json.dump(self._alignment_metadata, f, indent=2)

        _log_memory("align")

    def _stream_patches_to_disk(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """
        Extract, filter, and write patches in a single pass.

        Returns metadata rows for valid and discarded patches without retaining
        patch arrays on the builder.
        """
        if (
            self._source_image is None
            or self._source_mask is None
            or self._target_image is None
            or self._target_mask is None
            or self._warp_matrix is None
        ):
            raise RuntimeError("align() must be called before _stream_patches_to_disk()")

        root = self.config.dataset_root
        valid_src_dir = root / "processed" / "valid" / "source"
        valid_tgt_dir = root / "processed" / "valid" / "target"
        discarded_src_dir = root / "discarded_patches" / "source"
        discarded_tgt_dir = root / "discarded_patches" / "target"
        for path in [valid_src_dir, valid_tgt_dir]:
            ensure_clean_directory(path)
        if self.config.save_discarded_patches:
            for path in [discarded_src_dir, discarded_tgt_dir]:
                ensure_clean_directory(path)

        m = self.config.margin

        def _crop(img: np.ndarray) -> np.ndarray:
            # img[0:-0] returns an empty array, so treat margin=0 as no crop.
            return img[m:-m, m:-m] if m > 0 else img

        cropped_source = _crop(self._source_image)
        cropped_source_mask = _crop(self._source_mask)
        source_iter = iter_image_with_grid(
            cropped_source,
            self.config.image_size,
            self.config.grid_movement,
            cropped_source_mask,
            max_mask_percentage=self.config.min_foreground_ratio,
        )

        extracted_pairs = 0
        _log_memory("extract_patches")
        valid_rows: list[dict[str, Any]] = []
        discarded_rows: list[dict[str, Any]] = []
        patch_w, patch_h = self.config.image_size
        for (x, y), src, src_mask in source_iter:
            if src_mask is None:
                raise RuntimeError("Patch extraction did not return source masks")
            target_x = x + m
            target_y = y + m
            tgt = warp_aligned_patch(
                self._target_image,
                self._warp_matrix,
                x=target_x,
                y=target_y,
                output_size=(patch_w, patch_h),
                is_mask=False,
            )
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
            patch_source_name = f"{x:05}_{y:05}_source{self._source_suffix}"
            patch_target_name = f"{x:05}_{y:05}_target{self._target_suffix}"

            is_valid, debug_info = is_valid_patch_pair(
                source_img=src,
                target_img=tgt,
                source_mask=src_mask,
                target_mask=tgt_mask,
                min_foreground_ratio=self.config.min_foreground_ratio,
                max_white_ratio=self.config.max_white_ratio,
                white_threshold=self.config.white_threshold,
                max_largest_white_component_ratio=self.config.max_largest_white_component_ratio,
            )
            if is_valid:
                cv2.imwrite(str(valid_src_dir / patch_source_name), src)
                cv2.imwrite(str(valid_tgt_dir / patch_target_name), tgt)
                valid_rows.append(
                    {
                        "x": x,
                        "y": y,
                        "source": patch_source_name,
                        "target": patch_target_name,
                    }
                )
            else:
                if self.config.save_discarded_patches:
                    cv2.imwrite(str(discarded_src_dir / patch_source_name), src)
                    cv2.imwrite(str(discarded_tgt_dir / patch_target_name), tgt)
                discarded_rows.append(
                    {
                        "sample_id": f"{x:05}_{y:05}",
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
        """Assign splits from staged valid patches and write manifests/metadata."""
        root = self.config.dataset_root
        valid_src_dir = root / "processed" / "valid" / "source"
        valid_tgt_dir = root / "processed" / "valid" / "target"
        discarded_root = root / "discarded_patches"
        manifests_dir = root / "manifests"
        metadata_dir = root / "metadata"
        splits_root = root / "splits"

        for path in [
            manifests_dir,
        ]:
            ensure_clean_directory(path)
        for split_name in ("train", "val", "test"):
            ensure_clean_directory(splits_root / split_name)
        metadata_dir.mkdir(parents=True, exist_ok=True)
        discarded_root.mkdir(parents=True, exist_ok=True)

        if self._effective_seed is not None:
            random.seed(self._effective_seed)
        split = split_items(
            valid_rows,
            [self.config.train_ratio, self.config.val_ratio, self.config.test_ratio],
        )
        split_names: tuple[Split, Split, Split] = ("train", "val", "test")
        split_dirs = {
            "train": splits_root / "train",
            "val": splits_root / "val",
            "test": splits_root / "test",
        }
        input_modality = _modality_from_filename(self._source_file)
        target_modality = _modality_from_filename(self._target_file)

        manifest_records: list[ManifestRecord] = []
        for split_name, subset in zip(split_names, split, strict=True):
            subset_dir = split_dirs[split_name]
            for row in subset:
                src_name = cast(str, row["source"])
                tgt_name = cast(str, row["target"])
                x = cast(int, row["x"])
                y = cast(int, row["y"])
                sample_id = f"{x:05}_{y:05}"  # unique only within this single-pair prepare run

                shutil.move(str(valid_src_dir / src_name), str(subset_dir / src_name))
                shutil.move(str(valid_tgt_dir / tgt_name), str(subset_dir / tgt_name))

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
                        width=self.config.image_size[0],
                        height=self.config.image_size[1],
                    )
                )

        discarded_manifest_records: list[ManifestRecord] = []
        for row in discarded_rows:
            src_name = cast(str, row["source_name"])
            tgt_name = cast(str, row["target_name"])
            sample_id = cast(str, row["sample_id"])
            x, y = [int(part) for part in sample_id.split("_")]
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
                    width=self.config.image_size[0],
                    height=self.config.image_size[1],
                )
            )

        manifest = DatasetManifest(records=tuple(manifest_records), dataset_root=root)
        manifest.validate()
        manifest.to_csv(manifests_dir / "manifest.csv")
        (manifests_dir / "manifest_metadata.json").write_text(
            json.dumps(_build_manifest_metadata(manifest_records), indent=2),
            encoding="utf-8",
        )

        discarded_manifest = DatasetManifest(
            records=tuple(discarded_manifest_records),
            dataset_root=root,
        )
        discarded_manifest.validate()
        discarded_manifest.to_csv(manifests_dir / "discarded_manifest.csv")

        with open(discarded_root / "discarded_log.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "sample_id",
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
            "num_train": len(split[0]),
            "num_val": len(split[1]),
            "num_test": len(split[2]),
            "seed": self._effective_seed,
        }
        with open(metadata_dir / "dataset_build.json", "w", encoding="utf-8") as f:
            json.dump(build_metadata, f, indent=2, default=str)

        fingerprint_metadata = build_dataset_fingerprint_metadata(
            dataset_root=root,
            preprocessing_config=serialize_preprocessing_config(self.config),
            source_path=root / self._source_file.name,
            target_path=root / self._target_file.name,
            prepared_at=build_metadata["completed_at"],
        )
        save_dataset_fingerprint(fingerprint_metadata, metadata_dir / "dataset_fingerprint.json")

        logger.info(
            "Saved: train=%s, val=%s, test=%s, discarded=%s",
            len(split[0]),
            len(split[1]),
            len(split[2]),
            len(discarded_rows),
        )
        _log_memory("split_and_save")
        return DatasetBuildResult(
            train_count=len(split[0]),
            val_count=len(split[1]),
            test_count=len(split[2]),
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
