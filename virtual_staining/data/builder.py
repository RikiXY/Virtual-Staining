from __future__ import annotations

import csv
import random

import cv2
import numpy as np

from virtual_staining.data.config import PreprocessingConfig
from virtual_staining.utils.env import collect_environment
from virtual_staining.data.preprocessing import (
    MASK_PARAMETER_GRID,
    align_from_scaled,
    calculate_mask_with_multiple_parameters,
    divide_image_with_grid,
    divide_image_with_positions,
    ensure_clean_directory,
    is_valid_patch_pair,
    split_items,
    validate_image_filename,
)
from virtual_staining.data.results import DatasetBuildResult


class DatasetBuilder:
    """
    Orchestrates the full dataset preparation pipeline.

    Owns the config, loaded images, and all stage-to-stage intermediate results.
    Low-level image operations are delegated to pure functions in preprocessing.py.

    Call run_all() to execute the complete pipeline, or call individual stage
    methods in order (compute_masks → align → extract_patches → filter_patches
    → split_and_save) to run incrementally.
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
        self._aligned_target: np.ndarray | None = None
        self._aligned_target_mask: np.ndarray | None = None
        self._source_patches: list[np.ndarray] | None = None
        self._source_patch_masks: list[np.ndarray] | None = None
        self._target_patches: list[np.ndarray] | None = None
        self._target_patch_masks: list[np.ndarray] | None = None
        self._positions: list[tuple[int, int]] | None = None
        self._named_source_images: list[tuple[np.ndarray, str]] | None = None
        self._named_target_images: list[tuple[np.ndarray, str]] | None = None
        self._discarded_source_images: list[tuple[np.ndarray, str]] | None = None
        self._discarded_target_images: list[tuple[np.ndarray, str]] | None = None
        self._discarded_log_rows: list[dict] | None = None

    # ------------------------------------------------------------------
    # Pipeline stages
    # ------------------------------------------------------------------

    def compute_masks(self) -> None:
        """Load source and target images and compute tissue masks for both."""
        root = self.config.dataset_root
        if not root.exists():
            raise FileNotFoundError(f"Dataset root not found: {root}")

        self._source_image = cv2.imread(str(root / self._source_file.name))
        self._target_image = cv2.imread(str(root / self._target_file.name))
        if self._source_image is None or self._target_image is None:
            raise FileNotFoundError(
                f"Missing paired files. Expected '{self.config.source_name}' and "
                f"'{self.config.target_name}' inside: {root}"
            )

        print("Calculating masks...")
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

    def align(self) -> None:
        """Align the target image to the source reference frame."""
        if self._source_mask is None:
            raise RuntimeError("compute_masks() must be called before align()")

        print("Aligning images...")
        aligned_target, aligned_target_mask, _ = align_from_scaled(
            self._source_image,
            self._target_image,
            mask1=self._source_mask,
            mask2=self._target_mask,
            scale=0.5,
        )
        self._aligned_target = aligned_target
        self._aligned_target_mask = aligned_target_mask

        root = self.config.dataset_root
        stem = self._target_file.stem
        cv2.imwrite(str(root / f"aligned_{stem}{self._target_suffix}"), aligned_target)
        cv2.imwrite(
            str(root / f"aligned_mask_{stem}{self._target_suffix}"),
            aligned_target_mask,
        )

    def extract_patches(self) -> None:
        """Extract paired patches from the source and aligned target images."""
        if self._aligned_target is None:
            raise RuntimeError("align() must be called before extract_patches()")

        m = self.config.margin

        def _crop(img: np.ndarray) -> np.ndarray:
            # img[0:-0] returns an empty array, so treat margin=0 as no crop.
            return img[m:-m, m:-m] if m > 0 else img

        source_images, source_masks, positions = divide_image_with_grid(
            _crop(self._source_image),
            self.config.image_size,
            self.config.grid_movement,
            _crop(self._source_mask),
        )
        target_images = divide_image_with_positions(
            _crop(self._aligned_target),
            self.config.image_size,
            positions,
        )
        target_masks = divide_image_with_positions(
            _crop(self._aligned_target_mask),
            self.config.image_size,
            positions,
        )
        print(f"Extracted {len(source_images)} patch pairs")
        self._source_patches = source_images
        self._source_patch_masks = source_masks
        self._target_patches = target_images
        self._target_patch_masks = target_masks
        self._positions = positions

    def filter_patches(self) -> None:
        """Classify each patch pair as valid or discarded based on quality thresholds."""
        if self._positions is None:
            raise RuntimeError(
                "extract_patches() must be called before filter_patches()"
            )

        named_source: list[tuple[np.ndarray, str]] = []
        named_target: list[tuple[np.ndarray, str]] = []
        discarded_source: list[tuple[np.ndarray, str]] = []
        discarded_target: list[tuple[np.ndarray, str]] = []
        log_rows: list[dict] = []

        for (x, y), src, src_mask, tgt, tgt_mask in zip(
            self._positions,
            self._source_patches,
            self._source_patch_masks,
            self._target_patches,
            self._target_patch_masks,
        ):
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
                named_source.append((src, patch_source_name))
                named_target.append((tgt, patch_target_name))
            else:
                discarded_source.append((src, patch_source_name))
                discarded_target.append((tgt, patch_target_name))
                log_rows.append(
                    {
                        "sample_id": f"{x:05}_{y:05}",
                        "source_name": patch_source_name,
                        "target_name": patch_target_name,
                        "source_foreground_ratio": debug_info[
                            "source_foreground_ratio"
                        ],
                        "target_foreground_ratio": debug_info[
                            "target_foreground_ratio"
                        ],
                        "source_white_ratio": debug_info["source_white_ratio"],
                        "target_white_ratio": debug_info["target_white_ratio"],
                        "source_largest_white_component_ratio": debug_info[
                            "source_largest_white_component_ratio"
                        ],
                        "target_largest_white_component_ratio": debug_info[
                            "target_largest_white_component_ratio"
                        ],
                        "reasons": ";".join(debug_info["reasons"]),
                    }
                )

        self._named_source_images = named_source
        self._named_target_images = named_target
        self._discarded_source_images = discarded_source
        self._discarded_target_images = discarded_target
        self._discarded_log_rows = log_rows

    def split_and_save(self) -> DatasetBuildResult:
        """Split valid pairs into train/val/test and write all output files."""
        if self._named_source_images is None:
            raise RuntimeError(
                "filter_patches() must be called before split_and_save()"
            )

        root = self.config.dataset_root
        discarded_root = root / "discarded_patches"

        for d in [
            root / "dataset_train",
            root / "dataset_val",
            root / "dataset_test",
            discarded_root / "source",
            discarded_root / "target",
        ]:
            ensure_clean_directory(d)

        for img, name in self._discarded_source_images:
            cv2.imwrite(str(discarded_root / "source" / name), img)
        for img, name in self._discarded_target_images:
            cv2.imwrite(str(discarded_root / "target" / name), img)

        with open(
            discarded_root / "discarded_log.csv", "w", newline="", encoding="utf-8"
        ) as f:
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
            writer.writerows(self._discarded_log_rows)

        pairs = list(zip(self._named_source_images, self._named_target_images))
        split = split_items(
            pairs,
            [self.config.train_ratio, self.config.val_ratio, self.config.test_ratio],
        )

        for i, subset in enumerate(split):
            subset_dir = root / ["dataset_train", "dataset_val", "dataset_test"][i]
            for src_pair, tgt_pair in subset:
                cv2.imwrite(str(subset_dir / src_pair[1]), src_pair[0])
                cv2.imwrite(str(subset_dir / tgt_pair[1]), tgt_pair[0])

        print(
            f"Saved: train={len(split[0])}, val={len(split[1])}, "
            f"test={len(split[2])}, discarded={len(self._discarded_source_images)}"
        )
        return DatasetBuildResult(
            train_count=len(split[0]),
            val_count=len(split[1]),
            test_count=len(split[2]),
            skipped_count=len(self._discarded_source_images),
            output_root=root,
        )

    def run_all(self) -> DatasetBuildResult:
        """Run all pipeline stages in sequence and return the build result."""
        seed = (
            self.config.seed
            if self.config.seed is not None
            else random.randint(0, 2**32 - 1)
        )
        random.seed(seed)
        print(f"Seed set to {seed}")

        import json

        root = self.config.dataset_root
        self.config.to_yaml(root / "config.yaml")
        with open(root / "environment.json", "w", encoding="utf-8") as f:
            json.dump(collect_environment(), f, indent=2, default=str)

        self.compute_masks()
        self.align()
        self.extract_patches()
        self.filter_patches()
        return self.split_and_save()
