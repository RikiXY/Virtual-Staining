from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np

from virtual_staining.config.data import PreprocessingConfig
from virtual_staining.data.alignment import (
    AlignmentImage,
    AlignmentResult,
    identity_alignment,
    resolve_alignment,
    warp_aligned_mask_patch,
    warp_aligned_patch,
)
from virtual_staining.data.filtering import foreground_ratios, is_valid_patch_pair
from virtual_staining.data.layout import DatasetLayout
from virtual_staining.data.manifest import Split
from virtual_staining.data.patching import iter_patch_origins, mask_window_for_patch
from virtual_staining.data.preprocessing import (
    MASK_PARAMETER_GRID,
    calculate_mask_by_strategy,
    calculate_mask_with_multiple_parameters,
)
from virtual_staining.data.slide_sets import SlideAsset, SlideSet
from virtual_staining.data.splitting import assign_split_by_hash
from virtual_staining.utils.image_io import (
    RegionImageReader,
    load_grayscale_image,
    open_image_reader,
)


@dataclass
class AssetState:
    asset: SlideAsset
    reader: RegionImageReader | None = None
    preview: np.ndarray | None = None
    mask: np.ndarray | None = None
    shape: tuple[int, int] | None = None
    alignment: AlignmentResult | None = None

    def alignment_image(self) -> AlignmentImage:
        if self.preview is None or self.shape is None:
            raise RuntimeError("compute_masks() must be called before align()")
        metadata = self.reader.metadata if self.reader is not None else None
        return AlignmentImage(
            preview=self.preview,
            full_shape=self.shape,
            mask=self.mask,
            name=self.asset.modality,
            mpp=(metadata.mpp_x, metadata.mpp_y) if metadata is not None else (None, None),
        )


@dataclass(frozen=True)
class SetBuildResult:
    """Rows and alignment metadata from one set, independent of open image resources."""

    set_id: str
    split: Split | None
    valid_rows: tuple[dict[str, Any], ...]
    discarded_rows: tuple[dict[str, Any], ...]
    metadata: dict[str, str]
    error: str | None = None


class SlideSetProcessor:
    """Mask, align and write patches for exactly one slide set."""

    def __init__(
        self, config: PreprocessingConfig, slide_set: SlideSet, assigned_split: Split | None = None
    ) -> None:
        self.config = config
        self.slide_set = slide_set
        self.assigned_split: Split | None = assigned_split
        self.inputs = {asset.modality: AssetState(asset) for asset in slide_set.inputs}
        self.target = AssetState(slide_set.target)
        self.reference = self.inputs[slide_set.reference_modality]
        self._maskless = False

    def process(self) -> SetBuildResult:
        valid_rows: list[dict[str, Any]] = []
        discarded_rows: list[dict[str, Any]] = []
        error: str | None = None
        try:
            self.compute_masks()
            self.align()
            valid_rows, discarded_rows = self.stream_patches()
        except Exception as exc:
            if self.config.alignment.on_failure != "skip_set":
                raise
            error = str(exc)
        finally:
            self.close()
        metadata = {}
        for name, state in (*self.inputs.items(), ("target", self.target)):
            metadata[f"{name}__alignment_method"] = (
                state.alignment.method if state.alignment else ""
            )
            metadata[f"{name}__alignment_metadata"] = (
                json.dumps(state.alignment.metadata, sort_keys=True) if state.alignment else ""
            )
        return SetBuildResult(
            set_id=self.slide_set.set_id,
            split=self.assigned_split,
            valid_rows=tuple(valid_rows),
            discarded_rows=tuple(discarded_rows),
            metadata=metadata,
            error=error,
        )

    def _states(self) -> tuple[AssetState, ...]:
        return (*self.inputs.values(), self.target)

    def _calculate_mask(self, image: np.ndarray) -> np.ndarray:
        strategy = self.config.masks.strategy
        mask = (
            calculate_mask_with_multiple_parameters(image, MASK_PARAMETER_GRID)
            if strategy == "connected_components"
            else calculate_mask_by_strategy(
                image, strategy=strategy, parameters=MASK_PARAMETER_GRID
            )
        )
        mask[np.all(image == 0, axis=2)] = 0
        return mask

    def compute_masks(self) -> None:
        root = self.config.dataset_root
        if not root.is_dir():
            raise FileNotFoundError(f"Dataset root not found: {root}")
        for state in self._states():
            path = root / state.asset.path
            if self.config.io.tiled:
                state.reader = open_image_reader(path, backend=self.config.io.backend)
                width, height = state.reader.size
                state.shape = (height, width)
                state.preview = state.reader.read_preview(self.config.masks.scale)
            else:
                reader = open_image_reader(path, backend="pillow")
                try:
                    metadata = reader.metadata
                    state.shape = (metadata.height, metadata.width)
                    state.preview = reader.read_full()
                finally:
                    reader.close()
            if state.asset.mask_path is not None:
                try:
                    state.mask = load_grayscale_image(root / state.asset.mask_path)
                except (FileNotFoundError, RuntimeError) as exc:
                    raise ValueError(f"Could not read mask {state.asset.mask_path}") from exc
            elif self.config.masks.generation == "never":
                if self.config.filtering.foreground.enabled:
                    raise ValueError("maskless processing requires foreground.enabled=false")
                state.mask = np.full(state.preview.shape[:2], 255, dtype=np.uint8)
                self._maskless = True
            else:
                state.mask = self._calculate_mask(state.preview)

    def align(self) -> None:
        reference = self.reference.alignment_image()
        self.reference.alignment = identity_alignment()
        for state in self._states():
            if state is not self.reference:
                state.alignment = resolve_alignment(
                    reference,
                    state.alignment_image(),
                    self.config.alignment,
                    already_aligned=state.asset.already_aligned,
                )

    def extract_asset_patch(
        self, state: AssetState, *, x: int, y: int, width: int, height: int
    ) -> tuple[np.ndarray, np.ndarray]:
        if state.alignment is None or state.shape is None or state.mask is None:
            raise RuntimeError("compute_masks() and align() must be called before extraction")
        size = (width, height)
        if state.alignment.method == "identity":
            if state.reader is not None:
                image = state.reader.read_region(x, y, width, height)
            elif state.preview is not None:
                image = state.preview[y : y + height, x : x + width]
            else:
                raise RuntimeError("Asset preview must be loaded before extraction")
            mask_window = mask_window_for_patch(
                state.mask, state.shape, x=x, y=y, width=width, height=height
            )
            mask = cv2.resize(mask_window, size, interpolation=cv2.INTER_NEAREST)
        else:
            image_input = state.reader.read_region if state.reader is not None else state.preview
            if image_input is None:
                raise RuntimeError("Asset preview must be loaded before extraction")
            image = warp_aligned_patch(
                image_input,
                state.alignment.warp_matrix,
                x=x,
                y=y,
                output_size=size,
            )
            mask = warp_aligned_mask_patch(
                state.mask, state.alignment.warp_matrix, state.shape, x=x, y=y, output_size=size
            )
        if image.shape[:2] != (height, width) or mask.shape[:2] != (height, width):
            raise RuntimeError(
                f"Patch extraction mismatch for {state.asset.modality}: {image.shape}, {mask.shape}"
            )
        return image, mask

    def stream_patches(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if self.reference.shape is None or self.reference.alignment is None:
            raise RuntimeError("align() must be called before stream_patches()")
        patch_w, patch_h = self.config.patching.patch_size
        step_x, step_y = self.config.patching.grid_movement
        margin = self.config.patching.margin
        ref_h, ref_w = self.reference.shape
        layout = DatasetLayout(self.config.dataset_root)
        split_dirs = {
            name: layout.split_dir(name) / self.slide_set.set_id
            for name in ("train", "val", "test")
        }
        for path in split_dirs.values():
            path.mkdir(parents=True, exist_ok=True)
        discarded_dirs = {
            name: layout.discarded_patches_dir / self.slide_set.set_id / name
            for name in (*self.inputs, "target")
        }
        if self.config.patching.save_discarded_patches:
            for path in discarded_dirs.values():
                path.mkdir(parents=True, exist_ok=True)
        valid: list[dict[str, Any]] = []
        discarded: list[dict[str, Any]] = []
        foreground = self.config.filtering.foreground
        foreground_policy = (
            self.slide_set.reference_modality
            if foreground.policy == "reference"
            else foreground.policy
        )
        for x, y in iter_patch_origins(
            image_size=(ref_w, ref_h),
            patch_size=(patch_w, patch_h),
            grid_movement=(step_x, step_y),
            margin=margin,
        ):
            patches, masks = {}, {}
            for name, state in (*self.inputs.items(), ("target", self.target)):
                patches[name], masks[name] = self.extract_asset_patch(
                    state, x=x, y=y, width=patch_w, height=patch_h
                )
            ratios = foreground_ratios(masks)
            source = patches[self.slide_set.reference_modality]
            target = patches["target"]
            is_valid, debug = is_valid_patch_pair(
                source_img=source,
                target_img=target,
                source_mask=masks[self.slide_set.reference_modality],
                target_mask=masks["target"],
                min_foreground_ratio=0.0,
                max_white_ratio=self.config.filtering.max_white_ratio,
                white_threshold=self.config.filtering.white_threshold,
                max_largest_white_component_ratio=self.config.filtering.max_largest_white_component_ratio,
            )
            if foreground.enabled and ratios[foreground_policy] < foreground.min_ratio:
                is_valid = False
                cast(list[str], debug["reasons"]).append(f"foreground_{foreground.policy}")
            sample_id = f"{self.slide_set.set_id}__x{x:08}_y{y:08}"
            suffixes = {
                name: Path(state.asset.path).suffix.lower()
                for name, state in (*self.inputs.items(), ("target", self.target))
            }
            names = {name: f"{sample_id}__input__{name}{suffixes[name]}" for name in self.inputs}
            names["target"] = f"{sample_id}__target{suffixes['target']}"
            names["foreground_mask"] = f"{sample_id}__foreground_mask{suffixes['target']}"
            row = {
                "sample_id": sample_id,
                "x": x,
                "y": y,
                "inputs": names,
                "target": names["target"],
                "foreground_mask": names["foreground_mask"],
            }
            if is_valid:
                split = self.assigned_split or assign_split_by_hash(
                    seed=self.config.split.seed,
                    sample_id=sample_id,
                    ratios=(
                        self.config.split.train,
                        self.config.split.val,
                        self.config.split.test,
                    ),
                )
                for modality, image in patches.items():
                    cv2.imwrite(str(split_dirs[split] / names[modality]), image)
                if self.config.masks.save_patch_masks and not self._maskless:
                    cv2.imwrite(str(split_dirs[split] / names["foreground_mask"]), masks["target"])
                valid.append({**row, "split": split})
            else:
                if self.config.patching.save_discarded_patches:
                    for modality, image in patches.items():
                        cv2.imwrite(str(discarded_dirs[modality] / names[modality]), image)
                discarded.append(
                    {
                        **row,
                        "ratios": ratios,
                        "reasons": ";".join(cast(list[str], debug["reasons"])),
                    }
                )
        return valid, discarded

    def close(self) -> None:
        for state in self._states():
            if state.reader is not None:
                state.reader.close()
