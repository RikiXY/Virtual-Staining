from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
from PIL import Image

from virtual_staining.config.data import PreprocessingConfig
from virtual_staining.data.layout import DatasetLayout
from virtual_staining.data.manifest import Split
from virtual_staining.data.preprocessing import (
    MASK_PARAMETER_GRID,
    assign_split_by_hash,
    calculate_mask_by_strategy,
    calculate_mask_with_multiple_parameters,
    estimate_affine_from_scaled,
    is_valid_patch_pair,
    mask_window_for_patch,
    warp_aligned_mask_patch_from_mask_space,
    warp_aligned_patch,
)
from virtual_staining.data.slide_sets import SlideAsset, SlideSet
from virtual_staining.utils.image_io import RegionImageReader, open_image_reader


@dataclass(frozen=True)
class AlignmentResult:
    method: str
    warp_matrix: np.ndarray
    metadata: dict[str, Any]


@dataclass
class AssetState:
    asset: SlideAsset
    reader: RegionImageReader | None = None
    preview: np.ndarray | None = None
    mask: np.ndarray | None = None
    shape: tuple[int, int] | None = None
    alignment: AlignmentResult | None = None


@dataclass(frozen=True)
class SetBuildResult:
    """Rows and alignment metadata from one set, independent of open image resources."""

    set_id: str
    split: Split | None
    valid_rows: tuple[dict[str, Any], ...]
    discarded_rows: tuple[dict[str, Any], ...]
    metadata: dict[str, str]
    error: str | None = None


def _identity() -> np.ndarray:
    return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)


def _mask_to_preview_space(mask: np.ndarray, preview: np.ndarray) -> np.ndarray:
    if mask.shape == preview.shape[:2]:
        return mask
    return cv2.resize(mask, (preview.shape[1], preview.shape[0]), interpolation=cv2.INTER_NEAREST)


def _validate_identity(reference: AssetState, moving: AssetState, policy: Any) -> None:
    if not policy.validate_declared:
        return
    if reference.shape != moving.shape:
        raise ValueError(f"identity alignment requires equal geometry for {moving.asset.modality}")
    if reference.reader is None or moving.reader is None:
        return
    for name in ("mpp_x", "mpp_y"):
        left, right = (
            getattr(reference.reader.metadata, name, None),
            getattr(moving.reader.metadata, name, None),
        )
        if left is not None and right is not None and not np.isclose(left, right, rtol=0.01):
            raise ValueError(
                f"identity alignment has incompatible {name} for {moving.asset.modality}"
            )


def resolve_alignment(reference: AssetState, moving: AssetState, policy: Any) -> AlignmentResult:
    if (
        reference.shape is None
        or moving.shape is None
        or reference.preview is None
        or moving.preview is None
    ):
        raise RuntimeError("Asset masks and previews must be loaded before alignment")
    declared = moving.asset.already_aligned
    estimate = declared is not True and (declared is False or policy.mode in {"auto", "always"})
    if policy.mode == "never" and declared is False:
        raise ValueError(
            f"alignment.mode=never contradicts already_aligned=false for {moving.asset.modality}"
        )
    if not estimate:
        _validate_identity(reference, moving, policy)
        matrix = _identity()
        return AlignmentResult(
            "identity",
            matrix,
            {
                "method": "identity",
                "reason": "declared_aligned" if declared is True else "policy_never",
                "warp_matrix": matrix.tolist(),
            },
        )
    if reference.mask is None or moving.mask is None:
        raise ValueError(f"affine registration requires masks for {moving.asset.modality}")
    matrix, metadata = estimate_affine_from_scaled(
        reference.preview,
        moving.preview,
        mask_1=_mask_to_preview_space(reference.mask, reference.preview),
        mask_2=_mask_to_preview_space(moving.mask, moving.preview),
        scale=0.5,
    )
    preview_scale = reference.preview.shape[1] / reference.shape[1]
    if preview_scale != 1.0:
        matrix = np.asarray(matrix, dtype=np.float64).copy()
        matrix[:, 2] /= preview_scale
        metadata.translation_x = float(matrix[0, 2])
        metadata.translation_y = float(matrix[1, 2])
    metadata_dict = dataclasses.asdict(metadata)
    metadata_dict.update({"method": "affine_sift", "warp_matrix": matrix.tolist()})
    return AlignmentResult("affine_sift", matrix, metadata_dict)


def _read_image_size(path: Path) -> tuple[int, int]:
    original = Image.MAX_IMAGE_PIXELS
    try:
        Image.MAX_IMAGE_PIXELS = None
        with Image.open(path) as image:
            return image.size[1], image.size[0]
    finally:
        Image.MAX_IMAGE_PIXELS = original


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
        """Process the set under its failure policy and always close its readers."""
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
                state.shape = _read_image_size(path)
                state.preview = cv2.imread(str(path))
                if state.preview is None:
                    raise FileNotFoundError(f"Image not found: {path}")
            if state.asset.mask_path is not None:
                supplied = cv2.imread(str(root / state.asset.mask_path), cv2.IMREAD_GRAYSCALE)
                if supplied is None:
                    raise ValueError(f"Could not read mask {state.asset.mask_path}")
                state.mask = supplied
            elif self.config.masks.generation == "never":
                if self.config.filtering.foreground.enabled:
                    raise ValueError("maskless processing requires foreground.enabled=false")
                state.mask = np.full(state.preview.shape[:2], 255, dtype=np.uint8)
                self._maskless = True
            else:
                state.mask = self._calculate_mask(state.preview)

    def align(self) -> None:
        if self.reference.preview is None:
            raise RuntimeError("compute_masks() must be called before align()")
        self.reference.alignment = AlignmentResult(
            "identity",
            _identity(),
            {"method": "identity", "reason": "reference", "warp_matrix": _identity().tolist()},
        )
        for state in (*self.inputs.values(), self.target):
            if state is not self.reference:
                state.alignment = resolve_alignment(self.reference, state, self.config.alignment)

    @staticmethod
    def _warp_reader_patch(
        state: AssetState, *, x: int, y: int, size: tuple[int, int]
    ) -> np.ndarray:
        assert state.reader is not None and state.alignment is not None
        matrix = state.alignment.warp_matrix
        inverse = cv2.invertAffineTransform(matrix)
        width, height = size
        corners = cv2.transform(
            np.array(
                [[[x, y], [x + width, y], [x, y + height], [x + width, y + height]]],
                dtype=np.float64,
            ),
            inverse,
        )[0]
        rx, ry = int(np.floor(corners[:, 0].min())) - 2, int(np.floor(corners[:, 1].min())) - 2
        rw, rh = (
            max(1, int(np.ceil(corners[:, 0].max())) + 2 - rx),
            max(1, int(np.ceil(corners[:, 1].max())) + 2 - ry),
        )
        region = state.reader.read_region(rx, ry, rw, rh)
        local = matrix.copy()
        local[:, 2] += local[:, :2] @ np.array([rx, ry])
        return warp_aligned_patch(region, local, x=x, y=y, output_size=size, is_mask=False)

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
        elif state.reader is not None:
            image = self._warp_reader_patch(state, x=x, y=y, size=size)
            mask = warp_aligned_mask_patch_from_mask_space(
                state.mask, state.alignment.warp_matrix, state.shape, x=x, y=y, output_size=size
            )
        else:
            assert state.preview is not None
            image = warp_aligned_patch(
                state.preview,
                state.alignment.warp_matrix,
                x=x,
                y=y,
                output_size=size,
                is_mask=False,
            )
            mask = warp_aligned_patch(
                state.mask, state.alignment.warp_matrix, x=x, y=y, output_size=size, is_mask=True
            )
        if image.shape[:2] != (height, width) or mask.shape[:2] != (height, width):
            raise RuntimeError(
                f"Patch extraction mismatch for {state.asset.modality}: {image.shape}, {mask.shape}"
            )
        return image, mask

    def _foreground_ratios(self, masks: dict[str, np.ndarray]) -> dict[str, float]:
        ratios = {name: float(cv2.countNonZero(mask) / mask.size) for name, mask in masks.items()}
        ratios["all"] = min(ratios.values())
        combined_intersection = masks[next(iter(masks))]
        combined_union = masks[next(iter(masks))]
        for mask in list(masks.values())[1:]:
            combined_intersection = cv2.bitwise_and(combined_intersection, mask)
            combined_union = cv2.bitwise_or(combined_union, mask)
        ratios["intersection"] = float(
            cv2.countNonZero(combined_intersection) / combined_intersection.size
        )
        ratios["union"] = float(cv2.countNonZero(combined_union) / combined_union.size)
        return ratios

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
        for x in range(margin, max(margin, ref_w - margin - patch_w + 1), step_x):
            for y in range(margin, max(margin, ref_h - margin - patch_h + 1), step_y):
                patches, masks = {}, {}
                for name, state in (*self.inputs.items(), ("target", self.target)):
                    patches[name], masks[name] = self.extract_asset_patch(
                        state, x=x, y=y, width=patch_w, height=patch_h
                    )
                ratios = self._foreground_ratios(masks)
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
                names = {
                    name: f"{sample_id}__input__{name}{suffixes[name]}" for name in self.inputs
                }
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
                        cv2.imwrite(
                            str(split_dirs[split] / names["foreground_mask"]), masks["target"]
                        )
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
