from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np

from virtual_staining.config.data import PreprocessingConfig
from virtual_staining.data.alignment import (
    AlignmentError,
    AlignmentImage,
    AlignmentResult,
    GridGeometry,
    ImageGeometry,
    RegistrationBackend,
    RegistrationRequest,
    identity_alignment,
    resolve_alignment,
    warp_aligned_mask_patch,
    warp_aligned_patch,
)
from virtual_staining.data.filtering import foreground_ratios, is_valid_patch_pair
from virtual_staining.data.layout import DatasetLayout
from virtual_staining.data.patching import iter_patch_origins, mask_window_for_patch
from virtual_staining.data.preprocessing import (
    MASK_PARAMETER_GRID,
    calculate_mask_by_strategy,
    calculate_mask_with_multiple_parameters,
)
from virtual_staining.data.slide_sets import SlideAsset, SlideSet
from virtual_staining.data.splitting import assign_split_by_hash
from virtual_staining.split_contract import DATASET_SPLITS, DatasetSplit
from virtual_staining.utils.image_io import (
    RegionImageReader,
    load_grayscale_image,
    open_image_reader,
    read_image_metadata,
)


@dataclass
class AssetState:
    asset: SlideAsset
    reader: RegionImageReader | None = None
    preview: np.ndarray | None = None
    mask: np.ndarray | None = None
    shape: tuple[int, int] | None = None
    alignment: AlignmentResult | None = None
    mpp: tuple[float | None, float | None] = (None, None)

    def alignment_image(self) -> AlignmentImage:
        if self.preview is None or self.shape is None:
            raise RuntimeError("compute_masks() must be called before align()")
        return AlignmentImage(
            preview=self.preview,
            geometry=ImageGeometry(
                self.asset.modality,
                self.shape,
                self.mpp,
            ),
            grid=GridGeometry.resized_crop(
                self.preview.shape[:2],
                origin=(0, 0),
                scale=(
                    self.shape[1] / self.preview.shape[1],
                    self.shape[0] / self.preview.shape[0],
                ),
            ),
        )


def _verify_written_patch(path: Path, image: np.ndarray) -> None:
    """Require ``path`` to be a readable image with ``image``'s width and height."""
    try:
        metadata = read_image_metadata(path, backend="pillow")
    except (OSError, ValueError) as exc:
        raise OSError(f"Written patch {path} is not a readable image: {exc}") from exc
    expected = (image.shape[1], image.shape[0])
    if (metadata.width, metadata.height) != expected:
        raise OSError(
            f"Written patch {path} is {metadata.width}x{metadata.height}, expected "
            f"{expected[0]}x{expected[1]}"
        )


@dataclass(frozen=True)
class SetBuildResult:
    """Rows and alignment metadata from one set, independent of open image resources."""

    set_id: str
    split: DatasetSplit | None
    valid_rows: tuple[dict[str, Any], ...]
    discarded_rows: tuple[dict[str, Any], ...]
    metadata: dict[str, str]
    error: str | None = None


class AlignmentQCError(AlignmentError):
    """Explicit preparation QC disposition; backend execution may have succeeded."""

    def __init__(self, result: AlignmentResult, action: str) -> None:
        self.result, self.action = result, action
        status = result.qc.status if result.qc is not None else "unassessed"
        super().__init__(f"Registration QC {status}: {action}")


class SlideSetProcessor:
    """Mask, align and write patches for exactly one slide set."""

    def __init__(
        self,
        config: PreprocessingConfig,
        slide_set: SlideSet,
        assigned_split: DatasetSplit | None = None,
        *,
        registration_backend: RegistrationBackend | None = None,
    ) -> None:
        self.config = config
        self.slide_set = slide_set
        self.assigned_split: DatasetSplit | None = assigned_split
        self.registration_backend = registration_backend
        self.inputs = {asset.modality: AssetState(asset) for asset in slide_set.inputs}
        self.targets = {asset.modality: AssetState(asset) for asset in slide_set.targets}
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
        except AlignmentQCError as exc:
            if exc.action == "error":
                raise
            error = str(exc)
        except Exception as exc:
            if self.config.alignment.on_failure != "skip_set":
                raise
            error = str(exc)
        finally:
            self.close()
        metadata = {}
        for name, state in (*self.inputs.items(), *self.targets.items()):
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
        return (*self.inputs.values(), *self.targets.values())

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
                metadata = state.reader.metadata
                state.mpp = (metadata.mpp_x, metadata.mpp_y)
                state.preview = state.reader.read_preview(self.config.masks.scale)
            else:
                reader = open_image_reader(path, backend="pillow")
                try:
                    metadata = reader.metadata
                    state.shape = (metadata.height, metadata.width)
                    state.mpp = (metadata.mpp_x, metadata.mpp_y)
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
        self.reference.alignment = identity_alignment(
            reference.geometry,
            reference.geometry,
            RegistrationRequest("same_coordinate_frame", "identity"),
            reason="reference",
        )
        policy = self.config.alignment
        for state in self._states():
            if state is self.reference:
                continue
            moving = state.alignment_image()
            declared = state.asset.already_aligned
            estimate = declared is not True and (
                declared is False or policy.mode in {"auto", "always"}
            )
            if policy.mode == "never" and declared is False:
                raise AlignmentError("alignment.mode=never contradicts already_aligned=false")
            if not estimate and policy.validate_declared:
                reference.geometry.validate_shared_frame(moving.geometry)
            # Inventory alignment flags make no biological declaration or QC claim.
            request = RegistrationRequest(
                "unknown",
                "affine" if estimate else "identity",
                existing_alignment="identity"
                if declared is True
                else "unaligned"
                if declared is False
                else "unknown",
                diagnostic_region=(0, 0, reference.geometry.shape[1], reference.geometry.shape[0]),
            )
            backend = self.registration_backend or resolve_alignment
            result = backend(reference, moving, request)
            state.alignment = result
            if result.backend_status == "failed":
                assert result.attempt.failure is not None
                raise AlignmentError(result.attempt.failure.message)
            if self.registration_backend is None:
                state.alignment = replace(
                    result,
                    reason=(
                        None if estimate else "declared_aligned" if declared else "policy_never"
                    ),
                )
            else:
                status = result.qc.status if result.qc is not None else "unassessed"
                action = self.registration_backend.metadata["qc_disposition"].get(
                    status, "continue"
                )
                if action != "continue":
                    raise AlignmentQCError(result, action)

    def extract_asset_patch(
        self, state: AssetState, *, x: int, y: int, width: int, height: int
    ) -> tuple[np.ndarray, np.ndarray]:
        if state.alignment is None or state.shape is None or state.mask is None:
            raise RuntimeError("compute_masks() and align() must be called before extraction")
        if state.alignment.candidate is None:
            raise AlignmentError("Patch extraction requires a transform candidate")
        size = (width, height)
        source_budget = max(4, width * height)
        if (
            state.alignment.candidate.family == "identity"
            and state.alignment.candidate.moving.shape == state.alignment.candidate.reference.shape
        ):
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
                state.alignment.candidate,
                x=x,
                y=y,
                output_size=size,
                max_source_pixels=source_budget,
            ).image
            mask = warp_aligned_mask_patch(
                state.mask,
                state.alignment.candidate,
                GridGeometry.resized_crop(
                    state.mask.shape,
                    origin=(0, 0),
                    scale=(
                        state.shape[1] / state.mask.shape[1],
                        state.shape[0] / state.mask.shape[0],
                    ),
                ),
                x=x,
                y=y,
                output_size=size,
                max_source_pixels=source_budget,
            )
        if image.shape[:2] != (height, width) or mask.shape[:2] != (height, width):
            raise RuntimeError(
                f"Patch extraction mismatch for {state.asset.modality}: {image.shape}, {mask.shape}"
            )
        return image, mask

    def _is_valid_patch(
        self, patches: dict[str, np.ndarray], masks: dict[str, np.ndarray]
    ) -> tuple[bool, list[str]]:
        """Check the reference against every target; a sample needs all targets valid."""
        reference = self.slide_set.reference_modality
        reasons: list[str] = []
        for name in self.targets:
            _, debug = is_valid_patch_pair(
                source_img=patches[reference],
                target_img=patches[name],
                source_mask=masks[reference],
                target_mask=masks[name],
                min_foreground_ratio=0.0,
                max_white_ratio=self.config.filtering.max_white_ratio,
                white_threshold=self.config.filtering.white_threshold,
                max_largest_white_component_ratio=self.config.filtering.max_largest_white_component_ratio,
            )
            reasons.extend(r for r in cast(list[str], debug["reasons"]) if r not in reasons)
        foreground = self.config.filtering.foreground
        if foreground.enabled:
            ratios = foreground_ratios(masks)
            if foreground.policy == "reference":
                ratio = ratios[reference]
            elif foreground.policy == "target":
                ratio = min(ratios[name] for name in self.targets)
            else:
                ratio = ratios[foreground.policy]
            if ratio < foreground.min_ratio:
                reasons.append(f"foreground_{foreground.policy}")
        return not reasons, reasons

    def _write_sample(self, images: dict[Path, np.ndarray]) -> None:
        """Materialize every image of one sample or none.

        Each file is written, then read back far enough to prove its stored width and
        height equal the array's. Any failure removes every file this attempt wrote for
        the sample (including a corrupt one) and raises, so no row is committed for it.
        """
        owned: list[Path] = []
        try:
            for path, image in images.items():
                existed = path.exists()
                if not existed:
                    owned.append(path)
                written = cv2.imwrite(str(path), image)
                if written and existed:
                    owned.append(path)
                if not written:
                    raise OSError(f"Could not write patch {path}")
                _verify_written_patch(path, image)
        except BaseException:
            for path in owned:
                path.unlink(missing_ok=True)
            raise

    def stream_patches(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if self.reference.shape is None or self.reference.alignment is None:
            raise RuntimeError("align() must be called before stream_patches()")
        patch_w, patch_h = self.config.patching.patch_size
        step_x, step_y = self.config.patching.grid_movement
        margin = self.config.patching.margin
        ref_h, ref_w = self.reference.shape
        layout = DatasetLayout(self.config.dataset_root)
        split_dirs = {
            name: layout.split_dir(name) / self.slide_set.set_id for name in DATASET_SPLITS
        }
        for path in split_dirs.values():
            path.mkdir(parents=True, exist_ok=True)
        states = {**self.inputs, **self.targets}
        discarded_dirs = {
            name: layout.discarded_patches_dir / self.slide_set.set_id / name for name in states
        }
        if self.config.patching.save_discarded_patches:
            for path in discarded_dirs.values():
                path.mkdir(parents=True, exist_ok=True)
        save_masks = self.config.masks.save_patch_masks and not self._maskless
        suffixes = {name: Path(state.asset.path).suffix.lower() for name, state in states.items()}
        valid: list[dict[str, Any]] = []
        discarded: list[dict[str, Any]] = []
        for x, y in iter_patch_origins(
            image_size=(ref_w, ref_h),
            patch_size=(patch_w, patch_h),
            grid_movement=(step_x, step_y),
            margin=margin,
        ):
            # Every asset is extracted on the same reference-frame grid position.
            patches, masks = {}, {}
            for name, state in states.items():
                patches[name], masks[name] = self.extract_asset_patch(
                    state, x=x, y=y, width=patch_w, height=patch_h
                )
            is_valid, reasons = self._is_valid_patch(patches, masks)
            sample_id = f"{self.slide_set.set_id}__x{x:08}_y{y:08}"
            inputs = {name: f"{sample_id}__input__{name}{suffixes[name]}" for name in self.inputs}
            targets = {
                name: f"{sample_id}__target__{name}{suffixes[name]}" for name in self.targets
            }
            mask_files = {
                name: f"{sample_id}__foreground_mask__{name}{suffixes[name]}"
                for name in self.targets
            }
            row = {
                "sample_id": sample_id,
                "x": x,
                "y": y,
                "inputs": inputs,
                "targets": targets,
                # Only masks actually written are referenced by the manifest.
                "foreground_masks": mask_files if save_masks else {},
            }
            files = {**inputs, **targets}
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
                images = {split_dirs[split] / files[name]: patches[name] for name in files}
                if save_masks:
                    images.update(
                        {split_dirs[split] / mask_files[name]: masks[name] for name in self.targets}
                    )
                self._write_sample(images)
                valid.append({**row, "split": split})
            else:
                if self.config.patching.save_discarded_patches:
                    self._write_sample(
                        {discarded_dirs[name] / files[name]: patches[name] for name in files}
                    )
                discarded.append(
                    {**row, "ratios": foreground_ratios(masks), "reasons": ";".join(reasons)}
                )
        return valid, discarded

    def close(self) -> None:
        for state in self._states():
            if state.reader is not None:
                state.reader.close()
