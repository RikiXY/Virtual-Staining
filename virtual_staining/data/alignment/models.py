from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np


class AlignmentError(ValueError):
    """Invalid alignment inputs, incompatible geometry, or failed registration."""


@dataclass(frozen=True)
class AlignmentImage:
    """Borrowed image data; registration never modifies these arrays.

    ``preview`` is a uint8 BGR or grayscale image covering the whole image.
    ``full_shape`` is (height, width), in full-resolution pixels. A uint8 2D
    ``mask`` covers the same field of view at full or downsampled resolution
    (aspect ratio preserved, allowing pixel rounding). ``mpp`` is (x, y).
    No reader or file ownership crosses this boundary.
    """

    preview: np.ndarray
    full_shape: tuple[int, int]
    mask: np.ndarray | None = None
    name: str = "image"
    mpp: tuple[float | None, float | None] = (None, None)


@dataclass(frozen=True)
class _RegistrationDiagnostics:
    """Counts and overlap at estimation resolution; geometry at result resolution."""

    n_keypoints_reference: int
    n_keypoints_moving: int
    n_matches: int
    n_inliers: int
    inlier_ratio: float
    scale_x: float
    scale_y: float
    rotation_deg: float
    translation_x: float
    translation_y: float
    mask_iou: float | None = None


def _validate_affine(matrix: np.ndarray) -> None:
    if matrix.shape != (2, 3) or not np.isfinite(matrix).all():
        raise AlignmentError("Invalid affine transform: expected a finite 2x3 matrix")
    if abs(np.linalg.det(matrix[:, :2])) <= np.finfo(np.float64).eps:
        raise AlignmentError("Invalid affine transform: linear part is singular")


@dataclass(frozen=True)
class AlignmentResult:
    """Moving full-resolution (x, y) -> reference full-resolution (x, y).

    The 2x3 affine matrix owns read-only storage. Metadata is a fresh JSON-ready
    snapshot, so serialization cannot mutate the transform or its diagnostics.
    """

    method: Literal["identity", "affine_sift"]
    warp_matrix: np.ndarray
    reason: str | None = None
    diagnostics: _RegistrationDiagnostics | None = None

    def __post_init__(self) -> None:
        matrix = np.asarray(self.warp_matrix, dtype=np.float64)
        _validate_affine(matrix)
        matrix = np.frombuffer(matrix.tobytes(), dtype=np.float64).reshape(2, 3)
        object.__setattr__(self, "warp_matrix", matrix)

    @property
    def metadata(self) -> dict[str, str | int | float | None | list[list[float]]]:
        metadata: dict[str, str | int | float | None | list[list[float]]] = {}
        if self.diagnostics is not None:
            metadata.update(asdict(self.diagnostics))
            # Preserve the dataset metadata schema; registration itself uses neutral names.
            metadata["n_keypoints_src"] = metadata.pop("n_keypoints_reference")
            metadata["n_keypoints_tgt"] = metadata.pop("n_keypoints_moving")
        metadata.update(method=self.method, warp_matrix=self.warp_matrix.tolist())
        if self.reason is not None:
            metadata["reason"] = self.reason
        return metadata
