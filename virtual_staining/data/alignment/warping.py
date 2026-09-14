from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np

from virtual_staining.data.alignment.models import AlignmentError, _validate_affine


def _validate_mask_geometry(
    mask: np.ndarray, image_shape: tuple[int, int], *, name: str = "mask"
) -> None:
    """Require a 2D uint8 mask of the whole image, allowing downsampling rounding."""
    image_h, image_w = image_shape
    if image_h <= 0 or image_w <= 0:
        raise AlignmentError("Image shape must have positive height and width")
    if mask.ndim != 2 or mask.dtype != np.uint8 or min(mask.shape) <= 0:
        raise AlignmentError(f"Invalid {name} geometry: expected a nonempty 2D uint8 mask")
    mask_h, mask_w = mask.shape
    if (
        mask_h > image_h
        or mask_w > image_w
        or abs(mask_h * image_w - mask_w * image_h) > max(image_h, image_w)
    ):
        raise AlignmentError(
            f"Invalid {name} geometry: {mask.shape} must cover image {image_shape} "
            "at full or downsampled resolution with the same aspect ratio"
        )


def _rescale_transform(
    matrix: np.ndarray,
    *,
    reference_scale: tuple[float, float],
    moving_scale: tuple[float, float],
) -> np.ndarray:
    """Convert a scaled moving->reference matrix back to the unscaled frames.

    Scales are (x, y) scaled-pixels / original-pixels. This computes
    S_reference^-1 @ matrix @ S_moving, including unequal image/axis scales.
    Scaling preserves the origin, following the existing registration convention.
    """
    converted = np.asarray(matrix, dtype=np.float64).copy()
    _validate_affine(converted)
    converted[:, :2] *= np.asarray(moving_scale)
    converted /= np.asarray(reference_scale)[:, None]
    return converted


def _warp_image(
    image: np.ndarray, matrix: np.ndarray, output_size: tuple[int, int], *, is_mask: bool
) -> np.ndarray:
    """Apply input->output affine coordinates; output_size is (width, height)."""
    return cv2.warpAffine(
        image,
        matrix,
        output_size,
        flags=cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0 if is_mask else (255, 255, 255),
    )


def _read_patch_region(
    read_region: Callable[[int, int, int, int], np.ndarray],
    matrix: np.ndarray,
    x: int,
    y: int,
    output_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Read the inverse-mapped bounds; return region-local -> full reference matrix."""
    width, height = output_size
    corners = cv2.transform(
        np.array(
            [[[x, y], [x + width, y], [x, y + height], [x + width, y + height]]],
            dtype=np.float64,
        ),
        cv2.invertAffineTransform(matrix),
    )[0]
    # Two source pixels around the inverse bounds preserve bilinear edge sampling.
    rx, ry = int(np.floor(corners[:, 0].min())) - 2, int(np.floor(corners[:, 1].min())) - 2
    rw = max(1, int(np.ceil(corners[:, 0].max())) + 2 - rx)
    rh = max(1, int(np.ceil(corners[:, 1].max())) + 2 - ry)
    region = read_region(rx, ry, rw, rh)
    local = matrix.copy()
    local[:, 2] += local[:, :2] @ np.array([rx, ry])
    return region, local


def warp_aligned_patch(
    image: np.ndarray | Callable[[int, int, int, int], np.ndarray],
    warp_matrix: np.ndarray,
    *,
    x: int,
    y: int,
    output_size: tuple[int, int],
    is_mask: bool = False,
) -> np.ndarray:
    """Warp input pixels into a reference patch at (x, y), size (width, height).

    ``warp_matrix`` maps the input's pixel coordinates to full reference pixels.
    A reader callback accepts full moving (x, y, width, height) and returns a
    padded region (white for BGR images, zero for masks). IO owns reading and
    resources; this function calculates bounds and applies the local transform.
    Arrays and matrices are never modified. Masks use nearest-neighbor sampling.
    """
    matrix = np.asarray(warp_matrix, dtype=np.float64)
    _validate_affine(matrix)
    if min(output_size) <= 0:
        raise AlignmentError("Patch width and height must be positive")
    if callable(image):
        image, matrix = _read_patch_region(image, matrix, x, y, output_size)
    patch_matrix = matrix.copy()
    patch_matrix[:, 2] -= (x, y)
    return _warp_image(image, patch_matrix, output_size, is_mask=is_mask)


def warp_aligned_mask_patch(
    mask: np.ndarray,
    warp_matrix: np.ndarray,
    image_shape: tuple[int, int],
    *,
    x: int,
    y: int,
    output_size: tuple[int, int],
) -> np.ndarray:
    """Warp a whole-image mask at any valid resolution with nearest neighbors.

    ``image_shape`` is moving full (height, width); ``warp_matrix`` maps moving
    full pixels to reference full pixels. Mask pixels are scaled into moving
    full coordinates before applying it. Origin/size use reference (x, y)/(w, h).
    """
    _validate_mask_geometry(mask, image_shape)
    image_h, image_w = image_shape
    mask_h, mask_w = mask.shape
    matrix = _rescale_transform(
        warp_matrix,
        reference_scale=(1.0, 1.0),
        moving_scale=(image_w / mask_w, image_h / mask_h),
    )
    return warp_aligned_patch(mask, matrix, x=x, y=y, output_size=output_size, is_mask=True)
