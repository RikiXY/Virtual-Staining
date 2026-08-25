from __future__ import annotations

import dataclasses
import hashlib
import random
import shutil
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

import cv2
import numpy as np

from virtual_staining.config.data import (
    ALLOWED_MASK_STRATEGIES,
    MASK_STRATEGY_CONNECTED_COMPONENTS,
    MASK_STRATEGY_HSV,
)

# Only the N largest connected components are considered; smaller ones are noise.
N_TOP_COMPONENTS = 10
# Components whose ROI std dev is below this are uniform (background) and are masked out.
MIN_STD_DEV = 15

# Each (divisor, grid) pair controls one mask pass: the image is divided into a grid of
# (grid x grid) tiles, each of size (H/divisor x W/divisor). Using multiple passes at
# different scales makes the mask robust to both fine and coarse background regions.
MASK_PARAMETER_GRID = [(2, 3), (4, 6), (6, 9), (8, 15)]


ALLOWED_EXTENSIONS = {".tif", ".tiff", ".png"}
T = TypeVar("T")

MIN_INLIERS = 12
MIN_INLIER_RATIO = 0.10
SPLIT_NAMES: tuple[str, str, str] = ("train", "val", "test")
LOWE_RATIO_THRESHOLD = 0.75
RANSAC_REPROJECTION_THRESHOLD = 5.0
RANSAC_MAX_ITERS = 2000
RANSAC_CONFIDENCE = 0.99
RANSAC_REFINE_ITERS = 10


@dataclass
class AlignmentMetadata:
    """Alignment statistics captured during image registration."""

    n_keypoints_src: int
    n_keypoints_tgt: int
    n_matches: int
    n_inliers: int
    inlier_ratio: float
    scale_x: float
    scale_y: float
    rotation_deg: float
    translation_x: float
    translation_y: float
    warp_matrix: list[list[float]]
    mask_iou: float | None = None


def pad_image(img: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    """
    Expands the image with a white border.

    Parameters
    ----------
    img : np.ndarray
        Input image.
    x : int
        X coordinate of the border.
    y : int
        Y coordinate of the border.
    w : int
        Width of the output image.
    h : int
        Height of the output image.

    Returns
    -------
    padded_image : np.ndarray
        Expanded (padded) image.
    """
    top = y
    bottom = h - y - img.shape[0]
    left = x
    right = w - x - img.shape[1]
    padded_image = cv2.copyMakeBorder(
        img, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=255
    )
    return padded_image


def assign_split_by_hash(
    *,
    seed: int,
    sample_id: str,
    ratios: Sequence[float],
) -> str:
    """Assign a sample to train/val/test using a stable hash of seed and sample id."""
    if len(ratios) != len(SPLIT_NAMES):
        raise ValueError(f"Expected {len(SPLIT_NAMES)} split ratios, got {len(ratios)}")
    if any(ratio < 0 for ratio in ratios):
        raise ValueError("Split ratios must be non-negative")
    ratio_sum = sum(ratios)
    if not np.isclose(ratio_sum, 1.0):
        raise ValueError(f"Split ratios must sum to 1.0, got {ratio_sum}")

    digest = hashlib.sha256(f"{seed}:{sample_id}".encode()).digest()
    value = int.from_bytes(digest[:8], byteorder="big") / 2**64

    cumulative = 0.0
    for split_name, ratio in zip(SPLIT_NAMES, ratios, strict=True):
        cumulative += ratio
        if value < cumulative:
            return split_name
    return SPLIT_NAMES[-1]


def calculate_mask(img: np.ndarray) -> np.ndarray:
    """
    Finds the mask for the connected components in the image.

    Parameters
    ----------
    img : np.ndarray
        Input image.

    Returns
    -------
    mask : np.ndarray
        Image mask.
    """
    _, binary = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 230, 255, cv2.THRESH_BINARY)

    _, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    sorted_indices = np.argsort(stats[1:, cv2.CC_STAT_AREA])[::-1] + 1

    mask = np.zeros_like(binary).astype(np.uint8)

    for i in sorted_indices[:N_TOP_COMPONENTS]:
        x, y, w, h, area = stats[i]

        if w < 100 and h < 100:
            continue

        label_roi = labels[y : y + h, x : x + w]
        component_mask = (label_roi == i).astype(np.uint8) * 255
        countours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(component_mask, countours, -1, 255, thickness=cv2.FILLED)
        roi = img[y : y + h, x : x + w]
        roi_mask = component_mask

        std_dev = float(np.max(cv2.meanStdDev(roi, mask=roi_mask)[1]))

        if std_dev < MIN_STD_DEV:
            mask_roi = mask[y : y + h, x : x + w]
            mask_roi[component_mask == 255] = 255

    # Inverts the mask to get the foreground.
    # The mask is 255 for the foreground and 0 for the background.
    mask = cv2.bitwise_not(mask)
    return mask


def calculate_mask_with_grid(img: np.ndarray, sub_shape: tuple[int, int], grid: int) -> np.ndarray:
    """
    Finds the mask for the connected components of the image using a grid.

    Parameters
    ----------
    img : np.ndarray
        Input image.
    sub_shape : tuple[int, int]
        Size of the region of interest.
    grid : int
        Number of regions per side of the grid.

    Returns
    -------
    mask : np.ndarray
        Mask of the image.
    """
    mask = np.ones((img.shape[0], img.shape[1]), dtype=np.uint8) * 255
    step_y = max(1, img.shape[0] // grid)
    step_x = max(1, img.shape[1] // grid)

    for y in range(0, img.shape[0], step_y):
        for x in range(0, img.shape[1], step_x):
            y2 = min(y + sub_shape[0], img.shape[0])
            x2 = min(x + sub_shape[1], img.shape[1])
            roi = img[y:y2, x:x2]
            if roi.size == 0:
                continue

            roi_mask = calculate_mask(roi)
            mask[y:y2, x:x2] = cv2.bitwise_and(mask[y:y2, x:x2], roi_mask)
    return mask


def calculate_mask_with_multiple_parameters(
    img: np.ndarray, parameters: list[tuple[int, int]]
) -> np.ndarray:
    """
    Calculates the mask for the input image using multiple parameter pairs.

    Parameters
    ----------
    img : np.ndarray
        Input image.
    parameters : list[tuple[int, int]]
        List of (divisor, grid) pairs used to calculate the masks.

    Returns
    -------
    mask : np.ndarray
        Mask of the image.
    """
    mask = np.ones((img.shape[0], img.shape[1]), dtype=np.uint8) * 255

    for divisor, grid in parameters:
        sub_shape = (img.shape[0] // divisor, img.shape[1] // divisor)

        _mask = calculate_mask_with_grid(img, sub_shape, grid)

        mask = cv2.bitwise_and(mask, _mask)

    return mask


def apply_mask_morphology(mask: np.ndarray, *, kernel_size: int = 5) -> np.ndarray:
    """Clean small mask speckles and holes while preserving a binary foreground mask."""
    if kernel_size <= 1:
        return mask
    kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)


def calculate_hsv_tissue_mask(
    img: np.ndarray,
    *,
    min_saturation: int = 20,
    max_value: int = 245,
    morphology_kernel_size: int = 5,
) -> np.ndarray:
    """
    Build a foreground mask from HSV saturation and value thresholds.

    Tissue is considered foreground when it is either visibly saturated or dark
    enough to be distinct from bright background.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    foreground = (saturation >= min_saturation) | (value <= max_value)
    mask = foreground.astype(np.uint8) * 255
    return apply_mask_morphology(mask, kernel_size=morphology_kernel_size)


def calculate_mask_by_strategy(
    img: np.ndarray,
    *,
    strategy: str = MASK_STRATEGY_CONNECTED_COMPONENTS,
    parameters: list[tuple[int, int]] | None = None,
) -> np.ndarray:
    """Dispatch tissue-mask generation through a named strategy."""
    if strategy == MASK_STRATEGY_CONNECTED_COMPONENTS:
        return calculate_mask_with_multiple_parameters(
            img,
            MASK_PARAMETER_GRID if parameters is None else parameters,
        )
    if strategy == MASK_STRATEGY_HSV:
        return calculate_hsv_tissue_mask(img)
    raise ValueError(
        f"Unknown mask strategy {strategy!r}; expected one of {ALLOWED_MASK_STRATEGIES}"
    )


def _affine_diagnostics(warp_matrix: np.ndarray) -> dict[str, float]:
    a, b, tx = warp_matrix[0]
    c, d, ty = warp_matrix[1]
    return {
        "scale_x": float(np.sqrt(a * a + c * c)),
        "scale_y": float(np.sqrt(b * b + d * d)),
        "rotation_deg": float(np.degrees(np.arctan2(c, a))),
        "translation_x": float(tx),
        "translation_y": float(ty),
    }


def _ratio_test_matches(
    knn_matches: Sequence[Sequence[cv2.DMatch]],
    *,
    ratio_threshold: float = LOWE_RATIO_THRESHOLD,
) -> list[cv2.DMatch]:
    """Keep descriptor matches whose nearest neighbor is clearly better than the second."""
    good_matches = []
    for candidates in knn_matches:
        if len(candidates) < 2:
            continue
        best, second_best = candidates[0], candidates[1]
        if best.distance < ratio_threshold * second_best.distance:
            good_matches.append(best)
    return good_matches


def _aligned_mask_iou(
    mask_1: np.ndarray | None,
    mask_2: np.ndarray | None,
    warp_matrix: np.ndarray,
    output_size: tuple[int, int],
) -> float | None:
    """Compute foreground IoU after warping mask_2 into mask_1 coordinates."""
    if mask_1 is None or mask_2 is None:
        return None

    aligned_mask_2 = cv2.warpAffine(
        mask_2,
        warp_matrix,
        output_size,
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    foreground_1 = mask_1 > 0
    foreground_2 = aligned_mask_2 > 0
    union = np.logical_or(foreground_1, foreground_2)
    union_count = int(np.count_nonzero(union))
    if union_count == 0:
        return None
    intersection_count = int(np.count_nonzero(np.logical_and(foreground_1, foreground_2)))
    return intersection_count / union_count


def estimate_affine_transform(
    img1: np.ndarray,
    img2: np.ndarray,
    mask_1: np.ndarray | None = None,
    mask_2: np.ndarray | None = None,
    nfeatures: int = 10000,
    ratio_threshold: float = LOWE_RATIO_THRESHOLD,
) -> tuple[np.ndarray, AlignmentMetadata]:
    """
    Estimate an affine transform that maps ``img2`` into ``img1`` coordinates.

    Parameters
    ----------
    img1 : np.ndarray
        Reference image.
    img2 : np.ndarray
        Image to align to the reference.
    mask_1 : np.ndarray, optional
        The mask for the first image. Default is None.
    mask_2 : np.ndarray, optional
        The mask for the second image. Default is None.
    nfeatures : int, optional
        Number of features for SIFT computation. Default is 10000.
    ratio_threshold : float, optional
        Lowe-style nearest/second-nearest descriptor distance ratio threshold.
        Default is 0.75.

    Returns
    -------
    warp_matrix : np.ndarray
        The transformation matrix.
    metadata : AlignmentMetadata
        Keypoint, match, inlier counts and the warp matrix.
    """
    clahe = cv2.createCLAHE(clipLimit=18.0, tileGridSize=(8, 8))
    img1_clahe = img1
    img2_clahe = img2

    if len(img1_clahe.shape) == 3:
        img1_clahe = cv2.cvtColor(img1_clahe, cv2.COLOR_BGR2GRAY)
    if len(img2_clahe.shape) == 3:
        img2_clahe = cv2.cvtColor(img2_clahe, cv2.COLOR_BGR2GRAY)

    img1_clahe = clahe.apply(img1_clahe)
    img2_clahe = clahe.apply(img2_clahe)

    sift = cv2.SIFT_create(nfeatures=nfeatures)  # type: ignore[attr-defined]
    keypoints_1, descriptors_1 = sift.detectAndCompute(img1_clahe, mask_1)
    keypoints_2, descriptors_2 = sift.detectAndCompute(img2_clahe, mask_2)

    n_keypoints_1 = len(keypoints_1)
    n_keypoints_2 = len(keypoints_2)
    if n_keypoints_1 < 4 or n_keypoints_2 < 4 or descriptors_1 is None or descriptors_2 is None:
        raise ValueError(
            "Not enough features for alignment: "
            f"source={n_keypoints_1}, target={n_keypoints_2}, minimum=4"
        )

    bf = cv2.BFMatcher(cv2.NORM_L2)
    knn_matches = bf.knnMatch(descriptors_1, descriptors_2, k=2)
    filtered_matches = _ratio_test_matches(knn_matches, ratio_threshold=ratio_threshold)

    if len(filtered_matches) < 4:
        raise ValueError(
            "Not enough good descriptor matches for alignment after ratio test: "
            f"good={len(filtered_matches)}, minimum=4, ratio_threshold={ratio_threshold}"
        )

    points_1 = np.asarray(
        [keypoints_1[match.queryIdx].pt for match in filtered_matches],
        dtype=np.float32,
    ).reshape(-1, 1, 2)
    points_2 = np.asarray(
        [keypoints_2[match.trainIdx].pt for match in filtered_matches],
        dtype=np.float32,
    ).reshape(-1, 1, 2)

    warp_matrix, inlier_mask = cv2.estimateAffinePartial2D(
        points_2,
        points_1,
        method=cv2.RANSAC,
        ransacReprojThreshold=RANSAC_REPROJECTION_THRESHOLD,
        maxIters=RANSAC_MAX_ITERS,
        confidence=RANSAC_CONFIDENCE,
        refineIters=RANSAC_REFINE_ITERS,
    )

    if warp_matrix is None:
        raise ValueError(
            "Affine estimation failed after ratio-test matching and RANSAC: "
            "cv2.estimateAffinePartial2D returned None"
        )

    n_inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
    inlier_ratio = n_inliers / len(filtered_matches)
    if n_inliers < MIN_INLIERS:
        raise ValueError(
            f"Alignment rejected: only {n_inliers} inliers found (minimum {MIN_INLIERS} required)"
        )
    if inlier_ratio < MIN_INLIER_RATIO:
        raise ValueError(
            "Alignment rejected: inlier ratio "
            f"{inlier_ratio:.3f} is below minimum {MIN_INLIER_RATIO:.3f} "
            f"({n_inliers}/{len(filtered_matches)} inliers)"
        )

    diagnostics = _affine_diagnostics(warp_matrix)
    mask_iou = _aligned_mask_iou(
        mask_1,
        mask_2,
        warp_matrix,
        (img1.shape[1], img1.shape[0]),
    )
    metadata = AlignmentMetadata(
        n_keypoints_src=n_keypoints_1,
        n_keypoints_tgt=n_keypoints_2,
        n_matches=len(filtered_matches),
        n_inliers=n_inliers,
        inlier_ratio=inlier_ratio,
        scale_x=diagnostics["scale_x"],
        scale_y=diagnostics["scale_y"],
        rotation_deg=diagnostics["rotation_deg"],
        translation_x=diagnostics["translation_x"],
        translation_y=diagnostics["translation_y"],
        warp_matrix=warp_matrix.tolist(),
        mask_iou=mask_iou,
    )

    return warp_matrix, metadata


def warp_aligned_image(
    img: np.ndarray,
    warp_matrix: np.ndarray,
    output_size: tuple[int, int],
    *,
    is_mask: bool,
) -> np.ndarray:
    """Warp an image or mask with interpolation appropriate to its data type."""
    return cv2.warpAffine(
        img,
        warp_matrix,
        output_size,
        flags=cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0 if is_mask else (255, 255, 255),
    )


def warp_aligned_patch(
    img: np.ndarray,
    warp_matrix: np.ndarray,
    *,
    x: int,
    y: int,
    output_size: tuple[int, int],
    is_mask: bool,
) -> np.ndarray:
    """
    Warp one destination patch from an image aligned by ``warp_matrix``.

    ``warp_matrix`` maps the input image into the full reference image frame.
    The patch origin ``(x, y)`` is expressed in that full destination frame.
    """
    patch_matrix = np.asarray(warp_matrix, dtype=np.float64).copy()
    patch_matrix[0, 2] -= x
    patch_matrix[1, 2] -= y
    return warp_aligned_image(
        img,
        patch_matrix,
        output_size,
        is_mask=is_mask,
    )


def mask_window_for_patch(
    mask: np.ndarray,
    image_shape: tuple[int, int] | tuple[int, int, int],
    *,
    x: int,
    y: int,
    width: int,
    height: int,
) -> np.ndarray:
    """Return the mask-space window corresponding to a full-resolution image patch."""
    if width <= 0 or height <= 0:
        raise ValueError("Patch width and height must be positive")

    image_h, image_w = image_shape[:2]
    if image_h <= 0 or image_w <= 0:
        raise ValueError("Image shape must have positive height and width")

    mask_h, mask_w = mask.shape[:2]
    if mask_h <= 0 or mask_w <= 0:
        raise ValueError("Mask shape must have positive height and width")

    scale_x = mask_w / image_w
    scale_y = mask_h / image_h
    x0 = max(0, min(mask_w - 1, int(np.floor(x * scale_x))))
    y0 = max(0, min(mask_h - 1, int(np.floor(y * scale_y))))
    x1 = max(x0 + 1, int(np.ceil((x + width) * scale_x)))
    y1 = max(y0 + 1, int(np.ceil((y + height) * scale_y)))
    x1 = min(mask_w, x1)
    y1 = min(mask_h, y1)
    return mask[y0:y1, x0:x1]


def foreground_ratio_for_patch(
    mask: np.ndarray,
    image_shape: tuple[int, int] | tuple[int, int, int],
    *,
    x: int,
    y: int,
    width: int,
    height: int,
) -> float:
    """Compute approximate foreground coverage for an image patch from mask-space pixels."""
    window = mask_window_for_patch(mask, image_shape, x=x, y=y, width=width, height=height)
    return cv2.countNonZero(window) / window.size


def warp_aligned_mask_patch_from_mask_space(
    mask: np.ndarray,
    warp_matrix: np.ndarray,
    image_shape: tuple[int, int] | tuple[int, int, int],
    *,
    x: int,
    y: int,
    output_size: tuple[int, int],
) -> np.ndarray:
    """Warp a destination patch from a mask that may be stored below image resolution."""
    image_h, image_w = image_shape[:2]
    mask_h, mask_w = mask.shape[:2]
    scale_x = mask_w / image_w
    scale_y = mask_h / image_h
    if scale_x <= 0 or scale_y <= 0:
        raise ValueError("Mask and image shapes must be positive")

    mask_space_matrix = np.asarray(warp_matrix, dtype=np.float64).copy()
    mask_space_matrix[:, 0] /= scale_x
    mask_space_matrix[:, 1] /= scale_y
    return warp_aligned_patch(
        mask,
        mask_space_matrix,
        x=x,
        y=y,
        output_size=output_size,
        is_mask=True,
    )


def align_images(
    img1: np.ndarray,
    img2: np.ndarray,
    mask_1: np.ndarray | None = None,
    mask_2: np.ndarray | None = None,
    nfeatures: int = 10000,
    ratio_threshold: float = LOWE_RATIO_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, AlignmentMetadata]:
    """Align a moving image to a reference image and return warped outputs."""
    warp_matrix, metadata = estimate_affine_transform(
        img1,
        img2,
        mask_1=mask_1,
        mask_2=mask_2,
        nfeatures=nfeatures,
        ratio_threshold=ratio_threshold,
    )

    img2_aligned = warp_aligned_image(
        img2,
        warp_matrix,
        (img1.shape[1], img1.shape[0]),
        is_mask=False,
    )
    mask_2_aligned = None
    if mask_2 is not None:
        mask_2_aligned = warp_aligned_image(
            mask_2,
            warp_matrix,
            (img1.shape[1], img1.shape[0]),
            is_mask=True,
        )

    return img2_aligned, mask_2_aligned, warp_matrix, metadata


def estimate_affine_from_scaled(
    img1: np.ndarray,
    img2: np.ndarray,
    scale: float = 0.5,
    mask_1: np.ndarray | None = None,
    mask_2: np.ndarray | None = None,
    nfeatures: int = 10000,
    ratio_threshold: float = LOWE_RATIO_THRESHOLD,
) -> tuple[np.ndarray, AlignmentMetadata]:
    """
    Estimate a full-resolution affine transform from downscaled image inputs.

    Parameters
    ----------
    img1 : np.ndarray
        Source image used as reference.
    img2 : np.ndarray
        Target image to be aligned to the source image.
    scale : float, optional
        Scaling factor to resize images before alignment (default is 0.5).
    mask_1 : Optional[np.ndarray], optional
        Optional mask for the first image.
    mask_2 : Optional[np.ndarray], optional
        Optional mask for the second image.
    nfeatures : int, optional
        Number of features to use for alignment (default is 10000).
    ratio_threshold : float, optional
        Lowe-style nearest/second-nearest descriptor distance ratio threshold.

    Returns
    -------
    warp_matrix : np.ndarray
        The affine transformation matrix used for alignment.
    metadata : AlignmentMetadata
        Keypoint, match, inlier counts and the full-resolution warp matrix.
    """
    img1_scaled = cv2.resize(img1, None, fx=scale, fy=scale)
    img2_scaled = cv2.resize(img2, None, fx=scale, fy=scale)

    if mask_1 is None or mask_2 is None:
        raise ValueError("Error: the scaled mask is None. Cannot align images.")
    else:
        mask_1_scaled = cv2.resize(
            mask_1, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST
        )
        mask_2_scaled = cv2.resize(
            mask_2, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST
        )

    warp_matrix, metadata = estimate_affine_transform(
        img1_scaled,
        img2_scaled,
        mask_1=mask_1_scaled if mask_1 is not None else None,
        mask_2=mask_2_scaled if mask_2 is not None else None,
        nfeatures=nfeatures,
        ratio_threshold=ratio_threshold,
    )

    warp_matrix[0, 2] /= scale
    warp_matrix[1, 2] /= scale
    diagnostics = _affine_diagnostics(warp_matrix)
    metadata = dataclasses.replace(
        metadata,
        scale_x=diagnostics["scale_x"],
        scale_y=diagnostics["scale_y"],
        rotation_deg=diagnostics["rotation_deg"],
        translation_x=diagnostics["translation_x"],
        translation_y=diagnostics["translation_y"],
        warp_matrix=warp_matrix.tolist(),
    )

    return warp_matrix, metadata


def align_from_scaled(
    img1: np.ndarray,
    img2: np.ndarray,
    scale: float = 0.5,
    mask_1: np.ndarray | None = None,
    mask_2: np.ndarray | None = None,
    nfeatures: int = 10000,
    ratio_threshold: float = LOWE_RATIO_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, AlignmentMetadata]:
    """
    Aligns two images by first scaling them, estimating the transformation on the scaled images,
    and then applying the transformation to the original images.
    """
    warp_matrix, metadata = estimate_affine_from_scaled(
        img1,
        img2,
        scale=scale,
        mask_1=mask_1,
        mask_2=mask_2,
        nfeatures=nfeatures,
        ratio_threshold=ratio_threshold,
    )

    img2_aligned = warp_aligned_image(
        img2,
        warp_matrix,
        (img1.shape[1], img1.shape[0]),
        is_mask=False,
    )
    mask_2_aligned = None
    if mask_2 is not None:
        mask_2_aligned = warp_aligned_image(
            mask_2,
            warp_matrix,
            (img1.shape[1], img1.shape[0]),
            is_mask=True,
        )

    return img2_aligned, mask_2_aligned, warp_matrix, metadata


def extract_image(img: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    """
    Extracts a region from the image.

    Parameters
    ----------
    img : np.ndarray
        Input image
    x : int
        x-coordinate of the top-left corner
    y : int
        y-coordinate of the top-left corner
    w : int
        Width of the region
    h : int
        Height of the region

    Returns
    -------
    roi : np.ndarray
        Region of the image
    """
    return img[y : y + h, x : x + w]


def iter_image_with_grid(
    img: np.ndarray,
    img_size: tuple[int, int],
    grid_movement: tuple[int, int],
    mask: np.ndarray | None = None,
    max_mask_percentage: float = 0.4,
) -> Iterator[tuple[tuple[int, int], np.ndarray, np.ndarray | None]]:
    """
    Yield valid grid patches one at a time instead of materializing them all.

    Each yielded item contains ``((x, y), image_patch, mask_patch)`` where
    ``mask_patch`` is ``None`` when no mask was provided.
    """
    for x in range(0, img.shape[1], grid_movement[0]):
        for y in range(0, img.shape[0], grid_movement[1]):
            roi_img = extract_image(img, x, y, img_size[0], img_size[1])

            if roi_img.shape[0] < img_size[1] or roi_img.shape[1] < img_size[0]:
                continue

            roi_mask = None
            if mask is not None:
                roi_mask = extract_image(mask, x, y, img_size[0], img_size[1])
                if cv2.countNonZero(roi_mask) < max_mask_percentage * roi_mask.size:
                    continue

            yield (x, y), roi_img, roi_mask


def split_items(items: list[T], ratios: Sequence[float]) -> list[list[T]]:
    """
    Splits the input list into N sublists according to the specified ratios.

    Parameters
    ----------
    items : list
        List to split.
    ratios : list[int]
        Split ratios (e.g. [0.7, 0.15, 0.15]).

    Returns
    -------
    output : list[list]
        List of generated sublists.
    """
    if len(ratios) < 2:
        raise ValueError("At least 2 ratios must be specified")
    if any(ratio < 0 for ratio in ratios):
        raise ValueError("All ratios must be >= 0")
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("The sum of ratios must equal 1")

    shuffled = items.copy()
    random.shuffle(shuffled)

    n = len(shuffled)
    cumulative = 0.0
    output: list[list[T]] = []
    start = 0
    for i, ratio in enumerate(ratios):
        cumulative += ratio
        end = n if i == len(ratios) - 1 else round(cumulative * n)
        output.append(shuffled[start:end])
        start = end
    return output


def validate_image_filename(filename: str, role: str) -> Path:
    file_path = Path(filename)
    suffix = file_path.suffix.lower()

    if not file_path.name:
        raise ValueError(f"{role} filename is empty.")
    if suffix not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"{role} must use one of these extensions: {', '.join(sorted(ALLOWED_EXTENSIONS))}. "
            f"Received: {filename}"
        )
    return file_path


def compute_white_stats(
    img: np.ndarray,
    white_threshold: int = 245,
    *,
    largest_component_threshold: float | None = None,
) -> tuple[float, float]:
    """
    Compute the white-pixel ratio and largest white-component ratio in one pass.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    white_mask = gray >= white_threshold
    white_ratio = float(np.mean(white_mask))

    if largest_component_threshold is not None and white_ratio <= largest_component_threshold:
        return white_ratio, 0.0

    white_mask_u8 = white_mask.astype(np.uint8) * 255
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
        white_mask_u8,
        connectivity=8,
    )

    if num_labels <= 1:
        return white_ratio, 0.0

    largest_area = int(np.max(stats[1:, cv2.CC_STAT_AREA]))
    return white_ratio, float(largest_area / white_mask_u8.size)


def ensure_clean_directory(directory: str | Path) -> None:
    """
    Removes an output directory if it already exists and recreates it empty.

    Parameters
    ----------
    directory : str | Path
        Directory to clean and recreate.
    """
    directory = Path(directory)
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)


def is_valid_patch_pair(
    source_img: np.ndarray,
    target_img: np.ndarray,
    source_mask: np.ndarray,
    target_mask: np.ndarray,
    min_foreground_ratio: float,
    max_white_ratio: float,
    white_threshold: int,
    max_largest_white_component_ratio: float,
) -> tuple[bool, dict[str, float | list[str]]]:
    """
    Validates a source/target patch pair using both mask coverage and white ratio.

    Returns
    -------
    is_valid : bool
        True if the patch pair is valid, False otherwise.
    debug_info : dict[str, float | list[str]]
        Dictionary containing computed ratios and discard reasons.
    """
    source_foreground_ratio = cv2.countNonZero(source_mask) / source_mask.size
    target_foreground_ratio = cv2.countNonZero(target_mask) / target_mask.size

    source_white_ratio, source_largest_white_component_ratio = compute_white_stats(
        source_img,
        white_threshold,
        largest_component_threshold=max_largest_white_component_ratio,
    )
    target_white_ratio, target_largest_white_component_ratio = compute_white_stats(
        target_img,
        white_threshold,
        largest_component_threshold=max_largest_white_component_ratio,
    )

    reasons: list[str] = []

    if source_foreground_ratio < min_foreground_ratio:
        reasons.append("low_source_foreground")
    if target_foreground_ratio < min_foreground_ratio:
        reasons.append("low_target_foreground")
    if source_white_ratio > max_white_ratio:
        reasons.append("high_source_white_ratio")
    if target_white_ratio > max_white_ratio:
        reasons.append("high_target_white_ratio")
    if source_largest_white_component_ratio > max_largest_white_component_ratio:
        reasons.append("high_source_largest_white_component_ratio")
    if target_largest_white_component_ratio > max_largest_white_component_ratio:
        reasons.append("high_target_largest_white_component_ratio")

    debug_info = {
        "source_foreground_ratio": source_foreground_ratio,
        "target_foreground_ratio": target_foreground_ratio,
        "source_white_ratio": source_white_ratio,
        "target_white_ratio": target_white_ratio,
        "source_largest_white_component_ratio": source_largest_white_component_ratio,
        "target_largest_white_component_ratio": target_largest_white_component_ratio,
        "reasons": reasons,
    }

    return len(reasons) == 0, debug_info
