from __future__ import annotations

import dataclasses
import random
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

import cv2
import numpy as np

# Only the N largest connected components are considered; smaller ones are noise.
N_TOP_COMPONENTS = 10
# Components whose ROI std dev is below this are uniform (background) and are masked out.
MIN_STD_DEV = 10

# Each (divisor, grid) pair controls one mask pass: the image is divided into a grid of
# (grid × grid) tiles, each of size (H/divisor × W/divisor). Using multiple passes at
# different scales makes the mask robust to both fine and coarse background regions.
MASK_PARAMETER_GRID = [(2, 3), (4, 6), (6, 9), (8, 15)]

ALLOWED_EXTENSIONS = {".tif", ".tiff", ".png"}
T = TypeVar("T")

MIN_INLIERS = 4


@dataclass
class AlignmentMetadata:
    """Alignment statistics captured during image registration."""

    n_keypoints_src: int
    n_keypoints_tgt: int
    n_matches: int
    n_inliers: int
    warp_matrix: list[list[float]]


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

        component_mask = (labels == i).astype(np.uint8) * 255
        countours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(component_mask, countours, -1, 255, thickness=cv2.FILLED)

        roi = img[y : y + h, x : x + w]
        roi_mask = component_mask[y : y + h, x : x + w]

        std_dev = cv2.meanStdDev(roi, mask=roi_mask)[1][0, 0]

        if std_dev < MIN_STD_DEV:
            mask[component_mask == 255] = 255

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

    for y in range(0, img.shape[0], img.shape[0] // grid):
        for x in range(0, img.shape[1], img.shape[1] // grid):
            roi = img[y : y + sub_shape[0], x : x + sub_shape[1]]
            roi_mask = calculate_mask(roi)

            roi_mask = pad_image(roi_mask, x, y, img.shape[1], img.shape[0])

            mask = cv2.bitwise_and(mask, roi_mask)
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


def align_images(
    img1: np.ndarray,
    img2: np.ndarray,
    mask1: np.ndarray | None = None,
    mask2: np.ndarray | None = None,
    nfeatures: int = 10000,
    ed_distance: int = 200,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, AlignmentMetadata]:
    """
    Aligns a moving image to a reference image.

    Parameters
    ----------
    img1 : np.ndarray
        Reference image.
    img2 : np.ndarray
        Image to align to the reference.
    mask1 : np.ndarray, optional
        The mask for the first image. Default is None.
    mask2 : np.ndarray, optional
        The mask for the second image. Default is None.
    nfeatures : int, optional
        Number of features for SIFT computation. Default is 10000.
    ed_distance : int, optional
        Inclusive Euclidean distance threshold for filtering matches. Default is 200.

    Returns
    -------
    img2_aligned : np.ndarray
        The aligned second image.
    mask2_aligned : np.ndarray, optional
        The aligned mask for the second image.
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
    keypoints_1, descriptors_1 = sift.detectAndCompute(img1_clahe, mask1)
    keypoints_2, descriptors_2 = sift.detectAndCompute(img2_clahe, mask2)

    if len(keypoints_1) < 4 or len(keypoints_2) < 4:
        raise ValueError("Not enough features for alignment")

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descriptors_1, descriptors_2)

    filtered_matches = []
    for match in matches:
        distance = np.linalg.norm(
            np.array(keypoints_1[match.queryIdx].pt) - np.array(keypoints_2[match.trainIdx].pt)
        )
        if distance <= ed_distance:
            filtered_matches.append(match)

    if len(filtered_matches) < 4:
        raise ValueError("Not enough matches for alignment")

    points_1 = np.asarray(
        [keypoints_1[match.queryIdx].pt for match in filtered_matches],
        dtype=np.float32,
    ).reshape(-1, 1, 2)
    points_2 = np.asarray(
        [keypoints_2[match.trainIdx].pt for match in filtered_matches],
        dtype=np.float32,
    ).reshape(-1, 1, 2)

    warp_matrix, inlier_mask = cv2.estimateAffinePartial2D(points_2, points_1)

    if warp_matrix is None:
        raise ValueError("Affine estimation failed: cv2.estimateAffinePartial2D returned None")

    n_inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
    if n_inliers < MIN_INLIERS:
        raise ValueError(
            f"Alignment rejected: only {n_inliers} inliers found (minimum {MIN_INLIERS} required)"
        )

    metadata = AlignmentMetadata(
        n_keypoints_src=len(keypoints_1),
        n_keypoints_tgt=len(keypoints_2),
        n_matches=len(filtered_matches),
        n_inliers=n_inliers,
        warp_matrix=warp_matrix.tolist(),
    )

    img2_aligned = cv2.warpAffine(img2, warp_matrix, (img1.shape[1], img1.shape[0]))
    mask2_aligned = None
    if mask2 is not None:
        mask2_aligned = cv2.warpAffine(mask2, warp_matrix, (img1.shape[1], img1.shape[0]))

    return img2_aligned, mask2_aligned, warp_matrix, metadata


def align_from_scaled(
    img1: np.ndarray,
    img2: np.ndarray,
    scale: float = 0.5,
    mask1: np.ndarray | None = None,
    mask2: np.ndarray | None = None,
    nfeatures: int = 10000,
    ed_distance: int = 200,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, AlignmentMetadata]:
    """
    Aligns two images by first scaling them, estimating the transformation on the scaled images,
    and then applying the transformation to the original images.

    Parameters
    ----------
    img1 : np.ndarray
        Source image used as reference.
    img2 : np.ndarray
        Target image to be aligned to the source image.
    scale : float, optional
        Scaling factor to resize images before alignment (default is 0.5).
    mask1 : Optional[np.ndarray], optional
        Optional mask for the first image.
    mask2 : Optional[np.ndarray], optional
        Optional mask for the second image.
    nfeatures : int, optional
        Number of features to use for alignment (default is 10000).
    ed_distance : int, optional
        Euclidean distance threshold for feature matching (default is 200).

    Returns
    -------
    img2_aligned : np.ndarray
        The second image aligned to the first.
    mask2_aligned : Optional[np.ndarray]
        The aligned mask for the second image, if provided.
    warp_matrix : np.ndarray
        The affine transformation matrix used for alignment.
    metadata : AlignmentMetadata
        Keypoint, match, inlier counts and the full-resolution warp matrix.
    """
    img1_scaled = cv2.resize(img1, None, fx=scale, fy=scale)
    img2_scaled = cv2.resize(img2, None, fx=scale, fy=scale)

    if mask1 is None or mask2 is None:
        raise ValueError("Error: the scaled mask is None. Cannot align images.")
    else:
        mask1_scaled = cv2.resize(mask1, None, fx=scale, fy=scale)
        mask2_scaled = cv2.resize(mask2, None, fx=scale, fy=scale)

    _, _, warp_matrix, metadata = align_images(
        img1_scaled,
        img2_scaled,
        mask1_scaled if mask1 is not None else None,
        mask2_scaled if mask2 is not None else None,
        nfeatures,
        ed_distance,
    )

    warp_matrix[0, 2] /= scale
    warp_matrix[1, 2] /= scale
    metadata = dataclasses.replace(metadata, warp_matrix=warp_matrix.tolist())

    img2_aligned = cv2.warpAffine(img2, warp_matrix, (img1.shape[1], img1.shape[0]))
    mask2_aligned = None
    if mask2 is not None:
        mask2_aligned = cv2.warpAffine(mask2, warp_matrix, (img1.shape[1], img1.shape[0]))

    return img2_aligned, mask2_aligned, warp_matrix, metadata


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


def divide_image_with_grid(
    img: np.ndarray,
    img_size: tuple[int, int],
    grid_movement: tuple[int, int],
    mask: np.ndarray | None = None,
    max_mask_percentage=0.4,
) -> tuple[list[np.ndarray], list[np.ndarray] | None, list[tuple[int, int]]]:
    """
    Divides the input image into a grid of sub-images of size `img_size`.

    Parameters
    ----------
    img : np.ndarray
        Input image.
    img_size : tuple[int, int]
        Size of the region of interest (width, height).
    grid_movement : tuple[int, int]
        Step size for moving the grid (x, y).
    mask : np.ndarray, optional
        Image mask for filtering. Default is None.
    max_mask_percentage : float, optional
        Maximum allowed percentage of masked (non-zero) pixels for a region to be
        included. Default is 0.4.

    Returns
    -------
    images : list[np.ndarray]
        List of extracted sub-images.
    masks : list[np.ndarray] or None
        List of extracted mask regions, or None if no mask is provided.
    positions : list[tuple[int, int]]
        List of positions (x, y) of the top-left corner of each extracted sub-image.
    """
    images = []
    masks = []
    positions = []

    for x in range(0, img.shape[1], grid_movement[0]):
        for y in range(0, img.shape[0], grid_movement[1]):
            roi_img = extract_image(img, x, y, img_size[0], img_size[1])

            if roi_img.shape[0] < img_size[1] or roi_img.shape[1] < img_size[0]:
                continue

            # If the mask is too black, skip the image as it is mostly background
            roi_mask = None
            if mask is not None:
                roi_mask = extract_image(mask, x, y, img_size[0], img_size[1])
                if cv2.countNonZero(roi_mask) < max_mask_percentage * roi_mask.size:
                    continue
            images.append(roi_img)
            if mask is not None:
                masks.append(roi_mask)
            positions.append((x, y))

    # If the mask is None, do not return it
    if mask is None:
        masks = None
    return images, masks, positions


def divide_image_with_positions(
    img: np.ndarray, img_size: tuple[int, int], positions: list[tuple[int, int]]
) -> list[np.ndarray]:
    """
    Splits the image into a grid of images of size img_size using the specified positions.

    Parameters
    ----------
    img : np.ndarray
        Input image.
    img_size : tuple[int, int]
        Size of the region of interest.
    positions : list[tuple[int, int]]
        List of positions for the split images.

    Returns
    -------
    images : list[np.ndarray]
        List of the split images.
    """
    images = []
    for x, y in positions:
        roi_img = extract_image(img, x, y, img_size[0], img_size[1])

        if roi_img.shape[0] < img_size[1] or roi_img.shape[1] < img_size[0]:
            continue
        images.append(roi_img)

    return images


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

    output: list[list[T]] = []
    start = 0
    for ratio in ratios:
        end = start + int(len(shuffled) * ratio)
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


def compute_white_ratio(img: np.ndarray, white_threshold: int = 240) -> float:
    """
    Computes the ratio of near-white pixels in the input image.

    Parameters
    ----------
    img : np.ndarray
        Input image.
    white_threshold : int, optional
        Pixels with grayscale intensity greater than or equal to this threshold
        are considered white background. Default is 240.

    Returns
    -------
    white_ratio : float
        Ratio of near-white pixels in the image.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray >= white_threshold))


def compute_largest_white_component_ratio(
    img: np.ndarray,
    white_threshold: int = 245,
) -> float:
    """
    Computes the area ratio of the largest connected near-white component.

    Parameters
    ----------
    img : np.ndarray
        Input image.
    white_threshold : int, optional
        Pixels with grayscale intensity greater than or equal to this threshold
        are considered white. Default is 245.

    Returns
    -------
    largest_component_ratio : float
        Ratio between the largest white connected component area and the total
        patch area.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    white_mask = (gray >= white_threshold).astype(np.uint8) * 255

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
        white_mask,
        connectivity=8,
    )

    if num_labels <= 1:
        return 0.0

    largest_area = int(np.max(stats[1:, cv2.CC_STAT_AREA]))
    return float(largest_area / white_mask.size)


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

    source_white_ratio = compute_white_ratio(source_img, white_threshold)
    target_white_ratio = compute_white_ratio(target_img, white_threshold)

    source_largest_white_component_ratio = compute_largest_white_component_ratio(
        source_img,
        white_threshold,
    )
    target_largest_white_component_ratio = compute_largest_white_component_ratio(
        target_img,
        white_threshold,
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
