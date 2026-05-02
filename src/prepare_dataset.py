"""
Integrated preprocessing pipeline for paired histopathology samples.

This script loads two paired full-size images (a source image and a target
image), computes tissue masks, aligns the target image to the source reference,
extracts paired patches, and creates the `dataset_train`, `dataset_val`, and
`dataset_test` splits.
"""

# The original filename was `ollie_wan_kenobi`, born from an internal joke:
# we needed an "all-in-one" script, which gradually turned into "Ollie Wan",
# and from there "Kenobi" felt like the only possible ending.
# Ironically, none of the collaborators had even seen Star Wars.

import cv2, random, shutil
import argparse
import json
import csv
from pathlib import Path
from typing import Optional

import numpy as np

# Import the messages.json file that contains messages in Italian and English
script_dir = Path(__file__).resolve()
messages_path = script_dir.parent / "json" / "messages.json"
with messages_path.open("r", encoding="utf-8") as m:
    MESSAGES = json.load(m)

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
        img, top, bottom, left, right,
        borderType=cv2.BORDER_CONSTANT,
        value=255)
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
    
    # Binarizes the image with a threshold
    # ----- TO REVIEW WITH ANDREA -----
    # Do we want to make the threshold a parameter? Yes
    _, binary = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 230, 255, cv2.THRESH_BINARY)

    # Finds the connected components
    _, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    # Sorts the components by area in descending order
    n_filtered = 10
    sorted_indices = np.argsort(stats[1:, cv2.CC_STAT_AREA])[::-1] + 1

    # Creates an empty mask to filter the components
    mask = np.zeros_like(binary).astype(np.uint8)

    # For each component in descending order of area
    for i in sorted_indices[:n_filtered]:
        x, y, w, h, area = stats[i]

        # If the component is too small, skip it
        if w < 100 and h < 100:
            continue
        
        # Fills the mask
        component_mask = (labels == i).astype(np.uint8) * 255
        countours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(component_mask, countours, -1, 255, thickness=cv2.FILLED)
        
        # Extracts from the region of interest
        roi = img[y:y+h, x:x+w]
        roi_mask = component_mask[y:y+h, x:x+w]

        # Calculates the standard deviation of the region of interest
        std_dev = cv2.meanStdDev(roi, mask=roi_mask)[1][0, 0]

        # Filters components with too high standard deviation
        if std_dev < 10:
            mask[component_mask == 255] = 255
    
    # Inverts the mask to get the foreground. The mask is 255 for the foreground and 0 for the background
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
    
    # Total mask
    mask = np.ones((img.shape[0], img.shape[1]), dtype=np.uint8) * 255 

    # Dividing the image into a grid of grid*grid, finds the mask for each region
    for y in range(0, img.shape[0], img.shape[0]//grid):
        for x in range(0, img.shape[1], img.shape[1]//grid):
            
            # Find the mask for the region of interest of size sub_shape
            roi = img[y:y+sub_shape[0], x:x+sub_shape[1]]
            roi_mask = calculate_mask(roi)
            
            # Pads the mask to keep the original size
            roi_mask = pad_image(roi_mask, x, y, img.shape[1], img.shape[0])
            
            # Updates the total mask
            mask = cv2.bitwise_and(mask, roi_mask)
    return mask

def calculate_mask_with_multiple_parameters(img: np.ndarray, parameters: list[tuple[int, int]]) -> np.ndarray:
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
    
    # Create an empty mask
    mask = np.ones((img.shape[0], img.shape[1]), dtype=np.uint8) * 255
    
    # We use different parameters: sub_shape (2=half side) and grid to find the mask (3=3 squares per side)
    for divisor, grid in parameters:
        
        # We find the mask for the image
        sub_shape = (img.shape[0]//divisor, img.shape[1]//divisor)
        
        # We find the mask with the specified parameters
        _mask = calculate_mask_with_grid(img, sub_shape, grid)

        # We update the total mask
        mask = cv2.bitwise_and(mask, _mask)
        
    return mask

def align_images(img1: np.ndarray, img2: np.ndarray, mask1: Optional[np.ndarray] = None, mask2: Optional[np.ndarray] = None, nfeatures: int = 10000, ed_distance: int = 200) -> tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
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
    """
    
    # Applying CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=18.0, tileGridSize=(8, 8))
    img1_clahe = img1
    img2_clahe = img2

    # If the images are in color, convert them to grayscale
    if len(img1_clahe.shape) == 3:
        img1_clahe = cv2.cvtColor(img1_clahe, cv2.COLOR_BGR2GRAY)
    if len(img2_clahe.shape) == 3:
        img2_clahe = cv2.cvtColor(img2_clahe, cv2.COLOR_BGR2GRAY)

    # Apply CLAHE to the images
    img1_clahe = clahe.apply(img1_clahe)
    img2_clahe = clahe.apply(img2_clahe)

    # Calculate features with SIFT (Scale-Invariant Feature Transform)
    sift = cv2.SIFT_create(nfeatures=nfeatures)
    keypoints_1, descriptors_1 = sift.detectAndCompute(img1_clahe, mask1)
    keypoints_2, descriptors_2 = sift.detectAndCompute(img2_clahe, mask2)

    # Check if there are enough features
    if len(keypoints_1) < 4 or len(keypoints_2) < 4:
        raise ValueError(MESSAGES["not_enough_features"][lang])

    # Feature matching with BFMatcher (Brute Force Matcher)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descriptors_1, descriptors_2)

    # Filtering matches based on Euclidean distance
    filtered_matches = []
    for match in matches:
        distance = np.linalg.norm(np.array(keypoints_1[match.queryIdx].pt) - np.array(keypoints_2[match.trainIdx].pt))
        if distance <= ed_distance:
            filtered_matches.append(match)
    
    # Check if there are enough matches
    if len(filtered_matches) < 4:
        raise ValueError(MESSAGES["not_enough_matches"][lang])
    
    # Extracting keypoints from filtered matches
    points_1 = np.float32([keypoints_1[match.queryIdx].pt for match in filtered_matches]).reshape(-1, 1, 2)
    points_2 = np.float32([keypoints_2[match.trainIdx].pt for match in filtered_matches]).reshape(-1, 1, 2)

    # Calculating the transformation matrix
    warp_matrix, mask = cv2.estimateAffinePartial2D(points_2, points_1)

    # Aligning the image and the mask
    img2_aligned = cv2.warpAffine(img2, warp_matrix, (img1.shape[1], img1.shape[0]))
    mask2_aligned = None
    if mask2 is not None:
        mask2_aligned = cv2.warpAffine(mask2, warp_matrix, (img1.shape[1], img1.shape[0]))

    return img2_aligned, mask2_aligned, warp_matrix

def align_from_scaled(img1: np.ndarray, img2: np.ndarray, scale: float = 0.5, mask1: Optional[np.ndarray] = None, mask2: Optional[np.ndarray] = None, nfeatures: int = 10000, ed_distance: int = 200) -> tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    # ----- TO REVIEW WITH ANDREA -----
    # do we want to make the scale factor a parameter? Yes
    # do we want to make the Euclidean distance threshold a parameter? Yes
    # do we want to make the number of features a parameter? Yes
    """
    Aligns two images by first scaling them, estimating the transformation on the scaled images, and then applying the transformation to the original images.
    
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
    """
    
    # Scale the images for alignment with a scale factor of 0.5 (i.e., halve the dimensions)
    img1_scaled = cv2.resize(img1, None, fx=scale, fy=scale)
    img2_scaled = cv2.resize(img2, None, fx=scale, fy=scale)

    if mask1 is None or mask2 is None:
        print(MESSAGES["scaled_mask_error"][lang])
        raise ValueError(MESSAGES["scaled_mask_error"][lang])
    else:
        mask1_scaled = cv2.resize(mask1, None, fx=scale, fy=scale)
        mask2_scaled = cv2.resize(mask2, None, fx=scale, fy=scale)
    
    # Aligns the scaled images using the standard function
    _, _, warp_matrix = align_images(img1_scaled, img2_scaled, mask1_scaled if mask1 is not None else None, mask2_scaled if mask2 is not None else None, nfeatures, ed_distance)

    # Adjusts the homography matrix to the original size
    warp_matrix[0, 2] /= scale
    warp_matrix[1, 2] /= scale

    # Aligns the original image with the calculated homography matrix
    img2_aligned = cv2.warpAffine(img2, warp_matrix, (img1.shape[1], img1.shape[0]))
    mask2_aligned = None
    if mask2 is not None:
        mask2_aligned = cv2.warpAffine(mask2, warp_matrix, (img1.shape[1], img1.shape[0]))

    return img2_aligned, mask2_aligned, warp_matrix

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
    
    return img[y:y+h, x:x+w]

def divide_image_with_grid(img: np.ndarray, img_size: tuple[int, int], grid_movement: tuple[int, int], mask: Optional[np.ndarray] = None, max_mask_percentage = 0.4) -> tuple[list[np.ndarray], Optional[list[np.ndarray]], list[tuple[int, int]]]:
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
        Maximum allowed percentage of masked (non-zero) pixels for a region to be included. Default is 0.4.
    
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

    # Extracting images
    for x in range(0, img.shape[1], grid_movement[0]):
        for y in range(0, img.shape[0], grid_movement[1]):

            # Extracting the region of interest from the image
            roi_img = extract_image(img, x, y, img_size[0], img_size[1])

            # If the image is too small, skip it
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

def divide_image_with_positions(img: np.ndarray, img_size: tuple[int, int], positions: list[tuple[int, int]]) -> list[np.ndarray]:
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
        # Extracting the region of interest from the image
        roi_img = extract_image(img, x, y, img_size[0], img_size[1])

        # If the image is too small, skip the square
        if roi_img.shape[0] < img_size[1] or roi_img.shape[1] < img_size[0]:
            continue
        images.append(roi_img)
    
    return images

def split_items(items: list, ratios: list[int]) -> list[list]:
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
    if sum(ratios) > 1:
        raise ValueError("The sum of ratios must be <= 1")
    if any(ratio < 0 for ratio in ratios):
        raise ValueError("All ratios must be >= 0")
    
    # Random shuffle of the list
    shuffled = items.copy()
    random.shuffle(shuffled)

    output = []
    start = 0
    for ratio in ratios:
        end = start + int(len(shuffled) * ratio)
        output.append(shuffled[start:end])
        start = end
    return output

ALLOWED_EXTENSIONS = {".tif", ".tiff", ".png"}

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

def main(
    path: str,
    source_name: str,
    target_name: str,
    seed: Optional[int] = None,
    save_masks: bool = False,
    image_size: tuple[int, int] = (256, 256),
    grid_movement: tuple[int, int] = (256, 256),
    margin: int = 200,
) -> None:

    # ====================[SET SEED]====================
    # Set seed for reproducibility
    if seed is None:
        seed = random.randint(0, 2**32 - 1)

    random.seed(seed)
    print(MESSAGES["seed_set"][lang].format(seed=seed))
    # ==================================================


    # ====================[LOAD IMAGES]====================
    # Loading images
    print(MESSAGES["loading_images"][lang].format(path=path))
    if not Path(path).exists():
        raise FileNotFoundError(MESSAGES["check_path"][lang].format(path=path)) 
    source_file = validate_image_filename(source_name, "Source")
    target_file = validate_image_filename(target_name, "Target")
    
    source_stem = source_file.stem
    target_stem = target_file.stem
    source_suffix = source_file.suffix.lower()
    target_suffix = target_file.suffix.lower()

    source_image = cv2.imread(Path(path) / source_file.name)
    target_image = cv2.imread(Path(path) / target_file.name)
    if source_image is None or target_image is None:
        raise FileNotFoundError(
            f"Missing paired files. Expected '{source_name}' and '{target_name}' inside: {path}"
        )
    print(
        MESSAGES["images_loaded"][lang].format(
            source_image_shape=source_image.shape,
            target_image_shape=target_image.shape,
        )
    )
    # =====================================================


    # ====================[MASK PROCESSING]====================
    # Calculate masks for source and target images
    print(MESSAGES["calculate_masks"][lang])
    source_mask = calculate_mask_with_multiple_parameters(source_image, [(2, 3), (4, 6), (6, 9), (8, 15)])
    target_mask = calculate_mask_with_multiple_parameters(target_image, [(2, 3), (4, 6), (6, 9), (8, 15)])
    print(MESSAGES["masks_calculated"][lang])
    # Saving masks
    if save_masks:
        cv2.imwrite(Path(path) / f"mask_{source_stem}{source_suffix}", source_mask)
        cv2.imwrite(Path(path) / f"mask_{target_stem}{target_suffix}", target_mask)
    print(MESSAGES["mask_saved"][lang])
    # =========================================================


    # ====================[IMAGE ALIGNMENT]==================== 
    # Aligning the target image to the source image
    print(MESSAGES["aligning_images"][lang])
    aligned_target, aligned_target_mask, warp_matrix = align_from_scaled(
        source_image,
        target_image,
        mask1=source_mask,
        mask2=target_mask,
        scale=0.5,
    )
    print(MESSAGES["images_aligned"][lang])
    # Saving aligned images
    cv2.imwrite(Path(path) / f"aligned_{target_stem}{target_suffix}", aligned_target)
    cv2.imwrite(Path(path) / f"aligned_mask_{target_stem}{target_suffix}", aligned_target_mask)
    print(MESSAGES["images_aligned_saved"][lang])
    # =========================================================


    # ====================[DATASET CREATION]==================== 
    # Extracting sub-images
    print(MESSAGES["extracting_subimages"][lang])
    # ----- TO REVIEW WITH ANDREA -----
    # Do we want to make the margin a parameter? Yes
    # Do we want to make the sub-image size a parameter? Yes
    # Do we want to make the grid step a parameter? Yes
    
    source_images, source_masks, positions = divide_image_with_grid(
        source_image[margin:-margin, margin:-margin],
        image_size,
        grid_movement,
        source_mask[margin:-margin, margin:-margin],
    )
    target_images = divide_image_with_positions(
        aligned_target[margin:-margin, margin:-margin],
        image_size,
        positions,
    )
    target_masks = divide_image_with_positions(
        aligned_target_mask[margin:-margin, margin:-margin],
        image_size,
        positions,
    )
    print(MESSAGES["total_subimages"][lang].format(count=len(source_images)))

    # Final robust pair filter: check both masks
    # and the white-background ratio in source and target patches.
    min_foreground_ratio = 0.25
    max_white_ratio = 0.7
    white_threshold = 250
    max_largest_white_component_ratio = 0.20

    # Keep valid and discarded pairs separate so they can be inspected later.
    named_source_images = []
    named_target_images = []
    discarded_source_images = []
    discarded_target_images = []
    discarded_log_rows = []

    for (x, y), source_img, source_patch_mask, target_img, target_patch_mask in zip(
        positions,
        source_images,
        source_masks,
        target_images,
        target_masks,
    ):
        patch_source_name = f"{x:05}_{y:05}_source{source_suffix}"
        patch_target_name = f"{x:05}_{y:05}_target{target_suffix}"

        is_valid, debug_info = is_valid_patch_pair(
            source_img=source_img,
            target_img=target_img,
            source_mask=source_patch_mask,
            target_mask=target_patch_mask,
            min_foreground_ratio=min_foreground_ratio,
            max_white_ratio=max_white_ratio,
            white_threshold=white_threshold,
            max_largest_white_component_ratio=max_largest_white_component_ratio,
        )

        if is_valid:
            named_source_images.append((source_img, patch_source_name))
            named_target_images.append((target_img, patch_target_name))
        else:
            discarded_source_images.append((source_img, patch_source_name))
            discarded_target_images.append((target_img, patch_target_name))

            discarded_log_rows.append(
                {
                    "sample_id": f"{x:05}_{y:05}",
                    "source_name": patch_source_name,
                    "target_name": patch_target_name,
                    "source_foreground_ratio": debug_info["source_foreground_ratio"],
                    "target_foreground_ratio": debug_info["target_foreground_ratio"],
                    "source_white_ratio": debug_info["source_white_ratio"],
                    "target_white_ratio": debug_info["target_white_ratio"],
                    "source_largest_white_component_ratio": debug_info["source_largest_white_component_ratio"],
                    "target_largest_white_component_ratio": debug_info["target_largest_white_component_ratio"],
                    "reasons": ";".join(debug_info["reasons"]),
                }
            )

    print(MESSAGES["pair_renamed"][lang])

    # Also prepare a discarded-patches folder for debugging and visual inspection.
    discarded_root = Path(path) / "discarded_patches"
    discarded_source_dir = discarded_root / "source"
    discarded_target_dir = discarded_root / "target"

    # Clean output folders to avoid leftovers from previous runs.
    ensure_clean_directory(Path(path) / "dataset_train")
    ensure_clean_directory(Path(path) / "dataset_val")
    ensure_clean_directory(Path(path) / "dataset_test")
    ensure_clean_directory(discarded_source_dir)
    ensure_clean_directory(discarded_target_dir)

    for source_img, patch_source_name in discarded_source_images:
        cv2.imwrite(discarded_source_dir / patch_source_name, source_img)
    for target_img, patch_target_name in discarded_target_images:
        cv2.imwrite(discarded_target_dir / patch_target_name, target_img)
    
    discarded_log_path = discarded_root / "discarded_log.csv"
    with open(discarded_log_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
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
                "target_largest_white_component_ratio"
            ],
        )
        writer.writeheader()
        writer.writerows(discarded_log_rows)

    # Splitting the dataset into training, validation, and testing
    print(MESSAGES["dataset_subdivision"][lang])
    images = list(zip(named_source_images, named_target_images))
    split = split_items(images, [0.8, 0.05, 0.15])
    print(MESSAGES["pair_number_division"][lang].format(train=len(split[0]), val=len(split[1]), test=len(split[2])))

    # Saving the split images
    for i, subset in enumerate(split):
        subset_name = ["dataset_train", "dataset_val", "dataset_test"][i]
        subset_dir = Path(path) / subset_name
        for source_img, target_img in subset:
            cv2.imwrite(subset_dir / source_img[1], source_img[0])
            cv2.imwrite(subset_dir / target_img[1], target_img[0])

    print(MESSAGES["dataset_saved"][lang])
    # ==========================================================

if __name__ == "__main__":
    # ====================[ARGUMENT PARSING]====================
    # Parsing command line arguments. The first parser is for the language, the second for the main arguments
    lang_parser = argparse.ArgumentParser(add_help=False)
    lang_parser.add_argument(
        "--lang",
        type=str,
        choices=["it", "en"],
        default="en",
        help="Language for messages (default: en)"
    )
    lang_args, _ = lang_parser.parse_known_args()
    # Loading messages based on the chosen language
    help_path = script_dir.parent / "json" / "help.json"
    with help_path.open("r", encoding="utf-8") as h:
        HELP = json.load(h)
    # Main parser for the arguments
    parser = argparse.ArgumentParser(
    usage=(
        "python src/prepare_dataset.py --path PATH\n"
        "       --source-name SOURCE_NAME --target-name TARGET_NAME\n"
        "       [--seed SEED] [--save-masks] [--image-size WIDTH HEIGHT]\n"
        "       [--grid-movement STEP_X STEP_Y] [--margin MARGIN] [--lang {en,it}]"
    ),
        description=HELP["description"][lang_args.lang],
        formatter_class=argparse.RawTextHelpFormatter,
        parents=[lang_parser]
    )
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help=HELP["path"][lang_args.lang]
    )
    parser.add_argument(
        "--source-name",
        type=str,
        required=True,
        help="Source image filename with extension (.tif, .tiff, .png)"
    )
    parser.add_argument(
        "--target-name",
        type=str,
        required=True,
        help="Target image filename with extension (.tif, .tiff, .png)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        help=HELP["seed"][lang_args.lang]
    )
    parser.add_argument(
        "--save-masks",
        action="store_true",
        help=HELP["save_masks"][lang_args.lang]
    )
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=(256, 256),
        help="Patch size used for extraction (default: 256 256)"
    )
    parser.add_argument(
        "--grid-movement",
        type=int,
        nargs=2,
        metavar=("STEP_X", "STEP_Y"),
        default=(256, 256),
        help="Grid step used for patch extraction (default: 256 256)"
    )
    parser.add_argument(
        "--margin",
        type=int,
        default=200,
        help="Margin cropped from each border before patch extraction (default: 200)"
    )
    args = parser.parse_args()
    lang = args.lang
    # ==========================================================


    # Running the main function with the specified arguments
    main(
        path=args.path,
        source_name=args.source_name,
        target_name=args.target_name,
        seed=args.seed,
        save_masks=args.save_masks,
        image_size=tuple(args.image_size),
        grid_movement=tuple(args.grid_movement),
        margin=args.margin,
    )

