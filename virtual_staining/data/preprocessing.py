from __future__ import annotations

import cv2
import numpy as np

from virtual_staining.config.data import (
    ALLOWED_MASK_STRATEGIES,
    MASK_STRATEGY_CONNECTED_COMPONENTS,
    MASK_STRATEGY_HSV,
)

N_TOP_COMPONENTS = 10
MIN_STD_DEV = 15

MASK_PARAMETER_GRID = [(2, 3), (4, 6), (6, 9), (8, 15)]


def calculate_mask(img: np.ndarray) -> np.ndarray:
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

    mask = cv2.bitwise_not(mask)
    return mask


def calculate_mask_with_grid(img: np.ndarray, sub_shape: tuple[int, int], grid: int) -> np.ndarray:
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
    mask = np.ones((img.shape[0], img.shape[1]), dtype=np.uint8) * 255

    for divisor, grid in parameters:
        sub_shape = (img.shape[0] // divisor, img.shape[1] // divisor)

        _mask = calculate_mask_with_grid(img, sub_shape, grid)

        mask = cv2.bitwise_and(mask, _mask)

    return mask


def apply_mask_morphology(mask: np.ndarray, *, kernel_size: int = 5) -> np.ndarray:
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
