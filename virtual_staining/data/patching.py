from __future__ import annotations

from collections.abc import Iterator

import numpy as np


def iter_patch_origins(
    *,
    image_size: tuple[int, int],
    patch_size: tuple[int, int],
    grid_movement: tuple[int, int],
    margin: int,
) -> Iterator[tuple[int, int]]:
    image_width, image_height = image_size
    patch_width, patch_height = patch_size
    step_x, step_y = grid_movement

    stop_x = max(margin, image_width - margin - patch_width + 1)
    stop_y = max(margin, image_height - margin - patch_height + 1)
    for x in range(margin, stop_x, step_x):
        for y in range(margin, stop_y, step_y):
            yield x, y


def mask_window_for_patch(
    mask: np.ndarray,
    image_shape: tuple[int, int] | tuple[int, int, int],
    *,
    x: int,
    y: int,
    width: int,
    height: int,
) -> np.ndarray:
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
