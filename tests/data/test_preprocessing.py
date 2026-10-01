from __future__ import annotations

import cv2
import numpy as np
import pytest

from virtual_staining.data.preprocessing import (
    apply_mask_morphology,
    calculate_hsv_tissue_mask,
    calculate_mask,
    calculate_mask_by_strategy,
    calculate_mask_with_grid,
)


def test_calculate_mask_with_grid_matches_reference_behavior() -> None:
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
    expected = np.ones((64, 64), dtype=np.uint8) * 255

    for y in range(0, 64, 16):
        for x in range(0, 64, 16):
            roi_mask = calculate_mask(img[y : y + 16, x : x + 16])
            expected[y : y + 16, x : x + 16] = cv2.bitwise_and(
                expected[y : y + 16, x : x + 16],
                roi_mask,
            )

    actual = calculate_mask_with_grid(img, sub_shape=(16, 16), grid=4)

    assert np.array_equal(actual, expected)


def test_calculate_mask_with_grid_handles_grid_larger_than_image() -> None:
    rng = np.random.default_rng(3)
    img = rng.integers(0, 255, size=(4, 5, 3), dtype=np.uint8)

    mask = calculate_mask_with_grid(img, sub_shape=(2, 2), grid=10)

    assert mask.shape == (4, 5)
    assert mask.dtype == np.uint8


def test_calculate_mask_by_strategy_preserves_connected_components_default() -> None:
    img = np.full((128, 128, 3), 255, dtype=np.uint8)
    cv2.rectangle(img, (32, 32), (95, 95), (80, 80, 80), thickness=-1)

    direct = calculate_mask_by_strategy(
        img,
        strategy="connected_components",
        parameters=[(2, 2)],
    )
    expected = calculate_mask_with_grid(img, (64, 64), 2)

    assert np.array_equal(direct, expected)


def test_calculate_mask_uses_max_channel_std_for_background_components() -> None:
    img = np.full((128, 128, 3), 255, dtype=np.uint8)
    img[:, 64:, 1] = np.tile(np.arange(64, dtype=np.uint8), (128, 1))

    mask = calculate_mask(img)

    assert mask[32, 32] == 0
    assert mask[32, 96] == 255


def test_calculate_hsv_tissue_mask_detects_saturated_tissue_on_white_background() -> None:
    img = np.full((80, 80, 3), 255, dtype=np.uint8)
    cv2.rectangle(img, (20, 20), (59, 59), (120, 40, 180), thickness=-1)

    mask = calculate_hsv_tissue_mask(img, min_saturation=20, max_value=245)

    assert mask[40, 40] == 255
    assert mask[5, 5] == 0
    assert set(np.unique(mask)).issubset({0, 255})


def test_apply_mask_morphology_removes_small_speckles_and_fills_holes() -> None:
    mask = np.zeros((64, 64), dtype=np.uint8)
    cv2.rectangle(mask, (16, 16), (47, 47), 255, thickness=-1)
    cv2.rectangle(mask, (28, 28), (35, 35), 0, thickness=-1)
    mask[2, 2] = 255

    cleaned = apply_mask_morphology(mask, kernel_size=9)

    assert cleaned[2, 2] == 0
    assert cleaned[32, 32] == 255


def test_calculate_mask_by_strategy_rejects_unknown_strategy() -> None:
    img = np.full((16, 16, 3), 255, dtype=np.uint8)

    with pytest.raises(ValueError, match="Unknown mask strategy"):
        calculate_mask_by_strategy(img, strategy="unknown")
