from __future__ import annotations

from typing import Any
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from virtual_staining.data.preprocessing import (
    apply_mask_morphology,
    calculate_hsv_tissue_mask,
    calculate_mask,
    calculate_mask_by_strategy,
    calculate_mask_with_grid,
    foreground_ratio_for_patch,
    is_valid_patch_pair,
    mask_window_for_patch,
    pad_image,
    split_items,
)

# ---------------------------------------------------------------------------
# split_items
# ---------------------------------------------------------------------------


def test_split_items_covers_all_items() -> None:
    items = list(range(100))
    parts = split_items(items, [0.7, 0.15, 0.15])
    assert sum(len(p) for p in parts) == 100


def test_split_items_respects_ratios() -> None:
    items = list(range(100))
    parts = split_items(items, [0.8, 0.1, 0.1])
    assert len(parts[0]) == 80
    assert len(parts[1]) == 10
    assert len(parts[2]) == 10


def test_split_items_raises_on_single_ratio() -> None:
    with pytest.raises(ValueError):
        split_items([1, 2, 3], [1.0])


def test_split_items_raises_on_sum_exceeds_one() -> None:
    with pytest.raises(ValueError):
        split_items([1, 2, 3], [0.6, 0.6])


def test_split_items_raises_on_sum_below_one() -> None:
    with pytest.raises(ValueError):
        split_items([1, 2, 3], [0.5, 0.3])


def test_split_items_raises_on_negative_ratio() -> None:
    with pytest.raises(ValueError):
        split_items([1, 2, 3], [0.8, -0.1, 0.3])


def test_split_items_81_items_no_loss() -> None:
    items = list(range(81))
    parts = split_items(items, [0.8, 0.1, 0.1])
    assert sum(len(p) for p in parts) == 81


def test_split_items_101_items_no_loss() -> None:
    items = list(range(101))
    parts = split_items(items, [0.7, 0.15, 0.15])
    assert sum(len(p) for p in parts) == 101


def test_split_items_no_duplicates_no_missing() -> None:
    items = list(range(81))
    parts = split_items(items, [0.8, 0.1, 0.1])
    all_items = [item for part in parts for item in part]
    assert sorted(all_items) == items


# ---------------------------------------------------------------------------
# calculate_mask_with_grid
# ---------------------------------------------------------------------------


def _reference_calculate_mask_with_grid(
    img: np.ndarray, sub_shape: tuple[int, int], grid: int
) -> np.ndarray:
    mask = np.ones((img.shape[0], img.shape[1]), dtype=np.uint8) * 255

    for y in range(0, img.shape[0], img.shape[0] // grid):
        for x in range(0, img.shape[1], img.shape[1] // grid):
            roi = img[y : y + sub_shape[0], x : x + sub_shape[1]]
            roi_mask = calculate_mask(roi)
            roi_mask = pad_image(roi_mask, x, y, img.shape[1], img.shape[0])
            mask = cv2.bitwise_and(mask, roi_mask)

    return mask


def test_calculate_mask_with_grid_matches_reference_behavior() -> None:
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)

    mask = calculate_mask_with_grid(img, sub_shape=(16, 16), grid=4)
    reference_mask = _reference_calculate_mask_with_grid(img, sub_shape=(16, 16), grid=4)

    assert np.array_equal(mask, reference_mask)


def test_calculate_mask_with_grid_returns_expected_shape_and_dtype() -> None:
    rng = np.random.default_rng(1)
    img = rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)

    mask = calculate_mask_with_grid(img, sub_shape=(16, 16), grid=4)

    assert mask.shape == (64, 64)
    assert mask.dtype == np.uint8


def test_calculate_mask_with_grid_does_not_call_pad_image() -> None:
    rng = np.random.default_rng(2)
    img = rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)

    with patch("virtual_staining.data.preprocessing.pad_image") as mock_pad_image:
        calculate_mask_with_grid(img, sub_shape=(16, 16), grid=4)

    mock_pad_image.assert_not_called()


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


# ---------------------------------------------------------------------------
# is_valid_patch_pair helpers
# ---------------------------------------------------------------------------


def _solid_bgr(value: int, size: int = 32) -> np.ndarray:
    return np.full((size, size, 3), value, dtype=np.uint8)


def _solid_mask(value: int, size: int = 32) -> np.ndarray:
    return np.full((size, size), value, dtype=np.uint8)


def _call(
    src_bgr: np.ndarray,
    tgt_bgr: np.ndarray,
    src_mask: np.ndarray,
    tgt_mask: np.ndarray,
    min_fg: float = 0.25,
    max_white: float = 0.7,
    white_threshold: int = 250,
    max_lw: float = 0.20,
) -> tuple[bool, dict[str, Any]]:
    return is_valid_patch_pair(
        source_img=src_bgr,
        target_img=tgt_bgr,
        source_mask=src_mask,
        target_mask=tgt_mask,
        min_foreground_ratio=min_fg,
        max_white_ratio=max_white,
        white_threshold=white_threshold,
        max_largest_white_component_ratio=max_lw,
    )


# ---------------------------------------------------------------------------
# is_valid_patch_pair - acceptance
# ---------------------------------------------------------------------------


def test_valid_pair_is_accepted() -> None:
    tissue = _solid_bgr(80)  # dark BGR, not white
    mask = _solid_mask(255)  # fully foreground
    valid, info = _call(tissue, tissue, mask, mask)
    assert valid is True
    assert info["reasons"] == []


# ---------------------------------------------------------------------------
# is_valid_patch_pair - rejection reasons
# ---------------------------------------------------------------------------


def test_rejects_low_source_foreground() -> None:
    tissue = _solid_bgr(80)
    background_mask = _solid_mask(0)  # no foreground at all
    valid, info = _call(tissue, tissue, background_mask, _solid_mask(255))
    assert valid is False
    assert "low_source_foreground" in info["reasons"]


def test_rejects_low_target_foreground() -> None:
    tissue = _solid_bgr(80)
    valid, info = _call(tissue, tissue, _solid_mask(255), _solid_mask(0))
    assert valid is False
    assert "low_target_foreground" in info["reasons"]


def test_rejects_high_source_white_ratio() -> None:
    white = _solid_bgr(255)
    tissue = _solid_bgr(80)
    mask = _solid_mask(255)
    valid, info = _call(white, tissue, mask, mask, max_white=0.3)
    assert valid is False
    assert "high_source_white_ratio" in info["reasons"]


def test_rejects_high_target_white_ratio() -> None:
    white = _solid_bgr(255)
    tissue = _solid_bgr(80)
    mask = _solid_mask(255)
    valid, info = _call(tissue, white, mask, mask, max_white=0.3)
    assert valid is False
    assert "high_target_white_ratio" in info["reasons"]


# ---------------------------------------------------------------------------
# is_valid_patch_pair - debug_info keys
# ---------------------------------------------------------------------------


def test_debug_info_contains_required_keys() -> None:
    tissue = _solid_bgr(80)
    mask = _solid_mask(255)
    _, info = _call(tissue, tissue, mask, mask)
    expected_keys = {
        "source_foreground_ratio",
        "target_foreground_ratio",
        "source_white_ratio",
        "target_white_ratio",
        "source_largest_white_component_ratio",
        "target_largest_white_component_ratio",
        "reasons",
    }
    assert expected_keys.issubset(info.keys())


def test_mask_window_for_patch_maps_full_resolution_patch_to_mask_space() -> None:
    mask = np.zeros((5, 10), dtype=np.uint8)
    mask[1:4, 2:7] = 255

    window = mask_window_for_patch(
        mask,
        (20, 40, 3),
        x=8,
        y=4,
        width=12,
        height=8,
    )

    assert window.shape == (2, 3)
    assert np.all(window == 255)


def test_foreground_ratio_for_patch_uses_mask_space_window() -> None:
    mask = np.array(
        [
            [255, 0],
            [255, 255],
        ],
        dtype=np.uint8,
    )

    ratio = foreground_ratio_for_patch(mask, (8, 8, 3), x=0, y=0, width=8, height=8)

    assert ratio == pytest.approx(0.75)
