from __future__ import annotations

from typing import Any
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from virtual_staining.data.preprocessing import (
    AlignmentMetadata,
    align_from_scaled,
    align_images,
    calculate_mask,
    calculate_mask_with_grid,
    is_valid_patch_pair,
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


def _legacy_calculate_mask_with_grid(
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


def test_calculate_mask_with_grid_matches_legacy_behavior() -> None:
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)

    mask = calculate_mask_with_grid(img, sub_shape=(16, 16), grid=4)
    legacy_mask = _legacy_calculate_mask_with_grid(img, sub_shape=(16, 16), grid=4)

    assert np.array_equal(mask, legacy_mask)


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


# ---------------------------------------------------------------------------
# align_images - failure paths
# ---------------------------------------------------------------------------


def _textured_image(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(10, 200, (300, 300, 3), dtype=np.uint8)


def test_align_images_raises_when_warp_matrix_is_none() -> None:
    img = _textured_image()
    with (
        patch(
            "virtual_staining.data.preprocessing.cv2.estimateAffinePartial2D",
            return_value=(None, None),
        ),
        pytest.raises(ValueError, match="Affine estimation failed"),
    ):
        align_images(img, img)


def test_align_images_raises_on_low_inlier_count() -> None:
    img = _textured_image()
    zero_inliers = np.zeros((50, 1), dtype=np.uint8)
    with (
        patch(
            "virtual_staining.data.preprocessing.cv2.estimateAffinePartial2D",
            return_value=(np.eye(2, 3, dtype=np.float64), zero_inliers),
        ),
        pytest.raises(ValueError, match="inliers"),
    ):
        align_images(img, img)


# ---------------------------------------------------------------------------
# AlignmentMetadata - structure
# ---------------------------------------------------------------------------


def test_alignment_metadata_has_expected_fields() -> None:
    eye = np.eye(2, 3, dtype=np.float64)
    meta = AlignmentMetadata(
        n_keypoints_src=200,
        n_keypoints_tgt=180,
        n_matches=60,
        n_inliers=50,
        inlier_ratio=50 / 60,
        scale_x=1.0,
        scale_y=1.0,
        rotation_deg=0.0,
        translation_x=0.0,
        translation_y=0.0,
        warp_matrix=eye.tolist(),
    )
    assert meta.n_keypoints_src == 200
    assert meta.n_keypoints_tgt == 180
    assert meta.n_matches == 60
    assert meta.n_inliers == 50
    assert meta.inlier_ratio == 50 / 60
    assert len(meta.warp_matrix) == 2
    assert len(meta.warp_matrix[0]) == 3


def test_align_from_scaled_uses_nearest_neighbor_for_scaled_masks() -> None:
    img = _textured_image()
    mask = np.full(img.shape[:2], 255, dtype=np.uint8)
    metadata = AlignmentMetadata(
        n_keypoints_src=100,
        n_keypoints_tgt=100,
        n_matches=50,
        n_inliers=45,
        inlier_ratio=0.9,
        scale_x=1.0,
        scale_y=1.0,
        rotation_deg=0.0,
        translation_x=0.0,
        translation_y=0.0,
        warp_matrix=np.eye(2, 3, dtype=np.float64).tolist(),
    )

    resize_calls: list[dict[str, Any]] = []
    original_resize = cv2.resize

    def _record_resize(*args: Any, **kwargs: Any) -> np.ndarray:
        resize_calls.append(kwargs)
        return original_resize(*args, **kwargs)

    with (
        patch("virtual_staining.data.preprocessing.cv2.resize", side_effect=_record_resize),
        patch(
            "virtual_staining.data.preprocessing.estimate_affine_transform",
            return_value=(np.eye(2, 3, dtype=np.float64), metadata),
        ),
    ):
        align_from_scaled(img, img, scale=0.5, mask1=mask, mask2=mask)

    assert any(call.get("interpolation") == cv2.INTER_NEAREST for call in resize_calls[2:4])
