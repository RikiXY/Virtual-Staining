from __future__ import annotations

from typing import Any
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from virtual_staining.data.preprocessing import (
    AlignmentMetadata,
    _aligned_mask_iou,
    _ratio_test_matches,
    align_from_scaled,
    align_images,
    apply_mask_morphology,
    calculate_hsv_tissue_mask,
    calculate_mask,
    calculate_mask_by_strategy,
    calculate_mask_with_grid,
    estimate_affine_transform,
    foreground_ratio_for_patch,
    is_valid_patch_pair,
    mask_window_for_patch,
    pad_image,
    split_items,
    warp_aligned_image,
    warp_aligned_mask_patch_from_mask_space,
    warp_aligned_patch,
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


# ---------------------------------------------------------------------------
# align_images - failure paths
# ---------------------------------------------------------------------------


def _textured_image(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(10, 200, (300, 300, 3), dtype=np.uint8)


def _dmatch(query_idx: int, train_idx: int, distance: float) -> cv2.DMatch:
    return cv2.DMatch(_queryIdx=query_idx, _trainIdx=train_idx, _distance=distance)


def _keypoints(count: int) -> list[cv2.KeyPoint]:
    return [cv2.KeyPoint(float(index), float(index * 2), 1.0) for index in range(count)]


def _descriptors(count: int) -> np.ndarray:
    return np.arange(count * 128, dtype=np.float32).reshape(count, 128)


class _FakeSift:
    def __init__(
        self,
        keypoints_1: list[cv2.KeyPoint],
        descriptors_1: np.ndarray | None,
        keypoints_2: list[cv2.KeyPoint],
        descriptors_2: np.ndarray | None,
    ) -> None:
        self._results = iter(
            [
                (keypoints_1, descriptors_1),
                (keypoints_2, descriptors_2),
            ]
        )

    def detectAndCompute(
        self, _img: np.ndarray, _mask: np.ndarray | None
    ) -> tuple[list[cv2.KeyPoint], np.ndarray | None]:
        return next(self._results)


class _FakeMatcher:
    def __init__(self, knn_matches: list[list[cv2.DMatch]]) -> None:
        self.knn_matches = knn_matches
        self.calls: list[tuple[np.ndarray, np.ndarray, int]] = []

    def knnMatch(
        self, descriptors_1: np.ndarray, descriptors_2: np.ndarray, k: int
    ) -> list[list[cv2.DMatch]]:
        self.calls.append((descriptors_1, descriptors_2, k))
        return self.knn_matches


def test_ratio_test_matches_keeps_only_distinct_best_descriptor_matches() -> None:
    matches = _ratio_test_matches(
        [
            [_dmatch(0, 0, 10.0), _dmatch(0, 1, 20.0)],
            [_dmatch(1, 1, 18.0), _dmatch(1, 2, 20.0)],
            [_dmatch(2, 2, 1.0)],
        ],
        ratio_threshold=0.75,
    )

    assert [(match.queryIdx, match.trainIdx) for match in matches] == [(0, 0)]


def test_estimate_affine_transform_uses_ratio_test_and_explicit_ransac() -> None:
    matcher = _FakeMatcher(
        [
            *[
                [_dmatch(index, index, 10.0), _dmatch(index, index + 1, 20.0)]
                for index in range(12)
            ],
            [_dmatch(12, 12, 19.0), _dmatch(12, 13, 20.0)],
        ]
    )
    eye = np.eye(2, 3, dtype=np.float64)
    inliers = np.ones((12, 1), dtype=np.uint8)

    def _bf_matcher(norm: int, **kwargs: Any) -> _FakeMatcher:
        assert norm == cv2.NORM_L2
        assert "crossCheck" not in kwargs
        return matcher

    with (
        patch(
            "virtual_staining.data.preprocessing.cv2.SIFT_create",
            return_value=_FakeSift(
                _keypoints(14),
                _descriptors(14),
                _keypoints(14),
                _descriptors(14),
            ),
        ),
        patch("virtual_staining.data.preprocessing.cv2.BFMatcher", side_effect=_bf_matcher),
        patch(
            "virtual_staining.data.preprocessing.cv2.estimateAffinePartial2D",
            return_value=(eye, inliers),
        ) as estimate_affine,
    ):
        mask = np.ones((300, 300), dtype=np.uint8) * 255
        _, metadata = estimate_affine_transform(
            _textured_image(),
            _textured_image(1),
            mask_1=mask,
            mask_2=mask,
            ratio_threshold=0.75,
        )

    assert metadata.n_matches == 12
    assert metadata.n_inliers == 12
    assert metadata.inlier_ratio == pytest.approx(1.0)
    assert metadata.mask_iou == pytest.approx(1.0)
    assert matcher.calls[0][2] == 2
    assert estimate_affine.call_args.kwargs == {
        "method": cv2.RANSAC,
        "ransacReprojThreshold": 5.0,
        "maxIters": 2000,
        "confidence": 0.99,
        "refineIters": 10,
    }


def test_estimate_affine_transform_raises_on_low_feature_count() -> None:
    with (
        patch(
            "virtual_staining.data.preprocessing.cv2.SIFT_create",
            return_value=_FakeSift(_keypoints(3), _descriptors(3), _keypoints(4), _descriptors(4)),
        ),
        pytest.raises(ValueError, match="Not enough features"),
    ):
        estimate_affine_transform(_textured_image(), _textured_image(1))


def test_estimate_affine_transform_raises_on_low_ratio_test_match_count() -> None:
    matcher = _FakeMatcher(
        [
            [_dmatch(0, 0, 19.0), _dmatch(0, 1, 20.0)],
            [_dmatch(1, 1, 19.0), _dmatch(1, 2, 20.0)],
            [_dmatch(2, 2, 19.0), _dmatch(2, 3, 20.0)],
            [_dmatch(3, 3, 10.0), _dmatch(3, 4, 20.0)],
        ]
    )

    with (
        patch(
            "virtual_staining.data.preprocessing.cv2.SIFT_create",
            return_value=_FakeSift(_keypoints(4), _descriptors(4), _keypoints(5), _descriptors(5)),
        ),
        patch("virtual_staining.data.preprocessing.cv2.BFMatcher", return_value=matcher),
        pytest.raises(ValueError, match="Not enough good descriptor matches"),
    ):
        estimate_affine_transform(_textured_image(), _textured_image(1))


def test_estimate_affine_transform_raises_on_low_inlier_ratio() -> None:
    match_count = 200
    matcher = _FakeMatcher(
        [
            [_dmatch(index, index, 10.0), _dmatch(index, index + 1, 20.0)]
            for index in range(match_count)
        ]
    )
    eye = np.eye(2, 3, dtype=np.float64)
    low_ratio_inliers = np.zeros((match_count, 1), dtype=np.uint8)
    low_ratio_inliers[:12] = 1

    with (
        patch(
            "virtual_staining.data.preprocessing.cv2.SIFT_create",
            return_value=_FakeSift(
                _keypoints(match_count + 1),
                _descriptors(match_count + 1),
                _keypoints(match_count + 1),
                _descriptors(match_count + 1),
            ),
        ),
        patch("virtual_staining.data.preprocessing.cv2.BFMatcher", return_value=matcher),
        patch(
            "virtual_staining.data.preprocessing.cv2.estimateAffinePartial2D",
            return_value=(eye, low_ratio_inliers),
        ),
        pytest.raises(ValueError, match="inlier ratio"),
    ):
        estimate_affine_transform(_textured_image(), _textured_image(1))


def test_aligned_mask_iou_returns_overlap_diagnostic() -> None:
    mask_1 = np.zeros((8, 8), dtype=np.uint8)
    mask_2 = np.zeros((8, 8), dtype=np.uint8)
    mask_1[2:6, 2:6] = 255
    mask_2[2:6, 2:6] = 255

    iou = _aligned_mask_iou(
        mask_1,
        mask_2,
        np.eye(2, 3, dtype=np.float64),
        (8, 8),
    )

    assert iou == pytest.approx(1.0)


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
        mask_iou=0.75,
    )
    assert meta.n_keypoints_src == 200
    assert meta.n_keypoints_tgt == 180
    assert meta.n_matches == 60
    assert meta.n_inliers == 50
    assert meta.inlier_ratio == 50 / 60
    assert meta.mask_iou == pytest.approx(0.75)
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
        align_from_scaled(img, img, scale=0.5, mask_1=mask, mask_2=mask)

    assert any(call.get("interpolation") == cv2.INTER_NEAREST for call in resize_calls[2:4])


def test_warp_aligned_patch_matches_full_frame_warp_crop() -> None:
    rng = np.random.default_rng(4)
    img = rng.integers(0, 255, size=(96, 112, 3), dtype=np.uint8)
    warp_matrix = np.array(
        [
            [0.998, -0.035, 7.4],
            [0.035, 0.998, -5.2],
        ],
        dtype=np.float64,
    )

    full = warp_aligned_image(
        img,
        warp_matrix,
        (90, 80),
        is_mask=False,
    )
    patch = warp_aligned_patch(
        img,
        warp_matrix,
        x=13,
        y=17,
        output_size=(32, 24),
        is_mask=False,
    )

    assert np.allclose(patch, full[17 : 17 + 24, 13 : 13 + 32], atol=8)


def test_warp_aligned_patch_uses_nearest_neighbor_for_masks() -> None:
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[:, 16:] = 255
    warp_matrix = np.array(
        [
            [1.0, 0.0, 0.4],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )

    patch = warp_aligned_patch(
        mask,
        warp_matrix,
        x=0,
        y=0,
        output_size=(32, 32),
        is_mask=True,
    )

    assert set(np.unique(patch)).issubset({0, 255})


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


def test_warp_aligned_mask_patch_from_mask_space_scales_affine_columns() -> None:
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[4:12, 4:12] = 255
    warp_matrix = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )

    patch = warp_aligned_mask_patch_from_mask_space(
        mask,
        warp_matrix,
        (32, 32, 3),
        x=8,
        y=8,
        output_size=(16, 16),
    )

    assert patch.shape == (16, 16)
    assert set(np.unique(patch)).issubset({0, 255})
    assert cv2.countNonZero(patch) > 0
