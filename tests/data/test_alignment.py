from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import cv2
import numpy as np
import pytest

from virtual_staining.config.data import AlignmentConfig
from virtual_staining.data.alignment import (
    AlignmentError,
    AlignmentImage,
    AlignmentResult,
    identity_alignment,
    resolve_alignment,
    warp_aligned_mask_patch,
    warp_aligned_patch,
)
from virtual_staining.data.alignment.models import _RegistrationDiagnostics
from virtual_staining.data.alignment.registration import (
    _aligned_mask_iou,
    _estimate_affine,
    _ratio_test_matches,
)
from virtual_staining.data.alignment.warping import _warp_image
from virtual_staining.utils.image_io import PillowRegionImageReader

# ---------------------------------------------------------------------------
# _estimate_affine - failure paths
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


def test_estimate_affine_uses_ratio_test_and_explicit_ransac() -> None:
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
            "virtual_staining.data.alignment.registration.cv2.SIFT_create",
            return_value=_FakeSift(
                _keypoints(14),
                _descriptors(14),
                _keypoints(14),
                _descriptors(14),
            ),
        ),
        patch(
            "virtual_staining.data.alignment.registration.cv2.BFMatcher", side_effect=_bf_matcher
        ),
        patch(
            "virtual_staining.data.alignment.registration.cv2.estimateAffinePartial2D",
            return_value=(eye, inliers),
        ) as estimate_affine,
    ):
        mask = np.ones((300, 300), dtype=np.uint8) * 255
        _, metadata = _estimate_affine(
            _textured_image(),
            _textured_image(1),
            reference_mask=mask,
            moving_mask=mask,
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


@pytest.mark.parametrize(
    ("name", "mask"),
    [
        pytest.param("reference_mask", np.ones((299, 300), dtype=np.uint8), id="mask-1-2d"),
        pytest.param("moving_mask", np.ones((300, 299), dtype=np.uint8), id="mask-2-2d"),
        pytest.param("reference_mask", np.ones((300, 300, 1), dtype=np.uint8), id="mask-1-3d"),
        pytest.param("moving_mask", np.ones((300, 300, 1), dtype=np.uint8), id="mask-2-3d"),
    ],
)
def test_estimate_affine_rejects_invalid_mask_geometry_before_sift(
    name: str, mask: np.ndarray
) -> None:
    with (
        patch("virtual_staining.data.alignment.registration.cv2.SIFT_create") as sift,
        pytest.raises(ValueError, match=rf"{name} geometry"),
    ):
        if name == "reference_mask":
            _estimate_affine(_textured_image(), _textured_image(), reference_mask=mask)
        else:
            _estimate_affine(_textured_image(), _textured_image(), moving_mask=mask)

    sift.assert_not_called()


def test_estimate_affine_accepts_none_masks() -> None:
    image = _textured_image()

    warp_matrix, metadata = _estimate_affine(image, image, reference_mask=None, moving_mask=None)

    np.testing.assert_allclose(warp_matrix, np.eye(2, 3), atol=0.01)
    assert metadata.mask_iou is None


def test_estimate_affine_raises_on_low_feature_count() -> None:
    with (
        patch(
            "virtual_staining.data.alignment.registration.cv2.SIFT_create",
            return_value=_FakeSift(_keypoints(3), _descriptors(3), _keypoints(4), _descriptors(4)),
        ),
        pytest.raises(ValueError, match="Not enough features"),
    ):
        _estimate_affine(_textured_image(), _textured_image(1))


def test_estimate_affine_raises_on_low_ratio_test_match_count() -> None:
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
            "virtual_staining.data.alignment.registration.cv2.SIFT_create",
            return_value=_FakeSift(_keypoints(4), _descriptors(4), _keypoints(5), _descriptors(5)),
        ),
        patch("virtual_staining.data.alignment.registration.cv2.BFMatcher", return_value=matcher),
        pytest.raises(ValueError, match="Not enough good descriptor matches"),
    ):
        _estimate_affine(_textured_image(), _textured_image(1))


def test_estimate_affine_raises_on_low_inlier_ratio() -> None:
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
            "virtual_staining.data.alignment.registration.cv2.SIFT_create",
            return_value=_FakeSift(
                _keypoints(match_count + 1),
                _descriptors(match_count + 1),
                _keypoints(match_count + 1),
                _descriptors(match_count + 1),
            ),
        ),
        patch("virtual_staining.data.alignment.registration.cv2.BFMatcher", return_value=matcher),
        patch(
            "virtual_staining.data.alignment.registration.cv2.estimateAffinePartial2D",
            return_value=(eye, low_ratio_inliers),
        ),
        pytest.raises(ValueError, match="inlier ratio"),
    ):
        _estimate_affine(_textured_image(), _textured_image(1))


def test_aligned_mask_iou_returns_overlap_diagnostic() -> None:
    reference_mask = np.zeros((8, 8), dtype=np.uint8)
    moving_mask = np.zeros((8, 8), dtype=np.uint8)
    reference_mask[2:6, 2:6] = 255
    moving_mask[2:6, 2:6] = 255

    iou = _aligned_mask_iou(
        reference_mask,
        moving_mask,
        np.eye(2, 3, dtype=np.float64),
        (8, 8),
    )

    assert iou == pytest.approx(1.0)


def test_estimate_affine_raises_when_warp_matrix_is_none() -> None:
    img = _textured_image()
    with (
        patch(
            "virtual_staining.data.alignment.registration.cv2.estimateAffinePartial2D",
            return_value=(None, None),
        ),
        pytest.raises(ValueError, match="Affine estimation failed"),
    ):
        _estimate_affine(img, img)


def test_estimate_affine_raises_on_low_inlier_count() -> None:
    img = _textured_image()
    zero_inliers = np.zeros((50, 1), dtype=np.uint8)
    with (
        patch(
            "virtual_staining.data.alignment.registration.cv2.estimateAffinePartial2D",
            return_value=(np.eye(2, 3, dtype=np.float64), zero_inliers),
        ),
        pytest.raises(ValueError, match="inliers"),
    ):
        _estimate_affine(img, img)


# ---------------------------------------------------------------------------
# _RegistrationDiagnostics - structure
# ---------------------------------------------------------------------------


def test_alignment_metadata_has_expected_fields() -> None:
    eye = np.eye(2, 3, dtype=np.float64)
    meta = _RegistrationDiagnostics(
        n_keypoints_reference=200,
        n_keypoints_moving=180,
        n_matches=60,
        n_inliers=50,
        inlier_ratio=50 / 60,
        scale_x=1.0,
        scale_y=1.0,
        rotation_deg=0.0,
        translation_x=0.0,
        translation_y=0.0,
        mask_iou=0.75,
    )
    assert meta.n_keypoints_reference == 200
    assert meta.n_keypoints_moving == 180
    assert meta.n_matches == 60
    assert meta.n_inliers == 50
    assert meta.inlier_ratio == 50 / 60
    assert meta.mask_iou == pytest.approx(0.75)
    result = AlignmentResult("affine_sift", eye, diagnostics=meta)
    assert result.metadata["warp_matrix"] == eye.tolist()
    assert result.metadata["n_keypoints_src"] == 200
    assert result.metadata["n_keypoints_tgt"] == 180


def test_resolve_alignment_uses_nearest_neighbor_for_scaled_masks() -> None:
    img = _textured_image()
    mask = np.full(img.shape[:2], 255, dtype=np.uint8)
    metadata = _RegistrationDiagnostics(
        n_keypoints_reference=100,
        n_keypoints_moving=100,
        n_matches=50,
        n_inliers=45,
        inlier_ratio=0.9,
        scale_x=1.0,
        scale_y=1.0,
        rotation_deg=0.0,
        translation_x=0.0,
        translation_y=0.0,
    )

    resize_calls: list[dict[str, Any]] = []
    original_resize = cv2.resize

    def _record_resize(*args: Any, **kwargs: Any) -> np.ndarray:
        resize_calls.append(kwargs)
        return original_resize(*args, **kwargs)

    with (
        patch(
            "virtual_staining.data.alignment.registration.cv2.resize", side_effect=_record_resize
        ),
        patch(
            "virtual_staining.data.alignment.registration._estimate_affine",
            return_value=(np.eye(2, 3, dtype=np.float64), metadata),
        ),
    ):
        data = AlignmentImage(img, img.shape[:2], mask)
        resolve_alignment(data, data, AlignmentConfig())

    assert len(resize_calls) == 4
    assert all(resize_calls[i]["interpolation"] == cv2.INTER_NEAREST for i in (1, 3))


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

    full = _warp_image(
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


def test_warp_aligned_mask_patch_scales_affine_columns() -> None:
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[4:12, 4:12] = 255
    warp_matrix = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )

    patch = warp_aligned_mask_patch(
        mask,
        warp_matrix,
        (32, 32),
        x=8,
        y=8,
        output_size=(16, 16),
    )

    assert patch.shape == (16, 16)
    assert set(np.unique(patch)).issubset({0, 255})
    assert cv2.countNonZero(patch) > 0


def _centered_tissue_preview(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    image = np.full((*shape, 3), 255, dtype=np.uint8)
    mask = np.zeros(shape, dtype=np.uint8)
    top, bottom = shape[0] // 4, 3 * shape[0] // 4
    left, right = shape[1] // 4, 3 * shape[1] // 4
    rng = np.random.default_rng(0)
    image[top:bottom, left:right] = rng.integers(
        10, 200, image[top:bottom, left:right].shape, dtype=np.uint8
    )
    mask[top:bottom, left:right] = 255
    return image, mask


@pytest.mark.parametrize("full_size_masks", [False, True], ids=["preview", "full_size"])
def test_resolve_alignment_resizes_masks_to_each_preview(full_size_masks: bool) -> None:
    reference_preview, reference_mask = _centered_tissue_preview((640, 640))
    moving_preview = cv2.warpAffine(
        reference_preview,
        np.array([[1.0, 0.0, 12.0], [0.0, 1.0, -8.0]]),
        (652, 624),
        borderValue=(255, 255, 255),
    )
    moving_mask = cv2.warpAffine(
        reference_mask,
        np.array([[1.0, 0.0, 12.0], [0.0, 1.0, -8.0]]),
        (652, 624),
        flags=cv2.INTER_NEAREST,
    )
    reference_input_mask = (
        cv2.resize(reference_mask, (2560, 2560), interpolation=cv2.INTER_NEAREST)
        if full_size_masks
        else reference_mask.copy()
    )
    moving_input_mask = (
        cv2.resize(moving_mask, (2608, 2496), interpolation=cv2.INTER_NEAREST)
        if full_size_masks
        else moving_mask.copy()
    )
    reference_mask_before = reference_input_mask.copy()
    moving_mask_before = moving_input_mask.copy()
    reference = AlignmentImage(
        name="reference",
        preview=reference_preview,
        mask=reference_input_mask,
        full_shape=(2560, 2560),
    )
    moving = AlignmentImage(
        name="moving",
        preview=moving_preview,
        mask=moving_input_mask,
        full_shape=(2496, 2608),
    )

    result = resolve_alignment(reference, moving, AlignmentConfig(mode="always"))

    np.testing.assert_allclose(result.warp_matrix[:, :2], np.eye(2), atol=0.05)
    np.testing.assert_allclose(result.warp_matrix[:, 2], [-48.0, 32.0], atol=4.0)
    assert result.diagnostics is not None
    assert result.diagnostics.mask_iou is not None and result.diagnostics.mask_iou > 0.9
    assert reference.mask is not None
    assert moving.mask is not None
    assert np.array_equal(reference.mask, reference_mask_before)
    assert np.array_equal(moving.mask, moving_mask_before)


def _diagnostics() -> _RegistrationDiagnostics:
    return _RegistrationDiagnostics(100, 90, 20, 18, 0.9, 1.0, 1.0, 0.0, 0.0, 0.0, 0.8)


def _image(
    full_shape: tuple[int, int] = (80, 120), preview_shape: tuple[int, int] = (20, 30)
) -> AlignmentImage:
    return AlignmentImage(
        np.zeros((*preview_shape, 3), dtype=np.uint8),
        full_shape,
        np.full(preview_shape, 255, dtype=np.uint8),
        name="moving",
    )


def test_identity_and_result_metadata_cannot_mutate_alignment() -> None:
    identity = identity_alignment()
    assert identity.method == "identity"
    assert identity.metadata == {
        "method": "identity",
        "reason": "reference",
        "warp_matrix": np.eye(2, 3).tolist(),
    }
    identity.metadata.clear()
    assert identity.metadata["reason"] == "reference"
    with pytest.raises(ValueError, match="read-only"):
        identity.warp_matrix[0, 2] = 4
    with pytest.raises(FrozenInstanceError):
        identity.reason = "changed"  # pyright: ignore[reportAttributeAccessIssue]
    matrix = np.eye(2, 3)
    result = AlignmentResult("affine_sift", matrix, diagnostics=_diagnostics())
    matrix[0, 2] = 17
    np.testing.assert_array_equal(result.warp_matrix, np.eye(2, 3))
    assert result.metadata["warp_matrix"] == np.eye(2, 3).tolist()


@pytest.mark.parametrize("mode", ["never", "auto", "always"])
@pytest.mark.parametrize("declared", [None, True, False])
def test_alignment_policy(mode: str, declared: bool | None) -> None:
    data = _image()
    with patch(
        "virtual_staining.data.alignment.registration._estimate_affine",
        return_value=(np.eye(2, 3), _diagnostics()),
    ) as estimate:
        if mode == "never" and declared is False:
            with pytest.raises(AlignmentError, match="contradicts already_aligned=false"):
                resolve_alignment(data, data, AlignmentConfig(mode=mode), already_aligned=declared)
            estimate.assert_not_called()
            return
        result = resolve_alignment(data, data, AlignmentConfig(mode=mode), already_aligned=declared)
    if declared is True or mode == "never":
        estimate.assert_not_called()
        assert result.method == "identity"
        assert result.reason == ("declared_aligned" if declared else "policy_never")
    else:
        estimate.assert_called_once()
        assert result.method == "affine_sift"


@pytest.mark.parametrize("validate", [True, False])
@pytest.mark.parametrize("declared", [True, None])
@pytest.mark.parametrize("mismatch", ["geometry", "mpp_x", "mpp_y"])
def test_identity_validation(validate: bool, declared: bool | None, mismatch: str) -> None:
    reference = replace(_image(), mpp=(0.25, 0.25))
    moving = replace(
        reference,
        full_shape=(40, 120) if mismatch == "geometry" else reference.full_shape,
        mpp=(0.3 if mismatch == "mpp_x" else 0.25, 0.3 if mismatch == "mpp_y" else 0.25),
    )
    policy = AlignmentConfig(mode="never", validate_declared=validate)
    if validate:
        with pytest.raises(AlignmentError, match=mismatch):
            resolve_alignment(reference, moving, policy, already_aligned=declared)
    else:
        assert (
            resolve_alignment(reference, moving, policy, already_aligned=declared).method
            == "identity"
        )


@pytest.mark.parametrize("mpp", [(None, None), (0.252, 0.249), (None, 0.25)])
def test_identity_validation_accepts_missing_or_close_pixel_sizes(mpp) -> None:
    reference = replace(_image(), mpp=(0.25, 0.25))
    moving = replace(_image(preview_shape=(40, 60)), mpp=mpp)
    assert (
        resolve_alignment(reference, moving, AlignmentConfig(), already_aligned=True).method
        == "identity"
    )


@pytest.mark.parametrize("missing", ["reference", "moving"])
def test_registration_requires_both_masks(missing: str) -> None:
    data = _image()
    maskless = replace(data, mask=None)
    with pytest.raises(AlignmentError, match="requires masks"):
        resolve_alignment(
            maskless if missing == "reference" else data,
            maskless if missing == "moving" else data,
            AlignmentConfig(on_failure="skip_set"),
        )


@pytest.mark.parametrize("shape", [(80, 120), (40, 60), (20, 30), (7, 10)])
def test_registration_accepts_whole_image_masks_at_different_resolutions(shape) -> None:
    data = replace(_image(), mask=np.full(shape, 255, dtype=np.uint8))
    with patch(
        "virtual_staining.data.alignment.registration._estimate_affine",
        return_value=(np.eye(2, 3), _diagnostics()),
    ) as estimate:
        resolve_alignment(data, data, AlignmentConfig())
    reference, moving, reference_mask, moving_mask = estimate.call_args.args
    assert reference.shape == moving.shape == (10, 15, 3)
    assert reference_mask.shape == moving_mask.shape == (10, 15)
    assert np.all(reference_mask == 255) and np.all(moving_mask == 255)


@pytest.mark.parametrize(
    "mask",
    [
        np.zeros((20, 30, 1), dtype=np.uint8),
        np.zeros((0, 30), dtype=np.uint8),
        np.zeros((10, 30), dtype=np.uint8),
        np.zeros((160, 240), dtype=np.uint8),
        np.zeros((20, 30), dtype=np.float32),
    ],
)
def test_invalid_mask_geometry_is_rejected_before_sift_and_warp(mask: np.ndarray) -> None:
    data = replace(_image(), mask=mask)
    with (
        patch("virtual_staining.data.alignment.registration.cv2.SIFT_create") as sift,
        pytest.raises(AlignmentError, match="mask geometry"),
    ):
        resolve_alignment(_image(), data, AlignmentConfig())
    sift.assert_not_called()
    with pytest.raises(AlignmentError, match="mask geometry"):
        warp_aligned_mask_patch(mask, np.eye(2, 3), data.full_shape, x=0, y=0, output_size=(8, 8))


@pytest.mark.parametrize("full_shape", [(0, 120), (80, 0), (-1, 120)])
def test_registration_rejects_nonpositive_image_geometry(full_shape) -> None:
    with pytest.raises(AlignmentError, match="positive height and width"):
        resolve_alignment(_image(), _image(full_shape), AlignmentConfig())


def test_preview_to_full_conversion_handles_different_scales_and_updates_metadata() -> None:
    reference = _image((160, 240), (40, 60))
    moving = _image((96, 192), (48, 96))
    estimated = np.array([[0, -0.5, 3], [0.5, 0, -2]], dtype=np.float64)
    diagnostics = _diagnostics()
    with patch(
        "virtual_staining.data.alignment.registration._estimate_affine",
        return_value=(estimated, diagnostics),
    ):
        result = resolve_alignment(reference, moving, AlignmentConfig())
    expected = np.array([[0, -1, 24], [1, 0, -16]])
    np.testing.assert_allclose(result.warp_matrix, expected)
    np.testing.assert_allclose(result.warp_matrix @ [12, 6, 1], [18, -4])
    np.testing.assert_array_equal(estimated, [[0, -0.5, 3], [0.5, 0, -2]])
    assert diagnostics.translation_x == diagnostics.translation_y == 0
    assert result.metadata["warp_matrix"] == expected.tolist()
    assert result.metadata["translation_x"] == 24
    assert result.metadata["translation_y"] == -16
    assert result.metadata["scale_x"] == result.metadata["scale_y"] == 1.0
    assert result.metadata["rotation_deg"] == 90


def test_preview_rounding_uses_separate_x_and_y_scales() -> None:
    reference = _image((83, 123), (20, 30))
    moving = _image((85, 125), (20, 30))
    with patch(
        "virtual_staining.data.alignment.registration._estimate_affine",
        return_value=(np.array([[1.0, 0.0, 3.0], [0.0, 1.0, -2.0]]), _diagnostics()),
    ):
        result = resolve_alignment(reference, moving, AlignmentConfig())
    np.testing.assert_allclose(result.warp_matrix, [[123 / 125, 0, 24.6], [0, 83 / 85, -16.6]])


@pytest.mark.parametrize(
    "matrix",
    [np.zeros((2, 3)), np.full((2, 3), np.nan), np.full((2, 3), np.inf), np.eye(3)],
)
def test_invalid_affines_raise_alignment_errors_before_warp_or_reader_io(matrix) -> None:
    image = _textured_image()
    with (
        patch(
            "virtual_staining.data.alignment.registration.cv2.estimateAffinePartial2D",
            return_value=(matrix, np.ones((100, 1), dtype=np.uint8)),
        ),
        pytest.raises(AlignmentError, match="Invalid affine transform"),
    ):
        _estimate_affine(image, image)
    read_region = Mock()
    with pytest.raises(AlignmentError, match="Invalid affine transform"):
        warp_aligned_patch(read_region, matrix, x=0, y=0, output_size=(8, 8))
    read_region.assert_not_called()
    with pytest.raises(AlignmentError, match="Invalid affine transform"):
        warp_aligned_mask_patch(
            np.zeros((8, 8), dtype=np.uint8), matrix, (8, 8), x=0, y=0, output_size=(8, 8)
        )


def test_missing_descriptors_raise_alignment_error() -> None:
    with (
        patch(
            "virtual_staining.data.alignment.registration.cv2.SIFT_create",
            return_value=_FakeSift(_keypoints(4), None, _keypoints(4), _descriptors(4)),
        ),
        pytest.raises(AlignmentError, match="Not enough features"),
    ):
        _estimate_affine(_textured_image(), _textured_image())


def test_mask_iou_uses_moving_to_reference_transform_and_empty_union() -> None:
    reference = np.zeros((16, 24), dtype=np.uint8)
    reference[4:12, 6:14] = 255
    moving = np.zeros_like(reference)
    moving[2:10, 9:17] = 255
    matrix = np.array([[1.0, 0.0, -3.0], [0.0, 1.0, 2.0]])
    assert _aligned_mask_iou(reference, moving, matrix, (24, 16)) == 1.0
    assert _aligned_mask_iou(reference, moving, np.eye(2, 3), (24, 16)) == pytest.approx(30 / 98)
    assert _aligned_mask_iou(reference * 0, moving * 0, matrix, (24, 16)) is None


def test_real_affine_estimation_preserves_rotation_scale_and_translation() -> None:
    reference = _textured_image()
    known = cv2.getRotationMatrix2D((150, 150), 4.0, 1.03)
    known[:, 2] += (6, -4)
    moving = cv2.warpAffine(reference, cv2.invertAffineTransform(known), (300, 300))
    matrix, diagnostics = _estimate_affine(reference, moving)
    np.testing.assert_allclose(matrix[:, :2], known[:, :2], atol=0.005)
    np.testing.assert_allclose(matrix[:, 2], known[:, 2], atol=0.5)
    assert diagnostics.n_inliers >= 12


@pytest.mark.parametrize("origin", [(0, 0), (13, 17), (90, 70)])
@pytest.mark.parametrize("rotation", [0.0, 7.0])
def test_reader_patch_matches_in_memory_warp(tmp_path: Path, origin, rotation: float) -> None:
    image = np.random.default_rng(5).integers(0, 255, (96, 112, 3), dtype=np.uint8)
    path = tmp_path / "moving.png"
    assert cv2.imwrite(str(path), image)
    matrix = cv2.getRotationMatrix2D((56, 48), rotation, 0.97)
    matrix[:, 2] += (7.4, -5.2)
    before = matrix.copy()
    reader = PillowRegionImageReader(path)
    read_region = Mock(wraps=reader.read_region)
    try:
        actual = warp_aligned_patch(
            read_region, matrix, x=origin[0], y=origin[1], output_size=(32, 24)
        )
        expected = warp_aligned_patch(image, matrix, x=origin[0], y=origin[1], output_size=(32, 24))
    finally:
        reader.close()
    np.testing.assert_allclose(actual, expected, atol=8)
    np.testing.assert_array_equal(matrix, before)
    read_region.assert_called_once()
    _, _, width, height = read_region.call_args.args
    assert width < 112 and height < 96


def test_reader_errors_propagate_unchanged() -> None:
    failure = OSError("region unavailable")
    with pytest.raises(OSError) as caught:
        warp_aligned_patch(Mock(side_effect=failure), np.eye(2, 3), x=0, y=0, output_size=(8, 8))
    assert caught.value is failure


def test_mask_patch_scales_both_affine_columns_before_reference_crop() -> None:
    mask = np.zeros((8, 12), dtype=np.uint8)
    mask[2:6, 3:9] = 255
    matrix = np.array([[0.0, -1.0, 32.0], [1.0, 0.0, 2.0]])
    actual = warp_aligned_mask_patch(mask, matrix, (32, 48), x=6, y=8, output_size=(20, 24))
    # Direct known mask->patch matrix: four full pixels per mask pixel.
    expected = cv2.warpAffine(
        mask, np.array([[0.0, -4.0, 26.0], [4.0, 0.0, -6.0]]), (20, 24), flags=cv2.INTER_NEAREST
    )
    np.testing.assert_array_equal(actual, expected)
    assert set(np.unique(actual)) == {0, 255}
