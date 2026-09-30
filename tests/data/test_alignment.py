from __future__ import annotations

import json
from dataclasses import replace
from typing import Any, Literal
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from virtual_staining.data.alignment import (
    AlignmentError,
    AlignmentImage,
    AlignmentResult,
    AlignmentTransform,
    GridGeometry,
    ImageGeometry,
    QCPolicy,
    RegistrationRequest,
    SpatialEvidence,
    evaluate_alignment_qc,
    identity_alignment,
    resolve_alignment,
    warp_aligned_mask_patch,
    warp_aligned_patch,
)
from virtual_staining.data.alignment.models import Family
from virtual_staining.data.alignment.registration import _ratio_test_matches
from virtual_staining.utils.image_io import (
    convert_to_pyramidal_tiff,
    open_image_reader,
)


@pytest.fixture
def frames():
    return ImageGeometry("moving", (40, 60), (0.25, 0.5)), ImageGeometry(
        "reference", (40, 60), (None, 0.7)
    )


def transform(frames: tuple[ImageGeometry, ImageGeometry], matrix=None, family: Family = "affine"):
    return AlignmentTransform(*frames, family, np.eye(3) if matrix is None else np.array(matrix))


def image(geometry, values=None, grid=None):
    grid = grid or GridGeometry(geometry.shape, np.eye(3))
    return AlignmentImage(
        np.zeros(grid.shape, np.uint8) if values is None else values, geometry, grid
    )


def evidence(
    geometry,
    values,
    kind: Literal["tissue_support", "observation_validity"] = "tissue_support",
    grid=None,
):
    return SpatialEvidence(geometry, grid or GridGeometry(values.shape, np.eye(3)), values, kind)


@pytest.mark.parametrize(
    "matrix,family,point,expected",
    [
        (np.eye(3), "identity", [3, 4], [3, 4]),
        ([[1, 0, 5], [0, 1, -2], [0, 0, 1]], "similarity", [3, 4], [8, 2]),
        ([[0, -1, 0], [1, 0, 0], [0, 0, 1]], "similarity", [3, 4], [-4, 3]),
        ([[2, 0, 0], [0, 2, 0], [0, 0, 1]], "similarity", [3, 4], [6, 8]),
        ([[2, -2, 7], [2, 2, -3], [0, 0, 1]], "similarity", [3, 4], [5, 11]),
        ([[1, 0.5, 1], [0, 1, -2], [0, 0, 1]], "affine", [3, 4], [6, 2]),
    ],
)
def test_golden_forward_inverse(frames, matrix, family, point, expected):
    candidate = transform(frames, matrix, family)
    np.testing.assert_allclose(candidate.map_points(np.array([point])), [expected], atol=1e-12)
    np.testing.assert_allclose(
        candidate.inverse().map_points(np.array([expected])), [point], atol=1e-12
    )
    assert candidate.moving.name == "moving" and candidate.reference.name == "reference"
    assert candidate.matrix.dtype == np.float64
    with pytest.raises(ValueError):
        candidate.matrix[0, 0] = 9
    with pytest.raises(ValueError):
        candidate.matrix.setflags(write=True)


def test_composition_order_and_frame_check(frames):
    first = transform(frames, [[1, 0, 3], [0, 1, 2], [0, 0, 1]], "similarity")
    third = ImageGeometry("third", (40, 60))
    second = AlignmentTransform(frames[1], third, "similarity", np.diag([2.0, 2.0, 1.0]))
    combined = first.then(second)
    np.testing.assert_array_equal(combined.map_points(np.array([[1, 1]])), [[8, 6]])
    np.testing.assert_allclose(first.then(first.inverse()).matrix, np.eye(3), atol=1e-12)
    with pytest.raises(AlignmentError, match="intermediate"):
        second.then(first)


def test_estimation_grids_offsets_anisotropic_scale_serialization(frames):
    moving_grid = GridGeometry.resized_crop((4, 5), origin=(10, 20), scale=(2, 4))
    reference_grid = GridGeometry.resized_crop((7, 8), origin=(30, 40), scale=(3, 5))
    estimated = np.array([[1, 0, 2], [0, 1, -1], [0, 0, 1.0]])
    candidate = AlignmentTransform.from_estimated(
        frames[0], frames[1], estimated, moving_grid, reference_grid
    )
    # Moving grid (1,2) -> native (12.5,29.5); reference grid (3,1) -> native (40,47).
    np.testing.assert_allclose(
        candidate.map_points(np.array([[12.5, 29.5]])), [[40, 47]], atol=1e-12
    )
    assert candidate.family == "affine"
    restored = AlignmentTransform.from_dict(json.loads(json.dumps(candidate.to_dict())))
    np.testing.assert_array_equal(restored.matrix, candidate.matrix)
    assert restored.moving.mpp == (0.25, 0.5) and restored.reference.mpp == (None, 0.7)
    assert restored.moving_grid is not None
    np.testing.assert_array_equal(restored.moving_grid.grid_to_level0, moving_grid.grid_to_level0)


@pytest.mark.parametrize(
    "matrix",
    [
        np.eye(2, 3),
        np.eye(4),
        np.zeros((3, 3)),
        np.diag([0, 1, 1]),
        np.full((3, 3), np.nan),
        np.full((3, 3), np.inf),
        [[1, 0, 0], [0, 1, 0], [0.1, 0, 1]],
        np.diag([-1, 1, 1]),
    ],
)
def test_invalid_geometry(frames, matrix):
    with pytest.raises(AlignmentError):
        transform(frames, matrix)


@pytest.mark.parametrize(
    "change",
    [
        {"family": "deformable"},
        {"displacement": []},
        {"format": "old"},
        {"direction": "reference_to_moving"},
        {"matrix": [[1, 0, 0], [0, 1, 0]]},
    ],
)
def test_reject_superseded_or_unsupported_metadata(frames, change):
    data = transform(frames).to_dict() | change
    with pytest.raises(AlignmentError):
        AlignmentTransform.from_dict(data)
    with pytest.raises(AlignmentError):
        AlignmentResult.from_dict({"method": "identity", "warp_matrix": np.eye(2, 3).tolist()})


@pytest.mark.parametrize(
    "shape,matrix",
    [
        ((0, 3), np.eye(3)),
        ((2, 3), np.diag([0, 1, 1])),
        ((2, 3), [[1, 0.1, 0], [0, 1, 0], [0, 0, 1]]),
        ((2, 3), np.diag([-1, -1, 1])),
    ],
)
def test_invalid_grids(shape, matrix):
    with pytest.raises(AlignmentError):
        GridGeometry(shape, np.array(matrix))


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(relationship="same_coordinate_frame", family="affine"),
        dict(relationship="non_corresponding", family="identity"),
        dict(relationship="unknown", family="identity"),
        dict(relationship="serial_section", family="affine", purpose="dense_correspondence"),
        dict(
            relationship="same_section_restained", family="affine", allowed_families=("identity",)
        ),
        dict(relationship="same_coordinate_frame", family="identity", allowed_families=("affine",)),
        dict(relationship="same_section_restained", family="affine", existing_alignment="identity"),
        dict(
            relationship="same_coordinate_frame", family="identity", existing_alignment="unaligned"
        ),
        dict(relationship="same_section_restained", family="deformable"),
    ],
)
def test_blocked_permissions(kwargs):
    with pytest.raises(AlignmentError):
        RegistrationRequest(**kwargs)


@pytest.mark.parametrize(
    "relationship", ["same_section_different_modality", "same_section_restained", "serial_section"]
)
@pytest.mark.parametrize("family", ["identity", "similarity", "affine"])
def test_default_permissions(relationship, family):
    assert RegistrationRequest(relationship, family, purpose="spatial_association").family == family


def test_qc_states_identity_missing_and_backend_separation(frames, monkeypatch):
    request = RegistrationRequest("same_section_restained", "identity")
    result = identity_alignment(frames[1], frames[0], request)
    assert result.backend_status == "succeeded" and result.qc is None
    assert result.candidate is not None
    empty = evaluate_alignment_qc(result.candidate, request, QCPolicy())
    assert empty.status == "insufficient_evidence" and "qc_thresholds" in empty.missing_evidence
    policy = QCPolicy({"overlap": (0.9, None), "landmark_rms": (None, 0.1)})
    missing = evaluate_alignment_qc(result.candidate, request, policy)
    assert missing.status == "insufficient_evidence" and missing.metrics["landmark_rms"] is None
    points = np.array([[2.0, 3.0], [10.0, 15.0]])
    accepted = evaluate_alignment_qc(
        result.candidate, request, policy, moving_landmarks=points, reference_landmarks=points
    )
    assert accepted.status == "accepted" and "support_iou" in accepted.missing_evidence
    rejected = evaluate_alignment_qc(
        result.candidate, request, policy, moving_landmarks=points, reference_landmarks=points + 1
    )
    assert rejected.status == "rejected" and "threshold_failed:landmark_rms" in rejected.reasons
    persisted = replace(result, qc=accepted)
    restored = AlignmentResult.from_dict(json.loads(json.dumps(persisted.metadata)))
    assert restored.qc == accepted and restored.request == request
    monkeypatch.setattr(
        "virtual_staining.data.alignment.registration._estimate_affine",
        Mock(side_effect=OSError("backend down")),
    )
    failure = resolve_alignment(
        image(frames[1]), image(frames[0]), replace(request, family="affine")
    )
    assert failure.backend_status == "failed" and failure.qc is None and failure.candidate is None
    assert AlignmentResult.from_dict(failure.metadata).reason == "backend down"


def test_qc_geometry_permissions_overlap_identity_comparison(frames):
    candidate = transform(frames, [[4, 0, 1000], [0, 4, 0], [0, 0, 1]])
    request = RegistrationRequest("same_section_restained", "identity")
    qc = evaluate_alignment_qc(
        candidate, request, QCPolicy({"max_scale": (None, 2), "overlap": (0.5, None)})
    )
    assert qc.status == "rejected"
    assert qc.reasons == (
        "candidate_family_disallowed",
        "threshold_failed:max_scale",
        "threshold_failed:overlap",
    )
    request = replace(request, family="affine", purpose="dense_correspondence")
    qc = evaluate_alignment_qc(transform(frames), request, QCPolicy({"overlap": (0.9, None)}))
    assert qc.status == "insufficient_evidence" and "correspondence_evidence" in qc.missing_evidence
    candidate = transform(frames, [[1, 0, 2], [0, 1, -1], [0, 0, 1]])
    points = np.array([[3, 4], [5, 6]])
    qc = evaluate_alignment_qc(
        candidate,
        replace(request, correspondence_evidence=("held-out pairs",)),
        QCPolicy({"landmark_improvement": (2, None)}),
        moving_landmarks=points,
        reference_landmarks=points + [2, -1],
    )
    assert qc.status == "accepted"
    assert qc.metrics["landmark_improvement"] == pytest.approx(np.sqrt(5))


def test_support_validity_disagreement_empty_and_offset_grid(frames):
    grid = GridGeometry.resized_crop((4, 5), origin=(2, 3), scale=(2, 3))
    ones = np.ones(grid.shape, bool)
    moving = evidence(frames[0], ones, grid=grid)
    reference = evidence(frames[1], ones, grid=grid)
    request = RegistrationRequest("same_section_restained", "identity")
    policy = QCPolicy({"support_iou": (0.9, None), "observation_valid_fraction": (0.9, None)})
    qc = evaluate_alignment_qc(
        transform(frames, family="identity"),
        request,
        policy,
        moving_support=moving,
        reference_support=reference,
        moving_validity=replace(moving, kind="observation_validity", values=~ones),
        reference_validity=replace(reference, kind="observation_validity"),
    )
    assert qc.metrics["support_iou"] == 1 and qc.metrics["observation_valid_fraction"] == 0
    assert qc.status == "rejected"
    qc = evaluate_alignment_qc(
        transform(frames, family="identity"),
        request,
        policy,
        moving_support=replace(moving, values=~ones),
        reference_support=reference,
    )
    assert qc.status == "rejected" and qc.metrics["support_iou"] == 0
    qc = evaluate_alignment_qc(
        transform(frames, family="identity"),
        request,
        policy,
        moving_support=replace(moving, values=~ones),
        reference_support=replace(reference, values=~ones),
    )
    assert qc.status == "insufficient_evidence" and qc.metrics["support_iou"] is None


def test_linear_interpolation_validity_and_labels(frames):
    source = np.tile(np.arange(60, dtype=np.float64), (40, 1))
    candidate = transform(frames, [[1, 0, -0.25], [0, 1, 0], [0, 0, 1]])
    validity = np.ones((40, 60), bool)
    validity[2, 3] = False
    kwargs: dict[str, Any] = dict(x=2, y=2, output_size=(2, 1), max_source_pixels=4)
    patch = warp_aligned_patch(
        source,
        candidate,
        observation_validity=evidence(frames[0], validity, "observation_validity"),
        **kwargs,
    )
    np.testing.assert_array_equal(patch.image, [[2.25, 3.25]])
    assert patch.observation_known is not None and patch.observation_validity is not None
    assert patch.geometric_validity.all() and patch.observation_known.all()
    assert not patch.observation_validity.any()
    assert patch.tissue_support is None
    labels = (source % 3).astype(np.uint8)
    patch = warp_aligned_patch(labels, candidate, interpolation="nearest", **kwargs)
    np.testing.assert_array_equal(patch.image, [[2, 0]])
    assert patch.observation_validity is None and patch.observation_known is None
    for interpolation in ("nearest", "linear"):
        patch = warp_aligned_patch(
            source,
            transform(frames),
            x=-1,
            y=0,
            output_size=(3, 1),
            max_source_pixels=4,
            interpolation=interpolation,
        )
        np.testing.assert_array_equal(patch.geometric_validity, [[False, True, True]])


def test_offset_anisotropic_maps_unknown_and_mask_polarity(frames):
    grid = GridGeometry.resized_crop((2, 3), origin=(10, 6), scale=(2, 4))
    support = evidence(frames[0], np.array([[True, False, True], [False, True, False]]), grid=grid)
    validity = replace(support, kind="observation_validity")
    source = np.zeros(frames[0].shape, np.uint8)
    patch = warp_aligned_patch(
        source,
        transform(frames),
        x=10,
        y=7,
        output_size=(6, 2),
        max_source_pixels=20,
        tissue_support=support,
        observation_validity=validity,
    )
    assert patch.tissue_support is not None
    np.testing.assert_array_equal(patch.tissue_support[0], [True, True, False, False, True, True])
    assert patch.observation_known is not None
    assert not patch.observation_known.all()
    outside = warp_aligned_patch(
        source,
        transform(frames),
        x=0,
        y=0,
        output_size=(2, 2),
        max_source_pixels=4,
        observation_validity=validity,
    )
    assert outside.observation_known is not None
    assert not outside.observation_known.any()
    labels = support.values.astype(np.uint8) * 7
    mask = warp_aligned_mask_patch(
        labels, transform(frames), grid, x=10, y=7, output_size=(6, 2), max_source_pixels=4
    )
    np.testing.assert_array_equal(mask[0], [7, 7, 0, 0, 7, 7])


@pytest.mark.parametrize("backend", ["pillow", "openslide"])
@pytest.mark.parametrize("interpolation", ["linear", "nearest"])
def test_affine_reader_spy_bounded_subdivision_matches_array(
    tmp_path, frames, backend, interpolation
):
    frames = tuple(replace(frame, shape=(512, 512)) for frame in frames)
    source = np.random.default_rng(1).integers(0, 255, (*frames[0].shape, 3), dtype=np.uint8)
    path = tmp_path / "source.png"
    assert cv2.imwrite(str(path), source)
    if backend == "openslide":
        tiff = tmp_path / "source.tif"
        convert_to_pyramidal_tiff(path, tiff)
        path = tiff
    reader = open_image_reader(path, backend=backend)
    reader.read_full = Mock(side_effect=AssertionError("whole-slide read forbidden"))
    reader.read_preview = Mock(side_effect=AssertionError("preview read forbidden"))
    read = Mock(wraps=reader.read_region)
    candidate = transform(frames, [[0.7, -0.2, 5.3], [0.3, 1.1, -4.2], [0, 0, 1]])
    kwargs: dict[str, Any] = dict(x=-3, y=2, output_size=(40, 30), interpolation=interpolation)
    try:
        direct = warp_aligned_patch(source, candidate, max_source_pixels=10000, **kwargs)
        tiled = warp_aligned_patch(read, candidate, max_source_pixels=40, **kwargs)
    finally:
        reader.close()
    # Identical float64 inverse coordinates; one integer intensity unit allows rounding at ties.
    np.testing.assert_allclose(tiled.image, direct.image, atol=1, rtol=0)
    np.testing.assert_array_equal(tiled.geometric_validity, direct.geometric_validity)
    assert read.call_count > 1
    for call in read.call_args_list:
        x, y, w, h = call.args
        assert w * h <= 40 and x >= 0 and y >= 0 and x + w <= 512 and y + h <= 512
    reader.read_full.assert_not_called()
    reader.read_preview.assert_not_called()


def test_budget_and_reader_failure(frames):
    reader = Mock(side_effect=OSError("unavailable"))
    with pytest.raises(AlignmentError):
        warp_aligned_patch(
            reader, transform(frames), x=0, y=0, output_size=(2, 2), max_source_pixels=0
        )
    reader.assert_not_called()
    with pytest.raises(OSError, match="unavailable"):
        warp_aligned_patch(
            reader, transform(frames), x=0, y=0, output_size=(2, 2), max_source_pixels=4
        )


def test_real_sift_without_support_is_only_a_candidate(frames):
    rng = np.random.default_rng(9)
    pixels = rng.integers(10, 200, (200, 200), dtype=np.uint8)
    moving, reference = (replace(frame, shape=(200, 200)) for frame in frames)
    request = RegistrationRequest("same_section_different_modality", "similarity")
    result = resolve_alignment(image(reference, pixels), image(moving, pixels), request)
    assert result.backend_status == "succeeded" and result.qc is None
    assert result.candidate is not None
    np.testing.assert_allclose(result.candidate.matrix, np.eye(3), atol=0.01)
    assert result.diagnostics["n_inliers"] is not None
    assert result.diagnostics["n_inliers"] > 0
    # No features is a backend failure, never a rejected/accepted QC registration.
    result = resolve_alignment(image(reference), image(moving), request)
    assert result.backend_status == "failed" and result.qc is None


def test_ratio_test():
    def match(distance):
        return cv2.DMatch(_queryIdx=0, _trainIdx=0, _distance=distance)

    assert len(_ratio_test_matches([[match(1), match(3)], [match(3)], [match(2), match(2)]])) == 1


def test_estimator_grid_composition_and_backend_score_is_not_qc(frames, monkeypatch):
    moving_grid = GridGeometry.resized_crop((5, 6), origin=(2, 4), scale=(2, 3))
    reference_grid = GridGeometry.resized_crop((5, 6), origin=(8, 9), scale=(4, 5))
    estimated = np.array([[1.0, 0.2, 2], [0, 1, -1], [0, 0, 1]])
    estimate = Mock(return_value=(estimated, {"n_inliers": 0, "inlier_ratio": 0.0}))
    monkeypatch.setattr("virtual_staining.data.alignment.registration._estimate_affine", estimate)
    request = RegistrationRequest("same_section_restained", "affine")
    result = resolve_alignment(
        image(frames[1], grid=reference_grid), image(frames[0], grid=moving_grid), request
    )
    assert result.backend_status == "succeeded" and result.qc is None
    assert result.candidate is not None
    np.testing.assert_allclose(
        result.candidate.matrix,
        reference_grid.grid_to_level0 @ estimated @ np.linalg.inv(moving_grid.grid_to_level0),
        atol=1e-12,
    )
    insufficient = evaluate_alignment_qc(
        result.candidate, request, QCPolicy({"landmark_rms": (None, 1)})
    )
    assert insufficient.status == "insufficient_evidence"


@pytest.mark.parametrize(
    "matrix,family",
    [
        ([[1, 0.1, 0], [0, 1, 0], [0, 0, 1]], "similarity"),
        (np.diag([2.0, 2.0, 1]), "identity"),
        (np.diag([1e-8, 2e-8, 1]), "similarity"),
    ],
)
def test_family_must_match_geometry(frames, matrix, family):
    with pytest.raises(AlignmentError):
        transform(frames, matrix, family)


def test_identity_does_not_assume_tissue_or_validity(frames):
    request = RegistrationRequest("same_coordinate_frame", "identity")
    qc = evaluate_alignment_qc(
        transform(frames, family="identity"),
        request,
        QCPolicy({"support_iou": (0.5, None), "observation_valid_fraction": (0.9, None)}),
    )
    assert qc.status == "insufficient_evidence"
    assert qc.metrics["support_iou"] is None and qc.metrics["observation_valid_fraction"] is None
    assert qc.reasons == ("missing:observation_valid_fraction", "missing:support_iou")


@pytest.mark.parametrize(
    "thresholds",
    [
        {"backend_inlier_ratio": (0.9, None)},
        {"overlap": (None, None)},
        {"overlap": (1, 0)},
        {"landmark_rms": (None, float("inf"))},
    ],
)
def test_invalid_qc_policy(thresholds):
    with pytest.raises(AlignmentError):
        QCPolicy(thresholds)


def test_evidence_rejects_wrong_asset_kind_and_soft_values(frames):
    values = np.ones((2, 2), bool)
    with pytest.raises(AlignmentError):
        evidence(frames[0], values.astype(float))
    with pytest.raises(AlignmentError):
        replace(image(frames[0]), tissue_support=evidence(frames[1], values))
    with pytest.raises(AlignmentError):
        warp_aligned_patch(
            np.zeros(frames[0].shape),
            transform(frames),
            x=0,
            y=0,
            output_size=(2, 2),
            max_source_pixels=4,
            observation_validity=evidence(frames[0], values),
        )


def test_continuous_border_and_minimum_footprint(frames):
    candidate = transform(frames, [[1, 0, 0.25], [0, 1, 0.25], [0, 0, 1]])
    source = np.zeros(frames[0].shape, np.float64)
    patch = warp_aligned_patch(source, candidate, x=0, y=0, output_size=(1, 1), max_source_pixels=1)
    assert patch.image[0, 0] == pytest.approx(255 * (1 - 0.75**2))
    assert not patch.geometric_validity[0, 0]
    reader = Mock()
    with pytest.raises(AlignmentError, match="footprint"):
        warp_aligned_patch(reader, candidate, x=1, y=1, output_size=(1, 1), max_source_pixels=3)
    reader.assert_not_called()


def test_unknown_relationship_is_only_bounded_diagnostic(frames):
    request = RegistrationRequest("unknown", "identity", diagnostic_region=(2, 3, 10, 12))
    result = identity_alignment(frames[1], frames[0], request)
    assert result.request.purpose == "diagnostic" and result.request.correspondence_evidence == ()
    with pytest.raises(AlignmentError):
        replace(
            request, purpose="dense_correspondence", correspondence_evidence=("similar images",)
        )


def test_nearest_preserves_uint64_labels_exactly(frames):
    labels = np.full(frames[0].shape, 2**63 + 3, dtype=np.uint64)
    labels[2, 3] = 2**63 + 7
    candidate = transform(frames, [[1, 0, -0.5], [0, 1, 0], [0, 0, 1]])
    patch = warp_aligned_patch(
        labels,
        candidate,
        x=2,
        y=2,
        output_size=(2, 1),
        interpolation="nearest",
        max_source_pixels=1,
    )
    np.testing.assert_array_equal(patch.image, labels[2:3, 3:5])


def test_diagnostic_scope_is_bounded_to_reference(frames):
    request = RegistrationRequest("unknown", "identity", diagnostic_region=(59, 0, 2, 2))
    with pytest.raises(AlignmentError, match="reference geometry"):
        identity_alignment(frames[1], frames[0], request)
    request = replace(request, family="similarity", diagnostic_region=(0, 0, 5, 5))
    candidate = transform(frames, [[1, 0, 10], [0, 1, 0], [0, 0, 1]], "similarity")
    qc = evaluate_alignment_qc(candidate, request, QCPolicy({"overlap": (0.9, None)}))
    assert qc.status == "rejected" and qc.metrics["overlap"] == 0


@pytest.mark.parametrize("family", ["similarity", "affine"])
def test_estimation_preserves_explicit_family_restriction(frames, monkeypatch, family):
    monkeypatch.setattr(
        "virtual_staining.data.alignment.registration._estimate_affine",
        Mock(return_value=(np.eye(3), {"n_inliers": 100})),
    )
    request = RegistrationRequest("same_section_restained", family, allowed_families=(family,))
    result = resolve_alignment(image(frames[1]), image(frames[0]), request)
    assert result.candidate is not None and result.candidate.family == family
    qc = evaluate_alignment_qc(result.candidate, request, QCPolicy({"overlap": (0.9, None)}))
    assert qc.status == "accepted"


def test_sift_similarity_fits_native_coordinates_on_anisotropic_grids(frames, monkeypatch):
    import virtual_staining.data.alignment.registration as registration

    points = [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0), (7.0, 8.0)]
    keys = [cv2.KeyPoint(x, y, 1) for x, y in points]
    monkeypatch.setattr(
        registration.cv2,
        "SIFT_create",
        Mock(
            return_value=Mock(
                detectAndCompute=Mock(return_value=(keys, np.ones((4, 128), np.float32)))
            )
        ),
    )
    matches = [
        [
            cv2.DMatch(_queryIdx=i, _trainIdx=i, _distance=1),
            cv2.DMatch(_queryIdx=i, _trainIdx=i, _distance=3),
        ]
        for i in range(4)
    ]
    monkeypatch.setattr(
        registration.cv2, "BFMatcher", Mock(return_value=Mock(knnMatch=Mock(return_value=matches)))
    )
    native = np.array([[0.0, -2.0, 5.0], [2.0, 0.0, 7.0], [0.0, 0.0, 1.0]])
    fit = Mock(return_value=(native[:2], None))
    monkeypatch.setattr(registration.cv2, "estimateAffinePartial2D", fit)
    moving_grid = GridGeometry.resized_crop((10, 10), origin=(3, 4), scale=(2, 3))
    reference_grid = GridGeometry.resized_crop((10, 10), origin=(5, 6), scale=(4, 5))
    result = resolve_alignment(
        image(frames[1], grid=reference_grid),
        image(frames[0], grid=moving_grid),
        RegistrationRequest("same_section_restained", "similarity"),
    )
    assert result.candidate is not None and result.candidate.family == "similarity"
    assert result.diagnostics["n_inliers"] is None
    assert result.diagnostics["inlier_ratio"] is None
    np.testing.assert_allclose(result.candidate.matrix, native, atol=1e-12)
    np.testing.assert_array_equal(fit.call_args.args[0][0], [5.5, 11.0])
    np.testing.assert_array_equal(fit.call_args.args[1][0], [10.5, 18.0])
