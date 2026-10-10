from __future__ import annotations

import json
from dataclasses import asdict, replace
from typing import Any, Literal, get_args
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from virtual_staining.data.alignment import (
    AlignmentError,
    AlignmentImage,
    AlignmentResult,
    AlignmentTransform,
    FailureCategory,
    GridGeometry,
    ImageGeometry,
    QCPolicy,
    RegistrationAttempt,
    RegistrationFailure,
    RegistrationRequest,
    RegistrationResources,
    RegistrationRuntime,
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
        ((2, 3), [[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
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
    restored_failure = AlignmentResult.from_dict(failure.metadata).attempt.failure
    assert restored_failure is not None
    assert (
        restored_failure.category == "internal_error" and restored_failure.message == "backend down"
    )


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


def _full_grid_qc_counts(candidate, request, moving, reference):
    """Frozen pre-block evidence calculation; intentionally only used on small grids."""
    from virtual_staining.data.alignment.warping import _coordinates, _sample_evidence

    h, w = reference.grid.shape
    native = _coordinates(reference.grid.grid_to_level0, 0, 0, w, h)
    inverse = candidate.inverse().matrix
    points = native @ inverse[:2, :2].T + inverse[:2, 2]
    values, known = _sample_evidence(
        moving, points, conservative=reference.kind == "observation_validity"
    )
    for coords, asset in ((native, candidate.reference), (points, candidate.moving)):
        known &= (
            (coords[..., 0] >= -0.5)
            & (coords[..., 0] < asset.shape[1] - 0.5)
            & (coords[..., 1] >= -0.5)
            & (coords[..., 1] < asset.shape[0] - 0.5)
        )
    x, y, width, height = request.reference_region(candidate.reference)
    known &= (
        (native[..., 0] >= x - 0.5)
        & (native[..., 0] < x + width - 0.5)
        & (native[..., 1] >= y - 0.5)
        & (native[..., 1] < y + height - 0.5)
    )
    denominator = (
        int(np.count_nonzero(known & (values | reference.values)))
        if reference.kind == "tissue_support"
        else int(np.count_nonzero(known))
    )
    return int(np.count_nonzero(known & values & reference.values)), denominator


@pytest.mark.parametrize("pattern", ["true", "false", "mixed"])
@pytest.mark.parametrize("region", [None, (1, 1, 3, 3), (10, 10, 2, 2)])
@pytest.mark.parametrize(
    "mapping",
    [
        "identity",
        "translation",
        "fractional",
        "anisotropic",
        "rotation",
        "affine",
        "empty",
        "boundary",
    ],
)
def test_qc_block_counts_and_decisions_match_full_grid(monkeypatch, pattern, region, mapping):
    from virtual_staining.data.alignment import registration

    moving, reference = ImageGeometry("moving", (12, 14)), ImageGeometry("reference", (12, 14))
    matrix = np.eye(3)
    support_grid = GridGeometry((7, 9), np.eye(3))
    moving_grid = GridGeometry((6, 8), np.eye(3))
    if mapping in {"translation", "fractional", "empty"}:
        matrix[:2, 2] = {"translation": (7, 5), "fractional": (0.5, -0.25), "empty": (30, 0)}[
            mapping
        ]
    elif mapping == "anisotropic":
        support_grid = GridGeometry.resized_crop((7, 9), origin=(1, -1), scale=(1.25, 1.5))
        moving_grid = GridGeometry.resized_crop((6, 8), origin=(-1, 2), scale=(1.5, 0.75))
    elif mapping == "rotation":
        matrix = np.array([[0, -1, 10], [1, 0, 0], [0, 0, 1]])
    elif mapping == "affine":
        matrix = np.array([[1, 0.125, 0.5], [-0.25, 1, 1], [0, 0, 1]])
    elif mapping == "boundary":
        support_grid = GridGeometry((7, 9), np.array([[1, 0, -0.5], [0, 1, -0.5], [0, 0, 1]]))
    candidate = AlignmentTransform(moving, reference, "affine", matrix)
    request = RegistrationRequest("same_section_restained", "affine", diagnostic_region=region)
    # Validity has its own reference grid; it must not inherit support's block slices.
    validity_grid = GridGeometry.resized_crop((5, 7), origin=(0, 1), scale=(0.75, 1.25))
    kwargs = {}
    expected_counts = []
    for kind, grid in (("support", support_grid), ("validity", validity_grid)):
        for side, asset, current_grid in (
            ("reference", reference, grid),
            ("moving", moving, moving_grid),
        ):
            rows, columns = np.indices(current_grid.shape)
            values = np.full(current_grid.shape, pattern != "false", dtype=bool)
            if pattern == "mixed":
                values = (rows + columns + (side == "moving")) % 3 != 0
            kwargs[f"{side}_{kind}"] = evidence(
                asset,
                values,
                "tissue_support" if kind == "support" else "observation_validity",
                current_grid,
            )
        expected_counts.append(
            _full_grid_qc_counts(
                candidate, request, kwargs[f"moving_{kind}"], kwargs[f"reference_{kind}"]
            )
        )
    # The geometric/landmark decision is independent of evidence partitioning.
    landmarks = np.array([[1.0, 1.0] if region is None else region[:2]], dtype=float)
    landmark_args: dict[str, Any] = dict(moving_landmarks=landmarks, reference_landmarks=landmarks)
    baseline = evaluate_alignment_qc(candidate, request, QCPolicy(), **landmark_args)
    metrics = dict(baseline.metrics)
    metric_names = ("support_iou", "observation_valid_fraction")
    for name, (numerator, denominator) in zip(metric_names, expected_counts, strict=True):
        metrics[name] = float(numerator / denominator) if denominator else None
    policy = QCPolicy({name: (1, 1) for name in metric_names})
    rejected = tuple(
        f"threshold_failed:{name}"
        for name in sorted(metric_names)
        if metrics[name] is not None and metrics[name] != 1
    )
    missing = tuple(sorted(name for name, value in metrics.items() if value is None))
    unmet = tuple(f"missing:{name}" for name in sorted(metric_names) if metrics[name] is None)
    expected = replace(
        baseline,
        metrics=metrics,
        missing_evidence=missing,
        reasons=rejected + unmet,
        status="rejected" if rejected else "insufficient_evidence" if unmet else "accepted",
    )
    original_count = np.count_nonzero
    for block_size in (1, 4, 256):
        calls = []

        def count(values, calls=calls):
            result = original_count(values)
            calls.append(int(result))
            return result

        with monkeypatch.context() as patch:
            patch.setattr(registration, "_QC_BLOCK_SIZE", block_size)
            patch.setattr(registration.np, "count_nonzero", count)
            actual = evaluate_alignment_qc(candidate, request, policy, **kwargs, **landmark_args)
        assert actual == expected
        offset = 0
        for grid, (numerator, denominator) in zip(
            (support_grid, validity_grid), expected_counts, strict=True
        ):
            blocks = ((grid.shape[0] + block_size - 1) // block_size) * (
                (grid.shape[1] + block_size - 1) // block_size
            )
            assert sum(calls[offset : offset + 2 * blocks : 2]) == denominator
            assert sum(calls[offset + 1 : offset + 2 * blocks : 2]) == numerator
            offset += 2 * blocks
        assert offset == len(calls)


@pytest.mark.parametrize("shape", [(1, 1), (1, 257), (255, 17), (256, 256), (513, 519)])
@pytest.mark.parametrize("region", [None, (0, 0, 1, 1)])
def test_qc_coordinate_and_sampling_allocations_are_bounded(monkeypatch, shape, region):
    import weakref

    from virtual_staining.data.alignment import registration

    frames = ImageGeometry("moving", shape), ImageGeometry("reference", shape)
    candidate = transform(frames, family="identity")
    request = RegistrationRequest("same_section_restained", "identity", diagnostic_region=region)
    kwargs: dict[str, Any] = {
        f"{side}_{kind}": evidence(
            asset,
            np.ones(shape, bool),
            "tissue_support" if kind == "support" else "observation_validity",
        )
        for side, asset in zip(("moving", "reference"), frames, strict=True)
        for kind in ("support", "validity")
    }
    coordinates, sample = registration._coordinates, registration._sample_evidence
    origins, sampled, live = [], [], []

    def bounded_coordinates(matrix, x, y, width, height):
        assert all(item() is None for item in live)
        assert 0 < width <= 256 and 0 < height <= 256
        origins.append((x, y, width, height))
        native = coordinates(matrix, x, y, width, height)
        live.append(weakref.ref(native))
        return native

    def bounded_sample(evidence, points, *, conservative):
        assert points.shape[0] <= 256 and points.shape[1] <= 256
        sampled.append((points.shape, conservative))
        values, known = sample(evidence, points, conservative=conservative)
        live.extend(weakref.ref(item) for item in (points, values, known))
        return values, known

    monkeypatch.setattr(registration, "_coordinates", bounded_coordinates)
    monkeypatch.setattr(registration, "_sample_evidence", bounded_sample)
    qc = evaluate_alignment_qc(candidate, request, QCPolicy({"support_iou": (1, 1)}), **kwargs)
    assert qc.status == "accepted"
    assert qc.metrics["support_iou"] == qc.metrics["observation_valid_fraction"] == 1
    expected_origins = [
        (x, y, min(256, shape[1] - x), min(256, shape[0] - y))
        for y in range(0, shape[0], 256)
        for x in range(0, shape[1], 256)
    ]
    assert origins == expected_origins * 2
    assert sampled == [
        ((h, w, 2), conservative)
        for conservative in (False, True)
        for _, _, w, h in expected_origins
    ]
    assert all(item() is None for item in live)


@pytest.mark.parametrize("side", [None, "moving", "reference"])
def test_qc_missing_pair_never_allocates_blocks(frames, monkeypatch, side):
    from virtual_staining.data.alignment import registration

    monkeypatch.setattr(
        registration, "_coordinates", Mock(side_effect=AssertionError("allocation"))
    )
    kwargs: dict[str, Any] = {}
    if side is not None:
        asset = frames[0 if side == "moving" else 1]
        kwargs = {
            f"{side}_{kind}": evidence(
                asset,
                np.ones(asset.shape, bool),
                "tissue_support" if kind == "support" else "observation_validity",
            )
            for kind in ("support", "validity")
        }
    request = RegistrationRequest("same_section_restained", "identity")
    candidate = transform(frames, family="identity")
    policy = QCPolicy({"support_iou": (1, 1), "observation_valid_fraction": (1, 1)})
    assert evaluate_alignment_qc(candidate, request, policy, **kwargs) == evaluate_alignment_qc(
        candidate, request, policy
    )
    if side is not None:
        kwargs[f"{side}_support"] = replace(kwargs[f"{side}_support"], kind="observation_validity")
        with pytest.raises(AlignmentError, match="kind/asset"):
            evaluate_alignment_qc(candidate, request, policy, **kwargs)


def test_qc_sampling_failure_releases_blocks(frames, monkeypatch):
    import weakref

    from virtual_staining.data.alignment import registration

    sample = registration._sample_evidence
    live = []
    calls = 0

    def fail_second_block(evidence, points, *, conservative):
        nonlocal calls
        assert all(item() is None for item in live)
        live.append(weakref.ref(points))
        calls += 1
        if calls == 2:
            raise RuntimeError("sampling failed")
        return sample(evidence, points, conservative=conservative)

    monkeypatch.setattr(registration, "_QC_BLOCK_SIZE", 4)
    monkeypatch.setattr(registration, "_sample_evidence", fail_second_block)
    try:
        evaluate_alignment_qc(
            transform(frames),
            RegistrationRequest("same_section_restained", "affine"),
            QCPolicy(),
            moving_support=evidence(frames[0], np.ones(frames[0].shape, bool)),
            reference_support=evidence(frames[1], np.ones(frames[1].shape, bool)),
        )
    except RuntimeError as exc:
        assert str(exc) == "sampling failed"
    else:
        pytest.fail("Sampling failure was swallowed")
    assert calls == 2 and all(item() is None for item in live)


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
    frames = (frames[0], replace(frames[1], mpp=frames[0].mpp))
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


@pytest.mark.parametrize(
    "shape,mpp,axis",
    [
        ((120, 100), (0.25, 0.5), "geometry"),
        ((100, 100), (0.5, 0.5), "mpp_x"),
        ((100, 100), (0.25, 1.0), "mpp_y"),
        ((100, 100), (None, 1.0), "mpp_y"),
    ],
)
def test_shared_frame_rejects_incompatible_geometry_on_all_routes(shape, mpp, axis):
    reference = ImageGeometry("reference", (100, 100), (0.25, 0.5))
    moving = ImageGeometry("moving", shape, mpp)
    request = RegistrationRequest("same_coordinate_frame", "identity")
    candidate = AlignmentTransform(moving, reference, "identity", np.eye(3))
    with pytest.raises(AlignmentError, match=axis):
        identity_alignment(reference, moving, request)
    with pytest.raises(AlignmentError, match=axis):
        resolve_alignment(image(reference), image(moving), request)
    with pytest.raises(AlignmentError, match=axis):
        evaluate_alignment_qc(candidate, request, QCPolicy({"overlap": (0.9, None)}))
    valid = identity_alignment(reference, reference, request)
    with pytest.raises(AlignmentError, match=axis):
        replace(valid, candidate=candidate)
    metadata = valid.metadata
    metadata["candidate"] = candidate.to_dict()
    with pytest.raises(AlignmentError, match=axis):
        AlignmentResult.from_dict(metadata)


@pytest.mark.parametrize(
    "reference_mpp,moving_mpp",
    [
        ((0.25, 0.5), (0.25, 0.5)),
        ((0.25, 0.5), (0.252, 0.504)),
        ((0.25, 0.5), (None, 0.5)),
        ((None, 0.5), (0.25, 0.5)),
        ((0.25, None), (0.25, 0.5)),
        ((0.25, 0.5), (0.25, None)),
        ((None, None), (0.25, 0.5)),
        ((0.25, 0.5), (None, None)),
        ((None, None), (None, None)),
    ],
)
def test_shared_frame_accepts_compatible_and_unknown_spacing(reference_mpp, moving_mpp):
    reference = ImageGeometry("reference", (100, 100), reference_mpp)
    moving = ImageGeometry("moving", (100, 100), moving_mpp)
    request = RegistrationRequest("same_coordinate_frame", "identity")
    result = identity_alignment(reference, moving, request)
    assert result.candidate is not None and result.qc is None
    manual = AlignmentTransform(moving, reference, "identity", np.eye(3))
    for candidate in (result.candidate, manual):
        qc = evaluate_alignment_qc(candidate, request, QCPolicy({"overlap": (0.9, None)}))
        assert qc.status == "accepted"
        assert (
            qc.metrics["support_iou"] is None and qc.metrics["observation_valid_fraction"] is None
        )
        assert candidate.moving.mpp == moving_mpp and candidate.reference.mpp == reference_mpp


def test_identity_for_other_relationship_does_not_require_equal_native_frames():
    reference = ImageGeometry("reference", (100, 100), (0.25, 0.25))
    moving = ImageGeometry("moving", (120, 100), (0.5, 0.5))
    request = RegistrationRequest("same_section_restained", "identity")
    result = identity_alignment(reference, moving, request)
    assert result.candidate is not None
    assert (
        evaluate_alignment_qc(result.candidate, request, QCPolicy({"overlap": (0.9, None)})).status
        == "accepted"
    )


@pytest.mark.parametrize("category", get_args(FailureCategory))
def test_all_failure_categories_round_trip(category):
    failure = RegistrationFailure(category, "Observed failure", subcode="stable_study_subcode")
    assert RegistrationFailure.from_dict(json.loads(json.dumps(asdict(failure)))) == failure
    assert RegistrationFailure(category, "Observed failure").subcode is None


@pytest.mark.parametrize(
    "data",
    [
        {"category": "unknown", "message": "failure", "subcode": None},
        {"category": "optimizer_failed", "message": "", "subcode": None},
        {"category": "optimizer_failed", "message": None, "subcode": None},
        {"category": "optimizer_failed", "message": "failure", "subcode": ""},
        {"category": "optimizer_failed", "message": "failure", "subcode": 1},
        {"category": "optimizer_failed", "message": "failure"},
        {"category": "optimizer_failed", "message": "failure", "subcode": None, "extra": 1},
        [],
        None,
    ],
)
def test_malformed_failures_rejected(data):
    with pytest.raises(AlignmentError):
        RegistrationFailure.from_dict(data)


def test_attempt_and_result_round_trip_with_supplied_runtime_evidence(frames):
    grid = GridGeometry.resized_crop((5, 6), origin=(3, 7), scale=(2, 3))
    runtime = RegistrationRuntime(
        backend="external_backend",
        backend_version="1.2",
        model_id="model-a",
        checkpoint_id="checkpoint-a",
        checkpoint_hash="sha256:checkpoint",
        input_metadata_ref="metadata.json",
        input_fingerprint_ref="sha256:inputs",
        requested_moving_grid=grid,
        requested_reference_grid=grid,
        resolved_moving_grid=grid,
        resolved_reference_grid=grid,
        support_mode="supplied",
        device="cpu",
        precision="float64",
        determinism_mode="seeded",
        seed=17,
        scientific_parameter_hash="sha256:parameters",
        resources=RegistrationResources(
            peak_cpu_memory_bytes=1024,
            peak_gpu_memory_bytes=0,
            peak_temp_disk_bytes=2048,
            reader_count=2,
        ),
    )
    attempt = RegistrationAttempt(
        stage="optimizer",
        runtime=runtime,
        outcome="failed",
        run_id="run-a",
        case_id="case-a",
        attempt_id="attempt-a",
        started_at=100.0,
        ended_at=102.0,
        duration_seconds=2.0,
        failure=RegistrationFailure("resource_exhausted", "Allocation failed", "cpu_memory"),
        fallback_decision="use_alternative_backend",
        next_attempt_id="attempt-b",
        diagnostic_artifacts=("diagnostics/attempt-a.txt",),
    )
    encoded = json.loads(json.dumps(attempt.to_dict(), allow_nan=False))
    restored = RegistrationAttempt.from_dict(encoded)
    assert restored.to_dict() == attempt.to_dict()
    result = AlignmentResult(
        "failed",
        "external",
        RegistrationRequest("same_section_restained", "affine"),
        None,
        attempt=attempt,
    )
    assert (
        AlignmentResult.from_dict(json.loads(json.dumps(result.metadata))).metadata
        == result.metadata
    )
    identity = identity_alignment(
        frames[1], frames[0], RegistrationRequest("same_section_restained", "identity")
    )
    assert identity.candidate is not None
    qc = evaluate_alignment_qc(
        identity.candidate, identity.request, QCPolicy({"overlap": (0.9, None)})
    )
    accepted = replace(identity, qc=qc, attempt=replace(identity.attempt, qc_status=qc.status))
    assert AlignmentResult.from_dict(json.loads(json.dumps(accepted.metadata))).qc == qc
    assert identity.metadata["format"] == "virtual_staining.alignment.result/2"
    assert identity.metadata["candidate"]["format"] == "virtual_staining.alignment/1"
    # No identity, model, precision, determinism or resource evidence is fabricated.
    assert identity.attempt.attempt_id is None and identity.attempt.run_id is None
    assert identity.attempt.runtime.model_id is None and identity.attempt.runtime.resources is None
    assert identity.attempt.runtime.precision is None and identity.attempt.runtime.seed is None
    assert identity.attempt.started_at is not None and identity.attempt.ended_at is not None
    assert identity.attempt.duration_seconds is not None and identity.attempt.duration_seconds >= 0


@pytest.mark.parametrize(
    "change",
    [
        {"failure": "raw exception"},
        {"runtime": {}},
        {"stage": ""},
        {"outcome": "unknown"},
        {"outcome": []},
        {"duration_seconds": -1},
        {"duration_seconds": float("nan")},
        {"started_at": "today"},
        {"qc_status": "unknown"},
        {"qc_status": {}},
        {"diagnostic_artifacts": "not-an-array"},
        {"diagnostic_artifacts": ["x"] * 17},
        {"diagnostic_artifacts": ["x" * 2049]},
        {"diagnostic_artifacts": [None]},
        {"unknown_field": None},
    ],
)
def test_malformed_attempt_structures_rejected(change):
    attempt = RegistrationAttempt(
        stage="registration", runtime=RegistrationRuntime(backend="test"), outcome="succeeded"
    )
    with pytest.raises(AlignmentError):
        RegistrationAttempt.from_dict(attempt.to_dict() | change)


@pytest.mark.parametrize(
    "change",
    [
        {"backend": None},
        {"backend_version": 1},
        {"seed": True},
        {"resources": {"peak_cpu_memory_bytes": -1}},
        {"precision": []},
        {"resolved_moving_grid": {"shape": [2, 3], "grid_to_level0": [[1, 0], [0, 1]]}},
    ],
)
def test_malformed_runtime_structures_rejected(change):
    with pytest.raises(AlignmentError):
        RegistrationRuntime.from_dict(RegistrationRuntime(backend="test").to_dict() | change)


def test_result_attempt_failure_invariants_and_version_cutover(frames):
    request = RegistrationRequest("same_section_restained", "identity")
    result = identity_alignment(frames[1], frames[0], request)
    failure = RegistrationFailure("internal_error", "Unexpected error")
    with pytest.raises(AlignmentError, match="typed failure"):
        replace(result.attempt, outcome="failed")
    with pytest.raises(AlignmentError, match="success has none"):
        replace(result.attempt, failure=failure)
    failed_attempt = replace(result.attempt, outcome="failed", failure=failure)
    with pytest.raises(AlignmentError, match="no QC"):
        replace(failed_attempt, qc_status="rejected")
    with pytest.raises(AlignmentError, match="disagree"):
        replace(result, attempt=failed_attempt)
    with pytest.raises(AlignmentError, match="candidate"):
        replace(result, backend_status="failed", attempt=failed_attempt)
    assert result.candidate is not None
    rejected = evaluate_alignment_qc(result.candidate, request, QCPolicy({"overlap": (None, 0.5)}))
    with pytest.raises(AlignmentError, match="not QC/reason"):
        replace(
            result, backend_status="failed", candidate=None, qc=rejected, attempt=failed_attempt
        )
    with pytest.raises(AlignmentError, match="not QC/reason"):
        replace(
            result,
            backend_status="failed",
            candidate=None,
            reason="raw text",
            attempt=failed_attempt,
        )
    with pytest.raises(AlignmentError, match="non-QC"):
        replace(
            result,
            backend_status="failed",
            candidate=None,
            attempt=replace(
                failed_attempt, failure=RegistrationFailure("qc_rejected", "Not backend failure")
            ),
        )
    with pytest.raises(AlignmentError, match="QC disagree"):
        replace(result, qc=rejected, attempt=replace(result.attempt, qc_status="accepted"))
    qc_result = replace(result, qc=rejected)
    assert qc_result.backend_status == "succeeded" and qc_result.attempt.failure is None
    for old in (
        result.metadata | {"format": "virtual_staining.alignment/1"},
        {k: v for k, v in result.metadata.items() if k != "attempt"},
    ):
        with pytest.raises(AlignmentError):
            AlignmentResult.from_dict(old)


@pytest.mark.parametrize(
    "operation,error,category,stage",
    [
        (
            "extractor",
            cv2.error("extractor failed"),
            "feature_extraction_failed",
            "feature_extraction",
        ),
        ("no_features", None, "insufficient_content", "feature_extraction"),
        ("matcher", cv2.error("matcher failed"), "matching_failed", "matching"),
        ("no_matches", None, "matching_failed", "matching"),
        ("optimizer", cv2.error("optimizer failed"), "optimizer_failed", "optimizer"),
        ("no_transform", None, "optimizer_failed", "optimizer"),
        ("singular_transform", None, "geometry_invalid", "geometry"),
        ("malformed_transform", None, "geometry_invalid", "geometry"),
        (
            "unexpected",
            RuntimeError("qc_rejected or unsupported_device text is not a category"),
            "internal_error",
            "registration",
        ),
    ],
)
def test_owned_backend_failures_are_typed(frames, monkeypatch, operation, error, category, stage):
    import virtual_staining.data.alignment.registration as registration

    keys = [cv2.KeyPoint(float(i), float(i), 1.0) for i in range(4)]
    extractor = Mock(detectAndCompute=Mock(return_value=(keys, np.ones((4, 128), np.float32))))
    matcher = Mock(
        knnMatch=Mock(
            return_value=[
                [
                    cv2.DMatch(_queryIdx=i, _trainIdx=i, _distance=1),
                    cv2.DMatch(_queryIdx=i, _trainIdx=i, _distance=3),
                ]
                for i in range(4)
            ]
        )
    )
    optimizer = Mock(return_value=(np.eye(2, 3), np.ones((4, 1), np.uint8)))
    monkeypatch.setattr(registration.cv2, "SIFT_create", Mock(return_value=extractor))
    monkeypatch.setattr(registration.cv2, "BFMatcher", Mock(return_value=matcher))
    monkeypatch.setattr(registration.cv2, "estimateAffine2D", optimizer)
    if operation == "extractor":
        extractor.detectAndCompute.side_effect = error
    elif operation == "no_features":
        extractor.detectAndCompute.return_value = ([], None)
    elif operation == "matcher":
        matcher.knnMatch.side_effect = error
    elif operation == "no_matches":
        matcher.knnMatch.return_value = []
    elif operation == "optimizer":
        optimizer.side_effect = error
    elif operation == "no_transform":
        optimizer.return_value = (None, None)
    elif operation == "singular_transform":
        optimizer.return_value = (np.zeros((2, 3)), None)
    elif operation == "malformed_transform":
        optimizer.return_value = (np.eye(2), None)
    else:
        optimizer.side_effect = error
    result = resolve_alignment(
        image(frames[1]), image(frames[0]), RegistrationRequest("same_section_restained", "affine")
    )
    assert result.backend_status == "failed" and result.candidate is None and result.qc is None
    assert result.reason is None and result.attempt.stage == stage
    assert result.attempt.failure is not None
    assert result.attempt.failure.category == category
    assert result.attempt.failure.message
    assert result.attempt.runtime.backend == "opencv_sift"
    assert result.attempt.runtime.backend_version == cv2.__version__
    assert result.attempt.runtime.resources is None
    assert (
        AlignmentResult.from_dict(json.loads(json.dumps(result.metadata))).attempt.failure
        == result.attempt.failure
    )


@pytest.mark.parametrize(
    "error,category,subcode",
    [
        (MemoryError(), "resource_exhausted", "cpu_memory"),
        (ImportError("Dependency unavailable"), "backend_unavailable", "missing_dependency"),
        (KeyboardInterrupt(), "interrupted", None),
        (OSError("missing_checkpoint"), "internal_error", None),
        (ValueError("unexpected input to an internal helper"), "internal_error", None),
    ],
)
def test_resource_and_unexpected_failures_do_not_parse_exception_messages(
    frames, monkeypatch, error, category, subcode
):
    monkeypatch.setattr(
        "virtual_staining.data.alignment.registration._estimate_affine", Mock(side_effect=error)
    )
    result = resolve_alignment(
        image(frames[1]), image(frames[0]), RegistrationRequest("same_section_restained", "affine")
    )
    assert result.attempt.failure is not None
    assert result.attempt.failure.category == category and result.attempt.failure.subcode == subcode
    assert result.candidate is None and result.qc is None


def test_unavailable_sift_is_a_backend_failure(frames, monkeypatch):
    monkeypatch.setattr("virtual_staining.data.alignment.registration.cv2.SIFT_create", None)
    result = resolve_alignment(
        image(frames[1]), image(frames[0]), RegistrationRequest("same_section_restained", "affine")
    )
    assert result.attempt.failure is not None
    assert result.attempt.failure.category == "backend_unavailable"
    assert result.attempt.failure.subcode == "missing_dependency"
