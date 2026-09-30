from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import cv2
import numpy as np

from virtual_staining.data.alignment.models import (
    AlignmentError,
    AlignmentImage,
    AlignmentResult,
    AlignmentTransform,
    GridGeometry,
    ImageGeometry,
    QCDecision,
    QCPolicy,
    RegistrationRequest,
    SpatialEvidence,
)
from virtual_staining.data.alignment.warping import _coordinates, _sample_evidence


def _ratio_test_matches(
    knn_matches: Sequence[Sequence[cv2.DMatch]],
    *,
    ratio_threshold: float = 0.75,
) -> list[cv2.DMatch]:
    return [
        pair[0]
        for pair in knn_matches
        if len(pair) >= 2 and pair[0].distance < ratio_threshold * pair[1].distance
    ]


def _estimate_affine(
    reference: np.ndarray,
    moving: np.ndarray,
    *,
    family: str,
    reference_grid: GridGeometry,
    moving_grid: GridGeometry,
) -> tuple[np.ndarray, dict[str, int | float | None]]:
    """SIFT/RANSAC execution diagnostics are never QC acceptance evidence."""
    clahe = cv2.createCLAHE(clipLimit=18.0, tileGridSize=(8, 8))
    sift = cv2.SIFT_create(nfeatures=10000)  # type: ignore[attr-defined]
    features = []
    for image in (reference, moving):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        features.append(sift.detectAndCompute(clahe.apply(gray), None))
    (ref_keys, ref_desc), (mov_keys, mov_desc) = features
    if ref_desc is None or mov_desc is None or min(len(ref_keys), len(mov_keys)) < 4:
        raise AlignmentError("Not enough features for alignment")
    matches = _ratio_test_matches(cv2.BFMatcher(cv2.NORM_L2).knnMatch(ref_desc, mov_desc, k=2))
    if len(matches) < 4:
        raise AlignmentError("Not enough good descriptor matches for alignment")
    ref_points = np.array([ref_keys[m.queryIdx].pt for m in matches], dtype=np.float64)
    mov_points = np.array([mov_keys[m.trainIdx].pt for m in matches], dtype=np.float64)
    if family == "similarity":
        # Enforce similarity in native pixels, even with anisotropic estimation grids.
        # For this family RANSAC residuals are in reference level-0 pixels.
        ref_map, mov_map = reference_grid.grid_to_level0, moving_grid.grid_to_level0
        ref_points = ref_points @ ref_map[:2, :2].T + ref_map[:2, 2]
        mov_points = mov_points @ mov_map[:2, :2].T + mov_map[:2, 2]
    estimate = cv2.estimateAffinePartial2D if family == "similarity" else cv2.estimateAffine2D
    matrix, inliers = estimate(
        mov_points,
        ref_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=5.0,
        maxIters=2000,
        confidence=0.99,
        refineIters=10,
    )
    if matrix is None:
        raise AlignmentError("Affine estimation failed")
    count = int(inliers.sum()) if inliers is not None else None
    homogeneous = np.vstack((matrix, [0, 0, 1]))
    if family == "similarity":
        homogeneous = (
            np.linalg.inv(reference_grid.grid_to_level0) @ homogeneous @ moving_grid.grid_to_level0
        )
    return homogeneous, dict(
        n_keypoints_reference=len(ref_keys),
        n_keypoints_moving=len(mov_keys),
        n_matches=len(matches),
        n_inliers=count,
        inlier_ratio=count / len(matches) if count is not None else None,
    )


def identity_alignment(
    reference: ImageGeometry,
    moving: ImageGeometry,
    request: RegistrationRequest,
    reason: str | None = None,
) -> AlignmentResult:
    request.reference_region(reference)
    if request.family != "identity":
        raise AlignmentError("Identity candidate requires identity permission")
    return AlignmentResult(
        "succeeded",
        "identity",
        request,
        AlignmentTransform(moving, reference, "identity", np.eye(3)),
        reason=reason,
    )


def resolve_alignment(
    reference: AlignmentImage,
    moving: AlignmentImage,
    request: RegistrationRequest,
) -> AlignmentResult:
    """Estimate one direct candidate; no config, readers, runs or automatic QC acceptance.

    The requested family constrains native geometry, including after grid conversion.
    Similarity fitting uses native keypoint coordinates so anisotropic grid scales
    cannot change the permitted transform family.
    """
    request.reference_region(reference.geometry)
    if request.family == "identity":
        return identity_alignment(reference.geometry, moving.geometry, request)
    try:
        matrix, diagnostics = _estimate_affine(
            reference.preview,
            moving.preview,
            family=request.family,
            reference_grid=reference.grid,
            moving_grid=moving.grid,
        )
        candidate = AlignmentTransform.from_estimated(
            moving.geometry, reference.geometry, matrix, moving.grid, reference.grid
        )
        if request.family == "similarity" and candidate.family == "affine":
            raise AlignmentError("Grid conversion exceeds requested native similarity family")
        candidate = replace(candidate, family=request.family)
        return AlignmentResult("succeeded", "affine_sift", request, candidate, diagnostics)
    except (AlignmentError, cv2.error, RuntimeError, OSError) as exc:
        return AlignmentResult("failed", "affine_sift", request, None, reason=str(exc))


def _overlap(transform: AlignmentTransform, region: tuple[int, int, int, int]) -> float:
    """Fraction of requested reference pixel-cell extent covered by transformed moving extent."""
    h, w = transform.moving.shape
    polygon = transform.map_points(
        np.array([[-0.5, -0.5], [w - 0.5, -0.5], [w - 0.5, h - 0.5], [-0.5, h - 0.5]])
    )
    x, y, rw, rh = region
    for axis, edge, sign in (
        (0, x - 0.5, 1),
        (0, x + rw - 0.5, -1),
        (1, y - 0.5, 1),
        (1, y + rh - 0.5, -1),
    ):
        clipped = []
        for a, b in zip(polygon, np.roll(polygon, -1, axis=0), strict=True):
            a_in, b_in = sign * (a[axis] - edge) >= 0, sign * (b[axis] - edge) >= 0
            if a_in:
                clipped.append(a)
            if a_in != b_in:
                clipped.append(a + (b - a) * (edge - a[axis]) / (b[axis] - a[axis]))
        polygon = np.asarray(clipped)
        if len(polygon) == 0:
            return 0.0
    area = (
        abs(
            np.dot(polygon[:, 0], np.roll(polygon[:, 1], 1))
            - np.dot(polygon[:, 1], np.roll(polygon[:, 0], 1))
        )
        / 2
    )
    return float(area / (rh * rw))


def evaluate_alignment_qc(
    candidate: AlignmentTransform,
    request: RegistrationRequest,
    policy: QCPolicy,
    *,
    moving_landmarks: np.ndarray | None = None,
    reference_landmarks: np.ndarray | None = None,
    moving_support: SpatialEvidence | None = None,
    reference_support: SpatialEvidence | None = None,
    moving_validity: SpatialEvidence | None = None,
    reference_validity: SpatialEvidence | None = None,
) -> QCDecision:
    """Independent observations only; no backend-native scores enter this decision.

    Landmarks must be held out from estimation. Support IoU and paired usable fraction
    are evaluated on the supplied reference evidence grid over jointly known samples.
    Empty tissue union is unavailable, not zero. Missing evidence is retained even when
    the external policy does not require it. Acceptance is scoped to request.purpose;
    correspondence declarations remain separate from this geometric QC decision.
    """
    region = request.reference_region(candidate.reference)
    singular = np.linalg.svd(candidate.matrix[:2, :2], compute_uv=False)
    metrics: dict[str, float | None] = dict(
        min_scale=float(singular.min()),
        max_scale=float(singular.max()),
        translation=float(np.linalg.norm(candidate.matrix[:2, 2])),
        overlap=_overlap(candidate, region),
        landmark_rms=None,
        landmark_improvement=None,
        support_iou=None,
        observation_valid_fraction=None,
    )
    rejected, missing = [], []
    permitted = set(request.allowed_families or ("identity", "similarity", "affine"))
    if request.relationship == "same_coordinate_frame":
        permitted &= {"identity"}
    if (
        candidate.family not in permitted
        or (request.family == "identity" and candidate.family != "identity")
        or (request.family == "similarity" and candidate.family == "affine")
    ):
        rejected.append("candidate_family_disallowed")
    if request.existing_alignment == "identity" and candidate.family != "identity":
        rejected.append("candidate_contradicts_identity_declaration")
    if moving_landmarks is not None or reference_landmarks is not None:
        if moving_landmarks is None or reference_landmarks is None:
            raise AlignmentError("Both independent landmark arrays must be supplied")
        mapped = candidate.map_points(moving_landmarks)
        reference_points = candidate.inverse().map_points(reference_landmarks)
        if mapped.shape != reference_points.shape or len(mapped) == 0:
            raise AlignmentError("Independent landmarks must have matching nonempty Nx2 shapes")
        if request.diagnostic_region is not None:
            x, y, w, h = region
            if not np.all(
                (reference_landmarks >= [x - 0.5, y - 0.5])
                & (reference_landmarks < [x + w - 0.5, y + h - 0.5])
            ):
                raise AlignmentError("Landmarks exceed the declared diagnostic region")
        rms = float(np.sqrt(np.mean(np.sum((mapped - reference_landmarks) ** 2, axis=1))))
        identity_rms = float(
            np.sqrt(
                np.mean(np.sum((np.asarray(moving_landmarks) - reference_landmarks) ** 2, axis=1))
            )
        )
        metrics.update(landmark_rms=rms, landmark_improvement=identity_rms - rms)
    for metric, moving, reference, kind in (
        ("support_iou", moving_support, reference_support, "tissue_support"),
        ("observation_valid_fraction", moving_validity, reference_validity, "observation_validity"),
    ):
        for evidence, asset in ((moving, candidate.moving), (reference, candidate.reference)):
            if evidence is not None and (evidence.asset != asset or evidence.kind != kind):
                raise AlignmentError("QC evidence kind/asset mismatch")
        if moving is None or reference is None:
            continue
        h, w = reference.grid.shape
        native = _coordinates(reference.grid.grid_to_level0, 0, 0, w, h)
        inverse = candidate.inverse().matrix
        points = native @ inverse[:2, :2].T + inverse[:2, 2]
        values, known = _sample_evidence(
            moving, points, conservative=kind == "observation_validity"
        )
        # Evidence outside either native asset is not an observation of that asset.
        for coords, asset in ((native, candidate.reference), (points, candidate.moving)):
            known &= (
                (coords[..., 0] >= -0.5)
                & (coords[..., 0] < asset.shape[1] - 0.5)
                & (coords[..., 1] >= -0.5)
                & (coords[..., 1] < asset.shape[0] - 0.5)
            )
        x, y, width, height = region
        known &= (
            (native[..., 0] >= x - 0.5)
            & (native[..., 0] < x + width - 0.5)
            & (native[..., 1] >= y - 0.5)
            & (native[..., 1] < y + height - 0.5)
        )
        denominator = (
            int(np.count_nonzero(known & (values | reference.values)))
            if kind == "tissue_support"
            else int(np.count_nonzero(known))
        )
        if denominator:
            metrics[metric] = float(
                np.count_nonzero(known & values & reference.values) / denominator
            )
    missing.extend(name for name, value in metrics.items() if value is None)
    unmet = []
    for name, (low, high) in sorted(policy.thresholds.items()):
        value = metrics[name]
        if value is None:
            unmet.append(f"missing:{name}")
        elif (low is not None and value < low) or (high is not None and value > high):
            rejected.append(f"threshold_failed:{name}")
    if not policy.thresholds:
        unmet.append("missing:qc_thresholds")
        missing.append("qc_thresholds")
    if request.purpose == "dense_correspondence" and not request.correspondence_evidence:
        unmet.append("missing:correspondence_evidence")
        missing.append("correspondence_evidence")
    status = "rejected" if rejected else "insufficient_evidence" if unmet else "accepted"
    return QCDecision(status, metrics, tuple(sorted(missing)), tuple(rejected + unmet))
