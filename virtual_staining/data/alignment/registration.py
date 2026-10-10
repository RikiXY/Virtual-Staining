from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import InitVar, dataclass, field, replace
from time import perf_counter, time
from typing import Any, Literal

import cv2
import numpy as np

from virtual_staining.data.alignment.models import (
    AlignmentError,
    AlignmentImage,
    AlignmentResult,
    AlignmentTransform,
    FailureCategory,
    GridGeometry,
    ImageGeometry,
    QCDecision,
    QCPolicy,
    RegistrationAttempt,
    RegistrationFailure,
    RegistrationRequest,
    RegistrationRuntime,
    SpatialEvidence,
)
from virtual_staining.data.alignment.warping import _coordinates, _sample_evidence

_QC_BLOCK_SIZE = 256


@dataclass(frozen=True)
class RegistrationBackend:
    """Explicit preparation callable and frozen JSON semantic identity.

    The caller owns deterministic execution and must identify all relevant options,
    including independent QC policy/evidence versions. Results preserve the supplied
    request and use the real moving/reference geometry. No callable is serialized.
    """

    register: Callable[[AlignmentImage, AlignmentImage, RegistrationRequest], AlignmentResult]
    identifier: str
    version: str
    options: InitVar[dict[str, Any] | None] = None
    qc_disposition: InitVar[dict[str, Literal["continue", "skip_set", "error"]] | None] = None
    _metadata_json: str = field(init=False, repr=False)

    def __post_init__(self, options: dict[str, Any] | None, qc_disposition: dict | None) -> None:
        if not callable(self.register):
            raise ValueError("Registration backend must be callable")
        if any(not isinstance(v, str) or not v.strip() for v in (self.identifier, self.version)):
            raise ValueError("Registration backend requires a stable identifier and version")
        disposition = qc_disposition if qc_disposition is not None else {}
        if not isinstance(disposition, dict) or any(
            key not in {"unassessed", "accepted", "rejected", "insufficient_evidence"}
            or not isinstance(value, str)
            or value not in {"continue", "skip_set", "error"}
            for key, value in disposition.items()
        ):
            raise ValueError("Invalid registration QC disposition")
        options = options if options is not None else {}
        if not isinstance(options, dict):
            raise ValueError("Registration options must be a JSON object")
        metadata = dict(
            identifier=self.identifier,
            version=self.version,
            options=options,
            qc_disposition=disposition,
        )
        try:
            encoded = json.dumps(metadata, sort_keys=True, allow_nan=False)
            if json.loads(encoded) != metadata:
                raise ValueError("Options must use JSON values and string keys")
        except (TypeError, ValueError) as exc:
            raise ValueError("Registration options must contain finite JSON values") from exc
        object.__setattr__(self, "_metadata_json", encoded)

    @property
    def metadata(self) -> dict[str, Any]:
        """Detached semantic configuration, never per-asset execution results."""
        return json.loads(self._metadata_json)

    def __call__(
        self, reference: AlignmentImage, moving: AlignmentImage, request: RegistrationRequest
    ) -> AlignmentResult:
        result = self.register(reference, moving, request)
        if not isinstance(result, AlignmentResult) or result.request != request:
            raise AlignmentError(
                "Injected registration must return AlignmentResult for its request"
            )
        candidate = result.candidate
        if candidate is not None:
            if candidate.reference != reference.geometry or candidate.moving != moving.geometry:
                raise AlignmentError(
                    "Injected candidate must map the actual moving/reference frames"
                )
            if (
                candidate.family
                not in (request.allowed_families or ("identity", "similarity", "affine"))
                or (request.family == "identity" and candidate.family != "identity")
                or (request.family == "similarity" and candidate.family == "affine")
            ):
                raise AlignmentError("Injected candidate exceeds requested transform permissions")
        return result


class _BackendFailure(AlignmentError):
    def __init__(
        self, stage: str, category: FailureCategory, message: str, subcode: str | None = None
    ) -> None:
        message = message or f"Backend execution failed during {stage}"
        super().__init__(message)
        self.stage = stage
        self.failure = RegistrationFailure(category, message, subcode)


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
    if not callable(getattr(cv2, "SIFT_create", None)):
        raise _BackendFailure(
            "feature_extraction",
            "backend_unavailable",
            "OpenCV SIFT is unavailable",
            "missing_dependency",
        )
    try:
        clahe = cv2.createCLAHE(clipLimit=18.0, tileGridSize=(8, 8))
        sift = cv2.SIFT_create(nfeatures=10000)  # type: ignore[attr-defined]
        features = []
        for image in (reference, moving):
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
            features.append(sift.detectAndCompute(clahe.apply(gray), None))
    except cv2.error as exc:
        raise _BackendFailure("feature_extraction", "feature_extraction_failed", str(exc)) from exc
    (ref_keys, ref_desc), (mov_keys, mov_desc) = features
    if ref_desc is None or mov_desc is None or min(len(ref_keys), len(mov_keys)) < 4:
        raise _BackendFailure(
            "feature_extraction", "insufficient_content", "Not enough features for alignment"
        )
    try:
        matches = _ratio_test_matches(cv2.BFMatcher(cv2.NORM_L2).knnMatch(ref_desc, mov_desc, k=2))
    except cv2.error as exc:
        raise _BackendFailure("matching", "matching_failed", str(exc)) from exc
    if len(matches) < 4:
        raise _BackendFailure(
            "matching", "matching_failed", "Not enough good descriptor matches for alignment"
        )
    ref_points = np.array([ref_keys[m.queryIdx].pt for m in matches], dtype=np.float64)
    mov_points = np.array([mov_keys[m.trainIdx].pt for m in matches], dtype=np.float64)
    if family == "similarity":
        # Enforce similarity in native pixels, even with anisotropic estimation grids.
        # For this family RANSAC residuals are in reference level-0 pixels.
        ref_map, mov_map = reference_grid.grid_to_level0, moving_grid.grid_to_level0
        ref_points = ref_points @ ref_map[:2, :2].T + ref_map[:2, 2]
        mov_points = mov_points @ mov_map[:2, :2].T + mov_map[:2, 2]
    estimate = cv2.estimateAffinePartial2D if family == "similarity" else cv2.estimateAffine2D
    try:
        matrix, inliers = estimate(
            mov_points,
            ref_points,
            method=cv2.RANSAC,
            ransacReprojThreshold=5.0,
            maxIters=2000,
            confidence=0.99,
            refineIters=10,
        )
    except cv2.error as exc:
        raise _BackendFailure("optimizer", "optimizer_failed", str(exc)) from exc
    if matrix is None:
        raise _BackendFailure("optimizer", "optimizer_failed", "Affine estimation failed")
    if np.asarray(matrix).shape != (2, 3):
        raise AlignmentError("Estimator returned malformed affine geometry")
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
    started_at, started = time(), perf_counter()
    request.reference_region(reference)
    if request.family != "identity":
        raise AlignmentError("Identity candidate requires identity permission")
    return AlignmentResult(
        "succeeded",
        "identity",
        request,
        AlignmentTransform(moving, reference, "identity", np.eye(3)),
        reason=reason,
        attempt=RegistrationAttempt(
            stage="identity",
            runtime=RegistrationRuntime(backend="identity"),
            outcome="succeeded",
            started_at=started_at,
            ended_at=time(),
            duration_seconds=perf_counter() - started,
        ),
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
    started_at, started = time(), perf_counter()
    runtime = RegistrationRuntime(
        backend="opencv_sift",
        backend_version=cv2.__version__,
        device="cpu",
        support_mode="unused",
        requested_moving_grid=moving.grid,
        requested_reference_grid=reference.grid,
        resolved_moving_grid=moving.grid,
        resolved_reference_grid=reference.grid,
    )
    failure = None
    stage = "registration"
    candidate = None
    diagnostics = {}
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
    except _BackendFailure as exc:
        failure, stage = exc.failure, exc.stage
    except AlignmentError as exc:
        failure = RegistrationFailure("geometry_invalid", str(exc))
        stage = "geometry"
    except (Exception, KeyboardInterrupt) as exc:
        category: FailureCategory = "internal_error"
        subcode = None
        if isinstance(exc, MemoryError):
            category, subcode = "resource_exhausted", "cpu_memory"
        elif isinstance(exc, ImportError):
            category, subcode = "backend_unavailable", "missing_dependency"
        elif isinstance(exc, KeyboardInterrupt):
            category = "interrupted"
        failure = RegistrationFailure(category, str(exc) or type(exc).__name__, subcode)
    outcome = "failed" if failure is not None else "succeeded"
    return AlignmentResult(
        outcome,
        "affine_sift",
        request,
        candidate if failure is None else None,
        diagnostics,
        attempt=RegistrationAttempt(
            stage=stage,
            runtime=runtime,
            outcome=outcome,
            failure=failure,
            started_at=started_at,
            ended_at=time(),
            duration_seconds=perf_counter() - started,
        ),
    )


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
    if request.relationship == "same_coordinate_frame":
        candidate.reference.validate_shared_frame(candidate.moving)
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
        inverse = candidate.inverse().matrix
        x, y, width, height = region
        numerator = denominator = 0
        for row in range(0, h, _QC_BLOCK_SIZE):
            for column in range(0, w, _QC_BLOCK_SIZE):
                block_h, block_w = min(_QC_BLOCK_SIZE, h - row), min(_QC_BLOCK_SIZE, w - column)
                native = _coordinates(reference.grid.grid_to_level0, column, row, block_w, block_h)
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
                known &= (
                    (native[..., 0] >= x - 0.5)
                    & (native[..., 0] < x + width - 0.5)
                    & (native[..., 1] >= y - 0.5)
                    & (native[..., 1] < y + height - 0.5)
                )
                reference_values = reference.values[row : row + block_h, column : column + block_w]
                denominator += (
                    int(np.count_nonzero(known & (values | reference_values)))
                    if kind == "tissue_support"
                    else int(np.count_nonzero(known))
                )
                numerator += int(np.count_nonzero(known & values & reference_values))
                # Drop all block arrays, including the loop alias, before the next allocation.
                del native, points, coords, values, known, reference_values
        if denominator:
            metrics[metric] = float(numerator / denominator)
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
