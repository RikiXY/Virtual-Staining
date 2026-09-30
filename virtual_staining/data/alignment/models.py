from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any, Literal, get_args

import numpy as np

Family = Literal["identity", "similarity", "affine"]
_FAMILIES = {"identity", "similarity", "affine"}
_FORMAT = "virtual_staining.alignment/1"
_RESULT_FORMAT = "virtual_staining.alignment.result/2"


class AlignmentError(ValueError):
    """Invalid alignment inputs, incompatible geometry, or failed registration."""


def _matrix(value: np.ndarray) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise AlignmentError("Expected a finite homogeneous 3x3 matrix")
    if not np.array_equal(matrix[2], [0, 0, 1]):
        raise AlignmentError("Only affine homogeneous geometry is supported")
    determinant = np.linalg.det(matrix[:2, :2])
    if not np.isfinite(determinant) or determinant <= 0:
        raise AlignmentError("Singular geometry and reflection are not supported")
    if not np.isfinite(np.linalg.inv(matrix)).all():
        raise AlignmentError("Geometry has no finite inverse")
    return np.frombuffer(matrix.tobytes(), dtype=np.float64).reshape(3, 3)


def _shape(shape: tuple[int, int]) -> None:
    if len(shape) != 2 or any(type(v) is not int or v <= 0 for v in shape):
        raise AlignmentError("Shape must contain positive integer height and width")


def _fields(data: dict[str, Any], keys: str) -> None:
    if not isinstance(data, dict) or set(data) != set(keys.split()):
        raise AlignmentError("Malformed or unsupported alignment metadata")


@dataclass(frozen=True)
class ImageGeometry:
    """Named level-0 frame; shape is (rows, columns), MPP is nullable (x, y)."""

    name: str
    shape: tuple[int, int]
    mpp: tuple[float | None, float | None] = (None, None)

    def __post_init__(self) -> None:
        _shape(self.shape)
        if not isinstance(self.name, str) or not self.name:
            raise AlignmentError("An explicit asset/frame name is required")
        if len(self.mpp) != 2 or any(
            v is not None and (not np.isfinite(v) or v <= 0) for v in self.mpp
        ):
            raise AlignmentError("MPP must contain two positive finite values or nulls")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ImageGeometry:
        _fields(data, "name shape mpp")
        return cls(data["name"], tuple(data["shape"]), tuple(data["mpp"]))

    def validate_shared_frame(self, moving: ImageGeometry) -> None:
        """Require equal native shape and compatible known per-axis MPP.

        Unknown spacing is unavailable evidence, not an inferred physical scale.
        The reference-first comparison preserves the established alignment tolerance.
        """
        if self.shape != moving.shape:
            raise AlignmentError(f"identity alignment requires equal geometry for {moving.name}")
        for axis, left, right in zip(("x", "y"), self.mpp, moving.mpp, strict=True):
            if left is not None and right is not None and not np.isclose(left, right, rtol=0.01):
                raise AlignmentError(
                    f"identity alignment has incompatible mpp_{axis} for {moving.name}"
                )


@dataclass(frozen=True, eq=False)
class GridGeometry:
    """Grid pixel centres -> native level-0 centres, including crop and resize offsets.

    Integer (x, y) is a centre, x right/y down. A grid of shape (h, w)
    has pixel-cell extent [-.5, w-.5) x [-.5, h-.5). Array indices are (y, x).
    """

    shape: tuple[int, int]
    grid_to_level0: np.ndarray

    def __post_init__(self) -> None:
        _shape(self.shape)
        matrix = _matrix(self.grid_to_level0)
        if matrix[0, 1] != 0 or matrix[1, 0] != 0 or min(matrix[0, 0], matrix[1, 1]) <= 0:
            raise AlignmentError("Grid maps require positive per-axis scales and offsets")
        object.__setattr__(self, "grid_to_level0", matrix)

    @classmethod
    def resized_crop(
        cls, shape: tuple[int, int], *, origin: tuple[float, float], scale: tuple[float, float]
    ) -> GridGeometry:
        """Origin is the native centre of crop pixel (0, 0); scale is native/grid."""
        sx, sy = scale
        x, y = origin
        return cls(
            shape, np.array([[sx, 0, x + (sx - 1) / 2], [0, sy, y + (sy - 1) / 2], [0, 0, 1]])
        )

    def to_dict(self) -> dict[str, Any]:
        return {"shape": list(self.shape), "grid_to_level0": self.grid_to_level0.tolist()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GridGeometry:
        _fields(data, "shape grid_to_level0")
        return cls(tuple(data["shape"]), np.asarray(data["grid_to_level0"]))


def _family(matrix: np.ndarray) -> Family:
    if np.array_equal(matrix, np.eye(3)):
        return "identity"
    linear = matrix[:2, :2] / np.max(np.abs(matrix[:2, :2]))
    gram = linear.T @ linear
    if np.allclose(gram, np.eye(2) * np.trace(gram) / 2, rtol=1e-12, atol=1e-12):
        return "similarity"
    return "affine"


@dataclass(frozen=True, eq=False)
class AlignmentTransform:
    """Canonical forward mapping: moving level-0 centres -> reference level-0 centres."""

    moving: ImageGeometry
    reference: ImageGeometry
    family: Family
    matrix: np.ndarray
    moving_grid: GridGeometry | None = None
    reference_grid: GridGeometry | None = None

    def __post_init__(self) -> None:
        matrix = _matrix(self.matrix)
        if self.family not in _FAMILIES:
            raise AlignmentError("Unsupported transform family")
        actual = _family(matrix)
        if self.family == "identity" and not np.array_equal(matrix, np.eye(3)):
            raise AlignmentError("Identity requires an identity matrix")
        if self.family == "similarity" and actual == "affine":
            raise AlignmentError("Similarity requires rotation and isotropic scale")
        if (self.moving_grid is None) != (self.reference_grid is None):
            raise AlignmentError("Both estimation grids must be supplied together")
        object.__setattr__(self, "matrix", matrix)

    @classmethod
    def from_estimated(
        cls,
        moving: ImageGeometry,
        reference: ImageGeometry,
        matrix: np.ndarray,
        moving_grid: GridGeometry,
        reference_grid: GridGeometry,
    ) -> AlignmentTransform:
        native = (
            reference_grid.grid_to_level0
            @ _matrix(matrix)
            @ np.linalg.inv(moving_grid.grid_to_level0)
        )
        return cls(moving, reference, _family(native), native, moving_grid, reference_grid)

    def map_points(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
            raise AlignmentError("Points must be finite Nx2 (x, y) coordinates")
        return points @ self.matrix[:2, :2].T + self.matrix[:2, 2]

    def inverse(self) -> AlignmentTransform:
        matrix = np.linalg.inv(self.matrix)
        return AlignmentTransform(
            self.reference, self.moving, self.family, matrix, self.reference_grid, self.moving_grid
        )

    def then(self, following: AlignmentTransform) -> AlignmentTransform:
        """Apply self, then following; the intermediate native frame must match."""
        if self.reference != following.moving:
            raise AlignmentError("Composition requires the same intermediate frame geometry")
        matrix = following.matrix @ self.matrix
        family = _family(matrix)
        return AlignmentTransform(self.moving, following.reference, family, matrix)

    def to_dict(self) -> dict[str, Any]:
        return dict(
            format=_FORMAT,
            kind="transform",
            direction="moving_level0_to_reference_level0",
            moving=asdict(self.moving),
            reference=asdict(self.reference),
            family=self.family,
            matrix=self.matrix.tolist(),
            moving_grid=self.moving_grid.to_dict() if self.moving_grid else None,
            reference_grid=self.reference_grid.to_dict() if self.reference_grid else None,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AlignmentTransform:
        _fields(
            data, "format kind direction moving reference family matrix moving_grid reference_grid"
        )
        if (data["format"], data["kind"], data["direction"]) != (
            _FORMAT,
            "transform",
            "moving_level0_to_reference_level0",
        ):
            raise AlignmentError("Unsupported transform identity or direction")
        return cls(
            ImageGeometry.from_dict(data["moving"]),
            ImageGeometry.from_dict(data["reference"]),
            data["family"],
            np.asarray(data["matrix"]),
            GridGeometry.from_dict(data["moving_grid"])
            if data["moving_grid"] is not None
            else None,
            GridGeometry.from_dict(data["reference_grid"])
            if data["reference_grid"] is not None
            else None,
        )


@dataclass(frozen=True)
class SpatialEvidence:
    """Boolean evidence bound to an asset. True means tissue present / observation usable.

    Outside this map's grid the evidence is unknown. Neither kind asserts correspondence.
    Foreground/loss masks are not registration evidence.
    """

    asset: ImageGeometry
    grid: GridGeometry
    values: np.ndarray
    kind: Literal["tissue_support", "observation_validity"]

    def __post_init__(self) -> None:
        if self.kind not in {"tissue_support", "observation_validity"}:
            raise AlignmentError("Unsupported evidence semantics")
        if self.values.dtype != np.bool_ or self.values.shape != self.grid.shape:
            raise AlignmentError("Evidence must be a boolean array matching its grid")


@dataclass(frozen=True)
class AlignmentImage:
    """Borrowed uint8 BGR/grayscale estimation image with an explicit native mapping."""

    preview: np.ndarray
    geometry: ImageGeometry
    grid: GridGeometry
    tissue_support: SpatialEvidence | None = None
    observation_validity: SpatialEvidence | None = None

    def __post_init__(self) -> None:
        if (
            self.preview.dtype != np.uint8
            or self.preview.ndim not in (2, 3)
            or (self.preview.ndim == 3 and self.preview.shape[2] != 3)
            or self.preview.shape[:2] != self.grid.shape
        ):
            raise AlignmentError("Preview must be uint8 BGR/grayscale matching its explicit grid")
        for name in ("tissue_support", "observation_validity"):
            evidence = getattr(self, name)
            if evidence is not None and (evidence.asset != self.geometry or evidence.kind != name):
                raise AlignmentError("Evidence kind/asset does not match its image")


@dataclass(frozen=True)
class RegistrationRequest:
    """Relationship, existing alignment and permission are independent declarations.

    Unknown relationships require an explicit bounded diagnostic region (x, y, w, h),
    using integer pixel-centre origins and half-open index ranges. This bounds the
    scientific scope, not the backend's resource use.
    """

    relationship: str
    family: Family
    allowed_families: tuple[Family, ...] | None = None
    existing_alignment: Literal["unknown", "identity", "unaligned"] = "unknown"
    purpose: Literal["dense_correspondence", "spatial_association", "diagnostic"] = "diagnostic"
    diagnostic_region: tuple[int, int, int, int] | None = None
    correspondence_evidence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        defaults = {
            "same_coordinate_frame": {"identity"},
            "same_section_different_modality": _FAMILIES,
            "same_section_restained": _FAMILIES,
            "serial_section": _FAMILIES,
            "unknown": _FAMILIES,
            "non_corresponding": set(),
        }
        if self.relationship not in defaults or self.family not in _FAMILIES:
            raise AlignmentError("Unsupported relationship or transform family")
        if self.purpose not in {"dense_correspondence", "spatial_association", "diagnostic"}:
            raise AlignmentError("Unsupported registration purpose")
        allowed = defaults[self.relationship]
        if self.allowed_families is not None:
            if not self.allowed_families or not set(self.allowed_families) <= allowed:
                raise AlignmentError("Contradictory transform permissions")
            allowed = set(self.allowed_families)
        if self.family not in allowed:
            raise AlignmentError("Transform family blocked by relationship/permissions")
        if self.existing_alignment not in {"unknown", "identity", "unaligned"}:
            raise AlignmentError("Unsupported existing-alignment declaration")
        if self.existing_alignment == "identity" and self.family != "identity":
            raise AlignmentError("Identity declaration contradicts requested transform")
        if self.relationship == "same_coordinate_frame" and self.existing_alignment == "unaligned":
            raise AlignmentError("Coordinate-frame relationship contradicts unaligned declaration")
        if self.relationship == "serial_section" and self.purpose == "dense_correspondence":
            raise AlignmentError("Serial sections permit spatial association only")
        if self.relationship == "unknown" and (
            self.purpose != "diagnostic" or self.diagnostic_region is None
        ):
            raise AlignmentError("Unknown relationship requires a bounded diagnostic declaration")
        if self.diagnostic_region is not None:
            x, y, w, h = self.diagnostic_region
            if any(type(v) is not int for v in (x, y, w, h)) or min(w, h) <= 0:
                raise AlignmentError("Invalid diagnostic region")
        if any(not isinstance(v, str) or not v for v in self.correspondence_evidence):
            raise AlignmentError("Correspondence evidence must contain nonempty declarations")

    def reference_region(self, reference: ImageGeometry) -> tuple[int, int, int, int]:
        """QC scope, constrained to the explicitly named reference's native extent."""
        h, w = reference.shape
        region = self.diagnostic_region or (0, 0, w, h)
        x, y, width, height = region
        if min(x, y) < 0 or x + width > w or y + height > h:
            raise AlignmentError("Diagnostic region exceeds the explicit reference geometry")
        return region


@dataclass(frozen=True)
class QCPolicy:
    """Externally chosen inclusive ranges; no scientific thresholds are defaulted.

    Metrics: min_scale, max_scale, translation, overlap, landmark_rms,
    landmark_improvement (identity RMS minus candidate RMS), support_iou,
    observation_valid_fraction. Distances are reference level-0 pixels.
    """

    thresholds: dict[str, tuple[float | None, float | None]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        supported = {
            "min_scale",
            "max_scale",
            "translation",
            "overlap",
            "landmark_rms",
            "landmark_improvement",
            "support_iou",
            "observation_valid_fraction",
        }
        for name, bounds in self.thresholds.items():
            if name not in supported or len(bounds) != 2 or bounds == (None, None):
                raise AlignmentError("Unsupported or empty QC threshold")
            low, high = bounds
            if any(v is not None and not np.isfinite(v) for v in bounds):
                raise AlignmentError("QC thresholds must be finite")
            if low is not None and high is not None and low > high:
                raise AlignmentError("Contradictory QC thresholds")


@dataclass(frozen=True)
class QCDecision:
    status: Literal["accepted", "rejected", "insufficient_evidence"]
    metrics: dict[str, float | None]
    missing_evidence: tuple[str, ...]
    reasons: tuple[str, ...]


FailureCategory = Literal[
    "image_invalid",
    "metadata_invalid",
    "geometry_invalid",
    "insufficient_content",
    "insufficient_overlap",
    "support_generation_failed",
    "feature_extraction_failed",
    "matching_failed",
    "optimizer_failed",
    "qc_rejected",
    "affine_implausible",
    "deformation_implausible",
    "deformation_folding",
    "resource_exhausted",
    "backend_unavailable",
    "interrupted",
    "output_corrupt",
    "internal_error",
]


def _optional_text(name: str, value: str | None) -> None:
    if value is not None and (not isinstance(value, str) or not value.strip()):
        raise AlignmentError(f"{name} must be a nonempty string or null")


def _nonnegative_number(name: str, value: float | None) -> None:
    if value is not None and (
        type(value) not in (int, float) or not np.isfinite(value) or value < 0
    ):
        raise AlignmentError(f"{name} must be a finite nonnegative number or null")


@dataclass(frozen=True)
class RegistrationFailure:
    """Stable failure identity; subcodes are caller-defined nonempty strings."""

    category: FailureCategory
    message: str
    subcode: str | None = None

    def __post_init__(self) -> None:
        if self.category not in get_args(FailureCategory):
            raise AlignmentError("Unsupported registration failure category")
        if not isinstance(self.message, str) or not self.message.strip():
            raise AlignmentError("Failure message must be a nonempty string")
        _optional_text("Failure subcode", self.subcode)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RegistrationFailure:
        _fields(data, "category message subcode")
        return cls(**data)


@dataclass(frozen=True, kw_only=True)
class RegistrationResources:
    """Optional observed resource measurements; unavailable measurements stay null."""

    peak_cpu_memory_bytes: int | None = None
    peak_gpu_memory_bytes: int | None = None
    peak_temp_disk_bytes: int | None = None
    reader_count: int | None = None

    def __post_init__(self) -> None:
        for item in fields(self):
            value = getattr(self, item.name)
            if value is not None and (type(value) is not int or value < 0):
                raise AlignmentError(f"{item.name} must be a nonnegative integer or null")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RegistrationResources:
        _fields(data, " ".join(item.name for item in fields(cls)))
        return cls(**data)


_RUNTIME_GRIDS = (
    "requested_moving_grid",
    "requested_reference_grid",
    "resolved_moving_grid",
    "resolved_reference_grid",
)


@dataclass(frozen=True, kw_only=True)
class RegistrationRuntime:
    """Runtime evidence, not runtime configuration or an instrumentation manager.

    Requested/resolved resolution uses the existing grids: per-axis native pixels
    per grid pixel, with explicit centre offsets. Physical MPP is never inferred.
    """

    backend: str
    backend_version: str | None = None
    model_id: str | None = None
    checkpoint_id: str | None = None
    checkpoint_hash: str | None = None
    input_metadata_ref: str | None = None
    input_fingerprint_ref: str | None = None
    requested_moving_grid: GridGeometry | None = None
    requested_reference_grid: GridGeometry | None = None
    resolved_moving_grid: GridGeometry | None = None
    resolved_reference_grid: GridGeometry | None = None
    support_mode: str | None = None
    device: str | None = None
    precision: str | None = None
    determinism_mode: str | None = None
    seed: int | None = None
    scientific_parameter_hash: str | None = None
    resources: RegistrationResources | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.backend, str) or not self.backend.strip():
            raise AlignmentError("Runtime backend must be a nonempty identifier")
        for item in fields(self):
            value = getattr(self, item.name)
            if item.name in _RUNTIME_GRIDS:
                if value is not None and not isinstance(value, GridGeometry):
                    raise AlignmentError("Runtime resolution must use explicit GridGeometry")
            elif item.name not in {"seed", "resources"}:
                _optional_text(item.name, value)
        if self.seed is not None and (type(self.seed) is not int or self.seed < 0):
            raise AlignmentError("Seed must be a nonnegative integer or null")
        if self.resources is not None and not isinstance(self.resources, RegistrationResources):
            raise AlignmentError("Invalid resource measurements")

    def to_dict(self) -> dict[str, Any]:
        data = {item.name: getattr(self, item.name) for item in fields(self)}
        for name in _RUNTIME_GRIDS:
            data[name] = data[name].to_dict() if data[name] is not None else None
        data["resources"] = asdict(self.resources) if self.resources is not None else None
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RegistrationRuntime:
        _fields(data, " ".join(item.name for item in fields(cls)))
        values = dict(data)
        for name in _RUNTIME_GRIDS:
            if values[name] is not None:
                values[name] = GridGeometry.from_dict(values[name])
        if values["resources"] is not None:
            values["resources"] = RegistrationResources.from_dict(values["resources"])
        return cls(**values)


@dataclass(frozen=True, kw_only=True)
class RegistrationAttempt:
    """One attempt's evidence. Times are UTC Unix seconds; duration is monotonic seconds.

    Optional QC status is a recorded snapshot, not a decision made by the backend.
    At most 16 diagnostic references of at most 2048 characters are retained; no
    artifact contents, runtime handles, or retry execution belong in this record.
    """

    stage: str
    runtime: RegistrationRuntime
    outcome: Literal["succeeded", "failed"]
    run_id: str | None = None
    case_id: str | None = None
    attempt_id: str | None = None
    started_at: float | None = None
    ended_at: float | None = None
    duration_seconds: float | None = None
    qc_status: Literal["accepted", "rejected", "insufficient_evidence"] | None = None
    failure: RegistrationFailure | None = None
    fallback_decision: str | None = None
    next_attempt_id: str | None = None
    diagnostic_artifacts: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.stage, str) or not self.stage.strip():
            raise AlignmentError("Attempt stage must be a nonempty identifier")
        if not isinstance(self.runtime, RegistrationRuntime):
            raise AlignmentError("Attempt requires typed runtime evidence")
        if self.outcome not in ("succeeded", "failed"):
            raise AlignmentError("Unsupported attempt outcome")
        if self.failure is not None and not isinstance(self.failure, RegistrationFailure):
            raise AlignmentError("Attempt requires a typed failure")
        if (self.outcome == "failed") != (self.failure is not None):
            raise AlignmentError("Failed attempts require a typed failure; success has none")
        if self.qc_status not in (None, "accepted", "rejected", "insufficient_evidence"):
            raise AlignmentError("Unsupported attempt QC status")
        if self.outcome == "failed" and self.qc_status is not None:
            raise AlignmentError("Backend failure has no QC outcome")
        for name in ("run_id", "case_id", "attempt_id", "fallback_decision", "next_attempt_id"):
            _optional_text(name, getattr(self, name))
        for name in ("started_at", "ended_at", "duration_seconds"):
            _nonnegative_number(name, getattr(self, name))
        # Duration uses a monotonic clock; wall-clock adjustments need not match it.
        if not isinstance(self.diagnostic_artifacts, tuple) or len(self.diagnostic_artifacts) > 16:
            raise AlignmentError("Diagnostic artifacts require at most 16 references")
        for reference in self.diagnostic_artifacts:
            if not isinstance(reference, str) or not reference.strip() or len(reference) > 2048:
                raise AlignmentError("Diagnostic artifact references must be 1..2048 characters")

    def to_dict(self) -> dict[str, Any]:
        data = {item.name: getattr(self, item.name) for item in fields(self)}
        data.update(
            format=_RESULT_FORMAT,
            kind="attempt",
            runtime=self.runtime.to_dict(),
            failure=asdict(self.failure) if self.failure is not None else None,
            diagnostic_artifacts=list(self.diagnostic_artifacts),
        )
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RegistrationAttempt:
        _fields(data, "format kind " + " ".join(item.name for item in fields(cls)))
        if (data["format"], data["kind"]) != (_RESULT_FORMAT, "attempt"):
            raise AlignmentError("Unsupported attempt identity")
        values = {item.name: data[item.name] for item in fields(cls)}
        values["runtime"] = RegistrationRuntime.from_dict(values["runtime"])
        if values["failure"] is not None:
            values["failure"] = RegistrationFailure.from_dict(values["failure"])
        if not isinstance(values["diagnostic_artifacts"], list):
            raise AlignmentError("Diagnostic artifacts must be an array of references")
        values["diagnostic_artifacts"] = tuple(values["diagnostic_artifacts"])
        return cls(**values)


@dataclass(frozen=True)
class AlignmentResult:
    """Backend outcome, candidate, independent QC and declarations remain separate."""

    backend_status: Literal["succeeded", "failed"]
    method: str
    request: RegistrationRequest
    candidate: AlignmentTransform | None
    attempt: RegistrationAttempt = field(kw_only=True)
    diagnostics: dict[str, float | int | None] = field(default_factory=dict)
    qc: QCDecision | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.backend_status not in {"succeeded", "failed"}:
            raise AlignmentError("Unsupported backend outcome")
        if (self.backend_status == "succeeded") != (self.candidate is not None):
            raise AlignmentError("Backend outcome and candidate disagree")
        if not isinstance(self.attempt, RegistrationAttempt):
            raise AlignmentError("Result requires a typed registration attempt")
        if self.attempt.outcome != self.backend_status:
            raise AlignmentError("Backend outcome and attempt disagree")
        if self.backend_status == "failed":
            if self.qc is not None or self.reason is not None:
                raise AlignmentError(
                    "Backend failure uses typed failure information, not QC/reason"
                )
            if self.attempt.failure is None or self.attempt.failure.category == "qc_rejected":
                raise AlignmentError("Backend failure requires a non-QC typed failure")
        if self.attempt.qc_status is not None and (
            self.qc is None or self.attempt.qc_status != self.qc.status
        ):
            raise AlignmentError("Recorded attempt QC outcome and result QC disagree")
        if self.candidate is not None and self.request.relationship == "same_coordinate_frame":
            self.candidate.reference.validate_shared_frame(self.candidate.moving)
        if any(v is not None and not np.isfinite(v) for v in self.diagnostics.values()):
            raise AlignmentError("Backend diagnostics must be finite or null")
        if self.qc is not None:
            if self.qc.status not in {"accepted", "rejected", "insufficient_evidence"}:
                raise AlignmentError("Unsupported QC decision")
            if any(v is not None and not np.isfinite(v) for v in self.qc.metrics.values()):
                raise AlignmentError("QC metrics must be finite or null")

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(
            format=_RESULT_FORMAT,
            kind="result",
            attempt=self.attempt.to_dict(),
            backend_status=self.backend_status,
            method=self.method,
            request=asdict(self.request),
            candidate=self.candidate.to_dict() if self.candidate else None,
            diagnostics=dict(self.diagnostics),
            qc=asdict(self.qc) if self.qc else None,
            reason=self.reason,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AlignmentResult:
        _fields(
            data,
            "format kind backend_status method request candidate diagnostics qc reason attempt",
        )
        if (data["format"], data["kind"]) != (_RESULT_FORMAT, "result"):
            raise AlignmentError("Unsupported result identity")
        request = dict(data["request"])
        _fields(
            request,
            "relationship family allowed_families existing_alignment purpose "
            "diagnostic_region correspondence_evidence",
        )
        for key in ("allowed_families", "diagnostic_region", "correspondence_evidence"):
            if request[key] is not None:
                request[key] = tuple(request[key])
        qc = data["qc"]
        if qc is not None:
            _fields(qc, "status metrics missing_evidence reasons")
            qc = QCDecision(
                qc["status"],
                dict(qc["metrics"]),
                tuple(qc["missing_evidence"]),
                tuple(qc["reasons"]),
            )
        return cls(
            data["backend_status"],
            data["method"],
            RegistrationRequest(**request),
            AlignmentTransform.from_dict(data["candidate"])
            if data["candidate"] is not None
            else None,
            dict(data["diagnostics"]),
            qc,
            data["reason"],
            attempt=RegistrationAttempt.from_dict(data["attempt"]),
        )
