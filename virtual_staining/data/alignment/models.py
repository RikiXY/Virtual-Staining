from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np

Family = Literal["identity", "similarity", "affine"]
_FAMILIES = {"identity", "similarity", "affine"}
_FORMAT = "virtual_staining.alignment/1"


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


@dataclass(frozen=True)
class AlignmentResult:
    """Backend outcome, candidate, independent QC and declarations remain separate."""

    backend_status: Literal["succeeded", "failed"]
    method: str
    request: RegistrationRequest
    candidate: AlignmentTransform | None
    diagnostics: dict[str, float | int | None] = field(default_factory=dict)
    qc: QCDecision | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.backend_status not in {"succeeded", "failed"}:
            raise AlignmentError("Unsupported backend outcome")
        if (self.backend_status == "succeeded") != (self.candidate is not None):
            raise AlignmentError("Backend outcome and candidate disagree")
        if self.backend_status == "failed" and (self.qc is not None or not self.reason):
            raise AlignmentError("Backend failure requires a reason and has no QC decision")
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
            format=_FORMAT,
            kind="result",
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
        _fields(data, "format kind backend_status method request candidate diagnostics qc reason")
        if (data["format"], data["kind"]) != (_FORMAT, "result"):
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
        )
