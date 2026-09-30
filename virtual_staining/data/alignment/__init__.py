"""Direct registration geometry, independent QC, and bounded inverse resampling."""

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
    RegistrationResources,
    RegistrationRuntime,
    SpatialEvidence,
)
from virtual_staining.data.alignment.registration import (
    evaluate_alignment_qc,
    identity_alignment,
    resolve_alignment,
)
from virtual_staining.data.alignment.warping import (
    WarpedPatch,
    warp_aligned_mask_patch,
    warp_aligned_patch,
)

__all__ = [
    "AlignmentError",
    "AlignmentImage",
    "AlignmentResult",
    "AlignmentTransform",
    "FailureCategory",
    "GridGeometry",
    "ImageGeometry",
    "QCDecision",
    "QCPolicy",
    "RegistrationAttempt",
    "RegistrationFailure",
    "RegistrationRequest",
    "RegistrationResources",
    "RegistrationRuntime",
    "SpatialEvidence",
    "WarpedPatch",
    "evaluate_alignment_qc",
    "identity_alignment",
    "resolve_alignment",
    "warp_aligned_mask_patch",
    "warp_aligned_patch",
]
