"""Identity/SIFT registration and affine warping in explicit image coordinates."""

from virtual_staining.data.alignment.models import (
    AlignmentError,
    AlignmentImage,
    AlignmentResult,
    RegistrationDiagnostics,
)
from virtual_staining.data.alignment.registration import identity_alignment, resolve_alignment
from virtual_staining.data.alignment.warping import warp_aligned_mask_patch, warp_aligned_patch

__all__ = [
    "AlignmentError",
    "AlignmentImage",
    "AlignmentResult",
    "RegistrationDiagnostics",
    "identity_alignment",
    "resolve_alignment",
    "warp_aligned_mask_patch",
    "warp_aligned_patch",
]
