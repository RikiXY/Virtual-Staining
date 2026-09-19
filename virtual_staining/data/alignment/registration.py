from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import cv2
import numpy as np

from virtual_staining.config.data import AlignmentConfig
from virtual_staining.data.alignment.models import (
    AlignmentError,
    AlignmentImage,
    AlignmentResult,
    RegistrationDiagnostics,
    _validate_affine,
)
from virtual_staining.data.alignment.warping import (
    _rescale_transform,
    _validate_mask_geometry,
    _warp_image,
)

_MIN_INLIERS = 12
_MIN_INLIER_RATIO = 0.10
_LOWE_RATIO_THRESHOLD = 0.75
_RANSAC_REPROJECTION_THRESHOLD = 5.0
_RANSAC_MAX_ITERS = 2000
_RANSAC_CONFIDENCE = 0.99
_RANSAC_REFINE_ITERS = 10


def _affine_diagnostics(warp_matrix: np.ndarray) -> dict[str, float]:
    a, b, tx = warp_matrix[0]
    c, d, ty = warp_matrix[1]
    return {
        "scale_x": float(np.sqrt(a * a + c * c)),
        "scale_y": float(np.sqrt(b * b + d * d)),
        "rotation_deg": float(np.degrees(np.arctan2(c, a))),
        "translation_x": float(tx),
        "translation_y": float(ty),
    }


def _ratio_test_matches(
    knn_matches: Sequence[Sequence[cv2.DMatch]],
    *,
    ratio_threshold: float = _LOWE_RATIO_THRESHOLD,
) -> list[cv2.DMatch]:
    good_matches = []
    for candidates in knn_matches:
        if len(candidates) < 2:
            continue
        best, second_best = candidates[0], candidates[1]
        if best.distance < ratio_threshold * second_best.distance:
            good_matches.append(best)
    return good_matches


def _aligned_mask_iou(
    reference_mask: np.ndarray | None,
    moving_mask: np.ndarray | None,
    warp_matrix: np.ndarray,
    output_size: tuple[int, int],
) -> float | None:
    if reference_mask is None or moving_mask is None:
        return None

    aligned_moving_mask = _warp_image(moving_mask, warp_matrix, output_size, is_mask=True)
    foreground_1 = reference_mask > 0
    foreground_2 = aligned_moving_mask > 0
    union = np.logical_or(foreground_1, foreground_2)
    union_count = int(np.count_nonzero(union))
    if union_count == 0:
        return None
    intersection_count = int(np.count_nonzero(np.logical_and(foreground_1, foreground_2)))
    return intersection_count / union_count


def _estimate_affine(
    reference: np.ndarray,
    moving: np.ndarray,
    reference_mask: np.ndarray | None = None,
    moving_mask: np.ndarray | None = None,
    nfeatures: int = 10000,
    ratio_threshold: float = _LOWE_RATIO_THRESHOLD,
) -> tuple[np.ndarray, RegistrationDiagnostics]:
    for name, image, mask in (
        ("reference_mask", reference, reference_mask),
        ("moving_mask", moving, moving_mask),
    ):
        if mask is not None and mask.shape != image.shape[:2]:
            raise AlignmentError(
                f"{name} geometry must match image: expected {image.shape[:2]}, got {mask.shape}"
            )
        if mask is not None:
            _validate_mask_geometry(mask, image.shape[:2], name=name)
    clahe = cv2.createCLAHE(clipLimit=18.0, tileGridSize=(8, 8))
    reference_clahe = reference
    moving_clahe = moving

    if len(reference_clahe.shape) == 3:
        reference_clahe = cv2.cvtColor(reference_clahe, cv2.COLOR_BGR2GRAY)
    if len(moving_clahe.shape) == 3:
        moving_clahe = cv2.cvtColor(moving_clahe, cv2.COLOR_BGR2GRAY)

    reference_clahe = clahe.apply(reference_clahe)
    moving_clahe = clahe.apply(moving_clahe)

    sift = cv2.SIFT_create(nfeatures=nfeatures)  # type: ignore[attr-defined]
    reference_keypoints, reference_descriptors = sift.detectAndCompute(
        reference_clahe, reference_mask
    )
    moving_keypoints, moving_descriptors = sift.detectAndCompute(moving_clahe, moving_mask)

    n_reference_keypoints = len(reference_keypoints)
    n_moving_keypoints = len(moving_keypoints)
    if (
        n_reference_keypoints < 4
        or n_moving_keypoints < 4
        or reference_descriptors is None
        or moving_descriptors is None
    ):
        raise AlignmentError(
            "Not enough features for alignment: "
            f"reference={n_reference_keypoints}, moving={n_moving_keypoints}, minimum=4"
        )

    bf = cv2.BFMatcher(cv2.NORM_L2)
    knn_matches = bf.knnMatch(reference_descriptors, moving_descriptors, k=2)
    filtered_matches = _ratio_test_matches(knn_matches, ratio_threshold=ratio_threshold)

    if len(filtered_matches) < 4:
        raise AlignmentError(
            "Not enough good descriptor matches for alignment after ratio test: "
            f"good={len(filtered_matches)}, minimum=4, ratio_threshold={ratio_threshold}"
        )

    reference_points = np.asarray(
        [reference_keypoints[match.queryIdx].pt for match in filtered_matches],
        dtype=np.float32,
    ).reshape(-1, 1, 2)
    moving_points = np.asarray(
        [moving_keypoints[match.trainIdx].pt for match in filtered_matches],
        dtype=np.float32,
    ).reshape(-1, 1, 2)

    warp_matrix, inlier_mask = cv2.estimateAffinePartial2D(
        moving_points,
        reference_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=_RANSAC_REPROJECTION_THRESHOLD,
        maxIters=_RANSAC_MAX_ITERS,
        confidence=_RANSAC_CONFIDENCE,
        refineIters=_RANSAC_REFINE_ITERS,
    )

    if warp_matrix is None:
        raise AlignmentError(
            "Affine estimation failed after ratio-test matching and RANSAC: "
            "cv2.estimateAffinePartial2D returned None"
        )
    _validate_affine(warp_matrix)

    n_inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
    inlier_ratio = n_inliers / len(filtered_matches)
    if n_inliers < _MIN_INLIERS:
        raise AlignmentError(
            f"Alignment rejected: only {n_inliers} inliers found (minimum {_MIN_INLIERS} required)"
        )
    if inlier_ratio < _MIN_INLIER_RATIO:
        raise AlignmentError(
            "Alignment rejected: inlier ratio "
            f"{inlier_ratio:.3f} is below minimum {_MIN_INLIER_RATIO:.3f} "
            f"({n_inliers}/{len(filtered_matches)} inliers)"
        )

    diagnostics = _affine_diagnostics(warp_matrix)
    mask_iou = _aligned_mask_iou(
        reference_mask,
        moving_mask,
        warp_matrix,
        (reference.shape[1], reference.shape[0]),
    )
    metadata = RegistrationDiagnostics(
        n_keypoints_reference=n_reference_keypoints,
        n_keypoints_moving=n_moving_keypoints,
        n_matches=len(filtered_matches),
        n_inliers=n_inliers,
        inlier_ratio=inlier_ratio,
        scale_x=diagnostics["scale_x"],
        scale_y=diagnostics["scale_y"],
        rotation_deg=diagnostics["rotation_deg"],
        translation_x=diagnostics["translation_x"],
        translation_y=diagnostics["translation_y"],
        mask_iou=mask_iou,
    )

    return warp_matrix, metadata


def identity_alignment(reason: str = "reference") -> AlignmentResult:
    return AlignmentResult("identity", np.eye(2, 3, dtype=np.float64), reason=reason)


def _validate_identity(reference: AlignmentImage, moving: AlignmentImage) -> None:
    if reference.full_shape != moving.full_shape:
        raise AlignmentError(f"identity alignment requires equal geometry for {moving.name}")
    for name, left, right in zip(("mpp_x", "mpp_y"), reference.mpp, moving.mpp, strict=True):
        if left is not None and right is not None and not np.isclose(left, right, rtol=0.01):
            raise AlignmentError(f"identity alignment has incompatible {name} for {moving.name}")


def _registration_preview(image: AlignmentImage) -> tuple[np.ndarray, np.ndarray]:
    preview = image.preview
    if (
        preview.dtype != np.uint8
        or preview.ndim not in (2, 3)
        or (preview.ndim == 3 and preview.shape[2] != 3)
        or min(preview.shape[:2]) < 2
    ):
        raise AlignmentError(f"Invalid preview for {image.name}: expected uint8 BGR or grayscale")
    assert image.mask is not None
    _validate_mask_geometry(image.mask, image.full_shape, name=f"{image.name} mask")
    mask = image.mask
    if mask.shape != preview.shape[:2]:
        mask = cv2.resize(
            mask, (preview.shape[1], preview.shape[0]), interpolation=cv2.INTER_NEAREST
        )
    return (
        cv2.resize(preview, None, fx=0.5, fy=0.5),
        cv2.resize(mask, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST),
    )


def resolve_alignment(
    reference: AlignmentImage,
    moving: AlignmentImage,
    policy: AlignmentConfig,
    *,
    already_aligned: bool | None = None,
) -> AlignmentResult:
    estimate = already_aligned is not True and (
        already_aligned is False or policy.mode in {"auto", "always"}
    )
    if policy.mode == "never" and already_aligned is False:
        raise AlignmentError(
            f"alignment.mode=never contradicts already_aligned=false for {moving.name}"
        )
    if not estimate:
        if policy.validate_declared:
            _validate_identity(reference, moving)
        return identity_alignment("declared_aligned" if already_aligned is True else "policy_never")
    if reference.mask is None or moving.mask is None:
        raise AlignmentError(f"affine registration requires masks for {moving.name}")
    reference_preview, reference_mask = _registration_preview(reference)
    moving_preview, moving_mask = _registration_preview(moving)
    matrix, diagnostics = _estimate_affine(
        reference_preview, moving_preview, reference_mask, moving_mask
    )
    matrix = _rescale_transform(
        matrix,
        reference_scale=(
            reference_preview.shape[1] / reference.full_shape[1],
            reference_preview.shape[0] / reference.full_shape[0],
        ),
        moving_scale=(
            moving_preview.shape[1] / moving.full_shape[1],
            moving_preview.shape[0] / moving.full_shape[0],
        ),
    )
    diagnostics = replace(diagnostics, **_affine_diagnostics(matrix))
    return AlignmentResult("affine_sift", matrix, diagnostics=diagnostics)
