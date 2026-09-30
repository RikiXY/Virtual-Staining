from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.ndimage import map_coordinates

from virtual_staining.data.alignment.models import (
    AlignmentError,
    AlignmentTransform,
    GridGeometry,
    SpatialEvidence,
    _shape,
)


@dataclass(frozen=True)
class WarpedPatch:
    image: np.ndarray
    geometric_validity: np.ndarray
    observation_validity: np.ndarray | None
    observation_known: np.ndarray | None
    tissue_support: np.ndarray | None
    tissue_known: np.ndarray | None


def _coordinates(matrix: np.ndarray, x: int, y: int, width: int, height: int) -> np.ndarray:
    rows, columns = np.mgrid[y : y + height, x : x + width]
    return np.stack((columns, rows), axis=-1) @ matrix[:2, :2].T + matrix[:2, 2]


def _contributors(points: np.ndarray, linear: bool) -> list[np.ndarray]:
    if not linear:
        return [np.floor(points + 0.5)]
    low, high = np.floor(points), np.ceil(points)
    return [
        low,
        high,
        np.stack((low[..., 0], high[..., 1]), axis=-1),
        np.stack((high[..., 0], low[..., 1]), axis=-1),
    ]


def _inside(points: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    return (
        (points[..., 0] >= 0)
        & (points[..., 0] < shape[1])
        & (points[..., 1] >= 0)
        & (points[..., 1] < shape[0])
    )


def _sample_evidence(
    evidence: SpatialEvidence, native_points: np.ndarray, *, conservative: bool
) -> tuple[np.ndarray, np.ndarray]:
    inverse = np.linalg.inv(evidence.grid.grid_to_level0)
    points = native_points @ inverse[:2, :2].T + inverse[:2, 2]
    known = np.ones(points.shape[:-1], dtype=bool)
    values = np.ones_like(known)
    for contributor in _contributors(points, conservative):
        inside = _inside(contributor, evidence.grid.shape)
        col = np.clip(contributor[..., 0], 0, evidence.grid.shape[1] - 1).astype(int)
        row = np.clip(contributor[..., 1], 0, evidence.grid.shape[0] - 1).astype(int)
        known &= inside
        values &= inside & evidence.values[row, col]
    return values, known


def _resample(
    image: np.ndarray | Callable[[int, int, int, int], np.ndarray],
    source_shape: tuple[int, int],
    inverse: np.ndarray,
    *,
    x: int,
    y: int,
    output_size: tuple[int, int],
    interpolation: Literal["linear", "nearest"],
    max_source_pixels: int,
) -> np.ndarray:
    """Single resampling path for arrays and borrowed bounded region readers."""
    width, height = output_size
    corners = np.array(
        [[x, y], [x + width - 1, y], [x, y + height - 1], [x + width - 1, y + height - 1]],
        dtype=np.float64,
    )
    mapped = corners @ inverse[:2, :2].T + inverse[:2, 2]
    linear = interpolation == "linear"
    lower = np.floor(mapped.min(axis=0) if linear else mapped.min(axis=0) + 0.5)
    upper = np.ceil(mapped.max(axis=0)) if linear else np.floor(mapped.max(axis=0) + 0.5)
    # Clip before integer conversion; no out-of-slide reader padding is trusted.
    rx, ry = np.clip(lower, 0, [source_shape[1] - 1, source_shape[0] - 1]).astype(int)
    ex, ey = np.clip(upper + 1, [rx + 1, ry + 1], [source_shape[1], source_shape[0]]).astype(int)
    rw, rh = int(ex - rx), int(ey - ry)
    if rw * rh > max_source_pixels:
        if width == height == 1:
            raise AlignmentError("Source-region budget cannot hold one interpolation footprint")
        split_x = width >= height and width > 1
        first = width // 2 if split_x else height // 2
        sizes = (
            ((first, height), (width - first, height))
            if split_x
            else ((width, first), (width, height - first))
        )
        left = _resample(
            image,
            source_shape,
            inverse,
            x=x,
            y=y,
            output_size=sizes[0],
            interpolation=interpolation,
            max_source_pixels=max_source_pixels,
        )
        right = _resample(
            image,
            source_shape,
            inverse,
            x=x + first if split_x else x,
            y=y if split_x else y + first,
            output_size=sizes[1],
            interpolation=interpolation,
            max_source_pixels=max_source_pixels,
        )
        return np.concatenate((left, right), axis=1 if split_x else 0)
    region = image(int(rx), int(ry), rw, rh) if callable(image) else image[ry:ey, rx:ex]
    if region.shape[:2] != (rh, rw) or region.ndim not in (2, 3):
        raise AlignmentError("Region reader returned incompatible geometry")
    points = _coordinates(inverse, x, y, width, height)
    if not linear:
        nearest = np.floor(points + 0.5)
        inside = _inside(nearest, source_shape)
        columns = np.clip(nearest[..., 0] - rx, 0, rw - 1).astype(int)
        rows = np.clip(nearest[..., 1] - ry, 0, rh - 1).astype(int)
        sampled = region[rows, columns].copy()
        sampled[~inside] = 0
        return sampled
    points -= [rx, ry]
    coordinates = np.stack((points[..., 1], points[..., 0]))

    def sample(channel: np.ndarray) -> np.ndarray:
        return map_coordinates(
            channel, coordinates, order=1, mode="grid-constant", cval=255, prefilter=False
        )

    return (
        sample(region)
        if region.ndim == 2
        else np.stack([sample(region[..., channel]) for channel in range(region.shape[2])], axis=-1)
    )


def warp_aligned_patch(
    image: np.ndarray | Callable[[int, int, int, int], np.ndarray],
    transform: AlignmentTransform,
    *,
    x: int,
    y: int,
    output_size: tuple[int, int],
    max_source_pixels: int,
    interpolation: Literal["linear", "nearest"] = "linear",
    tissue_support: SpatialEvidence | None = None,
    observation_validity: SpatialEvidence | None = None,
) -> WarpedPatch:
    """Inverse-map a half-open reference index box; read no more than the supplied budget.

    Linear images use a white border; nearest labels/masks use zero. Geometric
    validity covers every contributing source pixel. Supplied observation validity
    additionally covers every contributor and its evidence footprint. Unknown evidence
    is represented by a separate known mask; absent maps return None.
    """
    width, height = output_size
    _shape((height, width))
    if type(x) is not int or type(y) is not int:
        raise AlignmentError("Patch origins must be integer pixel centres")
    if type(max_source_pixels) is not int or max_source_pixels <= 0:
        raise AlignmentError("A positive integer source-region budget is required")
    if interpolation not in {"linear", "nearest"}:
        raise AlignmentError("Unsupported interpolation policy")
    if not callable(image) and image.shape[:2] != transform.moving.shape:
        raise AlignmentError("Source array does not match moving level-0 geometry")
    for evidence, kind in (
        (tissue_support, "tissue_support"),
        (observation_validity, "observation_validity"),
    ):
        if evidence is not None and (evidence.asset != transform.moving or evidence.kind != kind):
            raise AlignmentError("Evidence kind/asset does not match moving geometry")
    inverse = np.linalg.inv(transform.matrix)
    result = _resample(
        image,
        transform.moving.shape,
        inverse,
        x=x,
        y=y,
        output_size=output_size,
        interpolation=interpolation,
        max_source_pixels=max_source_pixels,
    )
    points = _coordinates(inverse, x, y, width, height)
    geometric = np.ones((height, width), dtype=bool)
    validity = np.ones_like(geometric) if observation_validity is not None else None
    known = np.ones_like(geometric) if observation_validity is not None else None
    for contributor in _contributors(points, interpolation == "linear"):
        inside = _inside(contributor, transform.moving.shape)
        geometric &= inside
        if observation_validity is not None:
            values, available = _sample_evidence(
                observation_validity, contributor, conservative=True
            )
            assert validity is not None and known is not None
            validity &= values & inside
            known &= available & inside
    support, support_known = (
        (None, None)
        if tissue_support is None
        else _sample_evidence(tissue_support, points, conservative=False)
    )
    if support_known is not None:
        support_known &= geometric
    return WarpedPatch(result, geometric, validity, known, support, support_known)


def warp_aligned_mask_patch(
    mask: np.ndarray,
    transform: AlignmentTransform,
    grid: GridGeometry,
    *,
    x: int,
    y: int,
    output_size: tuple[int, int],
    max_source_pixels: int,
) -> np.ndarray:
    """Nearest foreground/loss-mask resampling; this does not create tissue evidence."""
    if mask.ndim != 2 or mask.shape != grid.shape:
        raise AlignmentError("Mask does not match its explicit grid")
    _shape((output_size[1], output_size[0]))
    if type(max_source_pixels) is not int or max_source_pixels <= 0:
        raise AlignmentError("A positive source-region budget is required")
    inverse = np.linalg.inv(transform.matrix @ grid.grid_to_level0)
    return _resample(
        mask,
        grid.shape,
        inverse,
        x=x,
        y=y,
        output_size=output_size,
        interpolation="nearest",
        max_source_pixels=max_source_pixels,
    )
