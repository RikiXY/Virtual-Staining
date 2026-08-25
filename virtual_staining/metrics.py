from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np

try:
    from skimage.metrics import structural_similarity
except ImportError as exc:
    raise ImportError(
        "Missing dependency: scikit-image. Install it with:\npip install scikit-image"
    ) from exc


@dataclass(frozen=True)
class MetricSpec:
    higher_is_better: bool
    thresholds: tuple[float, ...]


METRIC_SPECS: dict[str, MetricSpec] = {
    "ssim": MetricSpec(True, (0.85, 0.75, 0.65)),
    "psnr": MetricSpec(True, (25.0, 20.0, 15.0)),
    "mae": MetricSpec(False, (0.06, 0.10, 0.16)),
    "rmse": MetricSpec(False, (0.08, 0.12, 0.20)),
    "mse": MetricSpec(False, (0.0036, 0.0100, 0.0256)),
    "pcc_gray": MetricSpec(True, (0.95, 0.90, 0.80)),
    "pcc_rgb_mean": MetricSpec(True, (0.95, 0.90, 0.80)),
    "pcc_r": MetricSpec(True, (0.95, 0.90, 0.80)),
    "pcc_g": MetricSpec(True, (0.95, 0.90, 0.80)),
    "pcc_b": MetricSpec(True, (0.95, 0.90, 0.80)),
}

VALIDATION_IMAGE_METRIC_NAMES = (
    "val_ssim",
    "val_mae",
    "val_rmse",
    "val_psnr",
    "val_pcc_gray",
    "val_pcc_rgb_mean",
)

DEFAULT_METRICS = (
    "ssim",
    "psnr",
    "mae",
    "rmse",
    "mse",
    "pcc_rgb_mean",
    "pcc_gray",
)


def _metric_spec(metric_name: str) -> MetricSpec:
    try:
        return METRIC_SPECS[metric_name]
    except KeyError:
        raise ValueError(
            f"Unsupported metric '{metric_name}'. Supported metrics: {', '.join(METRIC_SPECS)}"
        ) from None


def is_higher_better_metric(metric_name: str) -> bool:
    """Returns True when larger values are better for a metric."""
    return _metric_spec(metric_name).higher_is_better


def get_metric_thresholds(metric_name: str) -> list[float]:
    """Returns the default thresholds used by comparison summaries."""
    return sorted(_metric_spec(metric_name).thresholds)


def compute_mae(target: np.ndarray, generated: np.ndarray) -> float:
    """Computes the Mean Absolute Error on normalised images."""
    return float(np.mean(np.abs(target - generated)))


def compute_rmse(target: np.ndarray, generated: np.ndarray) -> float:
    """Computes the Root Mean Squared Error on normalised images."""
    return float(np.sqrt(compute_mse(target, generated)))


def compute_mse(target: np.ndarray, generated: np.ndarray) -> float:
    """Computes the Mean Squared Error on normalised images."""
    return float(np.mean((target - generated) ** 2))


def compute_psnr(target: np.ndarray, generated: np.ndarray) -> float:
    """Computes the PSNR assuming normalised images in the [0,1] range."""
    mse = compute_mse(target, generated)
    if mse == 0.0:
        return float("inf")
    return float(20.0 * np.log10(1.0 / np.sqrt(mse)))


def compute_ssim(target: np.ndarray, generated: np.ndarray) -> float:
    """Computes the SSIM on normalised RGB images."""
    try:
        result = structural_similarity(target, generated, channel_axis=2, data_range=1.0)
    except TypeError:
        result = structural_similarity(target, generated, multichannel=True, data_range=1.0)
    return float(cast(float, result))


def compute_pcc(a: np.ndarray, b: np.ndarray) -> float:
    """Computes the Pearson correlation coefficient between two arrays."""
    a_flat = a.reshape(-1).astype(np.float64)
    b_flat = b.reshape(-1).astype(np.float64)
    if np.std(a_flat) == 0.0 or np.std(b_flat) == 0.0:
        return float("nan")
    return float(np.corrcoef(a_flat, b_flat)[0, 1])


def _rgb_to_gray_float(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image.astype(np.float64)
    if image.shape[2] < 3:
        return image[..., 0].astype(np.float64)
    image = image.astype(np.float64)
    return 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]


def compute_pcc_gray(target: np.ndarray, generated: np.ndarray) -> float:
    """Computes PCC after converting RGB images to grayscale."""
    return compute_pcc(_rgb_to_gray_float(target), _rgb_to_gray_float(generated))


def compute_pcc_rgb(target: np.ndarray, generated: np.ndarray) -> tuple[float, float, float, float]:
    """Computes per-channel RGB PCC and the mean across RGB channels."""
    if target.ndim != 3 or generated.ndim != 3 or target.shape[2] < 3 or generated.shape[2] < 3:
        pcc = compute_pcc(target, generated)
        return float("nan"), float("nan"), float("nan"), pcc

    pcc_r = compute_pcc(target[..., 0], generated[..., 0])
    pcc_g = compute_pcc(target[..., 1], generated[..., 1])
    pcc_b = compute_pcc(target[..., 2], generated[..., 2])
    pcc_values = np.array([pcc_r, pcc_g, pcc_b], dtype=np.float64)
    pcc_rgb_mean = float("nan") if np.isnan(pcc_values).all() else float(np.nanmean(pcc_values))
    return pcc_r, pcc_g, pcc_b, pcc_rgb_mean
