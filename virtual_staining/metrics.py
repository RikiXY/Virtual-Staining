from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import numpy as np

try:
    from skimage.metrics import structural_similarity
except ImportError as exc:
    raise ImportError(
        "Missing dependency: scikit-image. Install it with:\npip install scikit-image"
    ) from exc


MetricEvaluator = Callable[[np.ndarray, np.ndarray], dict[str, float]]


@dataclass(frozen=True)
class MetricSpec:
    higher_is_better: bool
    thresholds: tuple[float, ...]
    default: bool = False
    validation_name: str | None = None
    evaluator_group: MetricEvaluator | None = None
    report_order: int | None = None


def _evaluate_image_quality(target: np.ndarray, generated: np.ndarray) -> dict[str, float]:
    return {"ssim": compute_ssim(target, generated)}


def _evaluate_shared_error(target: np.ndarray, generated: np.ndarray) -> dict[str, float]:
    mse = compute_mse(target, generated)
    return {
        "mae": compute_mae(target, generated),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "psnr": float("inf") if mse == 0.0 else float(20.0 * np.log10(1.0 / np.sqrt(mse))),
    }


def _evaluate_pcc(target: np.ndarray, generated: np.ndarray) -> dict[str, float]:
    pcc_r, pcc_g, pcc_b, pcc_rgb_mean = compute_pcc_rgb(target, generated)
    return {
        "pcc_gray": compute_pcc_gray(target, generated),
        "pcc_r": pcc_r,
        "pcc_g": pcc_g,
        "pcc_b": pcc_b,
        "pcc_rgb_mean": pcc_rgb_mean,
    }


METRIC_SPECS: dict[str, MetricSpec] = {
    "ssim": MetricSpec(True, (0.85, 0.75, 0.65), True, "val_ssim", _evaluate_image_quality, 4),
    "psnr": MetricSpec(True, (25.0, 20.0, 15.0), True, "val_psnr", _evaluate_shared_error, 3),
    "mae": MetricSpec(False, (0.06, 0.10, 0.16), True, "val_mae", _evaluate_shared_error, 0),
    "rmse": MetricSpec(False, (0.08, 0.12, 0.20), True, "val_rmse", _evaluate_shared_error, 2),
    "mse": MetricSpec(False, (0.0036, 0.0100, 0.0256), True, None, _evaluate_shared_error, 1),
    "pcc_rgb_mean": MetricSpec(
        True, (0.95, 0.90, 0.80), True, "val_pcc_rgb_mean", _evaluate_pcc, 9
    ),
    "pcc_gray": MetricSpec(True, (0.95, 0.90, 0.80), True, "val_pcc_gray", _evaluate_pcc, 5),
    "pcc_r": MetricSpec(True, (0.95, 0.90, 0.80), False, None, _evaluate_pcc, 6),
    "pcc_g": MetricSpec(True, (0.95, 0.90, 0.80), False, None, _evaluate_pcc, 7),
    "pcc_b": MetricSpec(True, (0.95, 0.90, 0.80), False, None, _evaluate_pcc, 8),
}
METRIC_NAMES = tuple(METRIC_SPECS)
DEFAULT_METRICS = tuple(name for name, spec in METRIC_SPECS.items() if spec.default)
VALIDATION_IMAGE_METRIC_NAMES = tuple(
    spec.validation_name for spec in METRIC_SPECS.values() if spec.validation_name is not None
)
VALIDATION_METRIC_TO_BASE = {
    spec.validation_name: name
    for name, spec in METRIC_SPECS.items()
    if spec.validation_name is not None
}
REPORT_METRIC_NAMES = tuple(
    sorted(METRIC_SPECS, key=lambda name: METRIC_SPECS[name].report_order or 0)
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


def compute_standard_metrics(target: np.ndarray, generated: np.ndarray) -> dict[str, float]:
    evaluators: list[MetricEvaluator] = []
    for spec in METRIC_SPECS.values():
        if spec.evaluator_group is None:
            raise ValueError("Metric spec is missing an evaluator group")
        if spec.evaluator_group not in evaluators:
            evaluators.append(spec.evaluator_group)

    computed: dict[str, float] = {}
    for evaluator in evaluators:
        output = evaluator(target, generated)
        overlap = computed.keys() & output.keys()
        if overlap:
            raise ValueError(f"Metric evaluator keys overlap: {sorted(overlap)}")
        computed.update(output)

    expected = set(METRIC_SPECS)
    if set(computed) != expected:
        raise ValueError(
            "Metric evaluator output keys do not match registry: "
            f"expected {sorted(expected)}, got {sorted(computed)}"
        )
    return {name: computed[name] for name in METRIC_SPECS}
