from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np

try:
    from skimage.metrics import structural_similarity
except ImportError as exc:
    raise ImportError(
        "Missing dependency: scikit-image. Install it with:\npip install scikit-image"
    ) from exc

from virtual_staining.utils.image_io import load_rgb_image, to_float01


def validate_same_shape(target: np.ndarray, generated: np.ndarray) -> None:
    """Verifies that target and generated have exactly the same shape."""
    if target.shape != generated.shape:
        raise ValueError(
            "Target and generated images must have the same shape. "
            f"Got {target.shape} and {generated.shape}."
        )


def compute_mae(target: np.ndarray, generated: np.ndarray) -> float:
    """Computes the Mean Absolute Error on normalised images."""
    return float(np.mean(np.abs(target - generated)))


def compute_rmse(target: np.ndarray, generated: np.ndarray) -> float:
    """Computes the Root Mean Squared Error on normalised images."""
    return float(np.sqrt(np.mean((target - generated) ** 2)))


def compute_psnr(target: np.ndarray, generated: np.ndarray) -> float:
    """Computes the PSNR assuming normalised images in the [0,1] range."""
    mse = float(np.mean((target - generated) ** 2))

    if mse == 0.0:
        return float("inf")

    return float(20.0 * np.log10(1.0 / np.sqrt(mse)))


def compute_ssim(target: np.ndarray, generated: np.ndarray) -> float:
    """Computes the SSIM on normalised RGB images."""
    try:
        result = structural_similarity(
            target,
            generated,
            channel_axis=2,
            data_range=1.0,
        )
        return float(cast(float, result))
    except TypeError:
        result = structural_similarity(
            target,
            generated,
            multichannel=True,
            data_range=1.0,
        )
        return float(cast(float, result))


def evaluate_pair(
    target_path: str | Path,
    generated_path: str | Path,
) -> tuple[dict[str, float], tuple[int, int, int]]:
    """Computes the four metrics for a target/generated pair."""
    target = load_rgb_image(target_path)
    generated = load_rgb_image(generated_path)

    validate_same_shape(target, generated)
    shape = target.shape

    target_float = to_float01(target)
    generated_float = to_float01(generated)

    metrics = {
        "mae": compute_mae(target_float, generated_float),
        "rmse": compute_rmse(target_float, generated_float),
        "psnr": compute_psnr(target_float, generated_float),
        "ssim": compute_ssim(target_float, generated_float),
    }

    return metrics, shape
