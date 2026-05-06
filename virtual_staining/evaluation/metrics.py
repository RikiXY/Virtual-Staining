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
    return float(np.sqrt(compute_mse(target, generated)))


def compute_mse(target: np.ndarray, generated: np.ndarray) -> float:
    """Computes the Mean Squared Error on normalised images."""
    return float(np.mean((target - generated) ** 2))


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


def compute_pcc(a: np.ndarray, b: np.ndarray) -> float:
    """Computes the Pearson correlation coefficient between two arrays."""
    a_flat = a.reshape(-1).astype(np.float64)
    b_flat = b.reshape(-1).astype(np.float64)

    if np.std(a_flat) == 0.0 or np.std(b_flat) == 0.0:
        return float("nan")

    return float(np.corrcoef(a_flat, b_flat)[0, 1])


def rgb_to_gray_float(image: np.ndarray) -> np.ndarray:
    """Converts an RGB image to grayscale using standard luminance weights."""
    if image.ndim == 2:
        return image.astype(np.float64)

    if image.shape[2] < 3:
        return image[..., 0].astype(np.float64)

    image = image.astype(np.float64)
    return 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]


def compute_pcc_gray(target: np.ndarray, generated: np.ndarray) -> float:
    """Computes PCC after converting RGB images to grayscale."""
    target_gray = rgb_to_gray_float(target)
    generated_gray = rgb_to_gray_float(generated)
    return compute_pcc(target_gray, generated_gray)


def compute_pcc_rgb(
    target: np.ndarray, generated: np.ndarray
) -> tuple[float, float, float, float]:
    """Computes per-channel RGB PCC and the mean across RGB channels."""
    if (
        target.ndim != 3
        or generated.ndim != 3
        or target.shape[2] < 3
        or generated.shape[2] < 3
    ):
        pcc = compute_pcc(target, generated)
        return float("nan"), float("nan"), float("nan"), pcc

    pcc_r = compute_pcc(target[..., 0], generated[..., 0])
    pcc_g = compute_pcc(target[..., 1], generated[..., 1])
    pcc_b = compute_pcc(target[..., 2], generated[..., 2])
    pcc_values = np.array([pcc_r, pcc_g, pcc_b], dtype=np.float64)
    pcc_rgb_mean = (
        float("nan") if np.isnan(pcc_values).all() else float(np.nanmean(pcc_values))
    )
    return pcc_r, pcc_g, pcc_b, pcc_rgb_mean


def evaluate_pair(
    target_path: str | Path,
    generated_path: str | Path,
) -> tuple[dict[str, float], tuple[int, int, int]]:
    """Computes the standard metrics for a target/generated pair."""
    target = load_rgb_image(target_path)
    generated = load_rgb_image(generated_path)

    validate_same_shape(target, generated)
    shape = target.shape

    target_float = to_float01(target)
    generated_float = to_float01(generated)
    mse = compute_mse(target_float, generated_float)
    pcc_r, pcc_g, pcc_b, pcc_rgb_mean = compute_pcc_rgb(target_float, generated_float)

    metrics = {
        "mae": compute_mae(target_float, generated_float),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "psnr": compute_psnr(target_float, generated_float),
        "ssim": compute_ssim(target_float, generated_float),
        "pcc_gray": compute_pcc_gray(target_float, generated_float),
        "pcc_r": pcc_r,
        "pcc_g": pcc_g,
        "pcc_b": pcc_b,
        "pcc_rgb_mean": pcc_rgb_mean,
    }

    return metrics, shape
