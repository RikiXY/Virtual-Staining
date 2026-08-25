from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch

from virtual_staining.metrics import (
    VALIDATION_IMAGE_METRIC_NAMES as _VALIDATION_IMAGE_METRIC_NAMES,
)
from virtual_staining.metrics import (
    compute_mae,
    compute_pcc_gray,
    compute_pcc_rgb,
    compute_psnr,
    compute_rmse,
    compute_ssim,
)

VALIDATION_IMAGE_METRIC_NAMES = list(_VALIDATION_IMAGE_METRIC_NAMES)


class ValidationImageMetricAccumulator:
    """Aggregates validation image metrics computed on [0, 1] NumPy arrays."""

    def __init__(self) -> None:
        self._values = {name: [] for name in VALIDATION_IMAGE_METRIC_NAMES}

    def add_batch(self, generated: torch.Tensor, target: torch.Tensor) -> None:
        for generated_image, target_image in zip(
            normalized_tensor_batch_to_images(generated),
            normalized_tensor_batch_to_images(target),
            strict=True,
        ):
            pcc_rgb = compute_pcc_rgb(target_image, generated_image)
            values = {
                "val_ssim": compute_ssim(target_image, generated_image),
                "val_mae": compute_mae(target_image, generated_image),
                "val_rmse": compute_rmse(target_image, generated_image),
                "val_psnr": compute_psnr(target_image, generated_image),
                "val_pcc_gray": compute_pcc_gray(target_image, generated_image),
                "val_pcc_rgb_mean": pcc_rgb[3],
            }
            for name, value in values.items():
                self._values[name].append(value)

    def mean(self) -> dict[str, float]:
        return {name: _finite_mean(values) for name, values in self._values.items()}


def normalized_tensor_batch_to_images(tensor: torch.Tensor) -> list[np.ndarray]:
    """Converts normalized NCHW tensors from [-1, 1] to NHWC arrays in [0, 1]."""
    if tensor.ndim != 4:
        raise ValueError("validation image metric tensors must be NCHW batches")
    images = (tensor.detach().to(device="cpu", dtype=torch.float32) * 0.5 + 0.5).clamp(0.0, 1.0)
    images = images.permute(0, 2, 3, 1).contiguous().numpy()
    return [np.asarray(image, dtype=np.float32) for image in images]


def _finite_mean(values: Iterable[float]) -> float:
    finite_values = [value for value in values if np.isfinite(value)]
    if not finite_values:
        return float("nan")
    return float(np.mean(finite_values))
