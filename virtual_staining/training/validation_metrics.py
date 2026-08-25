from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch

from virtual_staining.metrics import (
    VALIDATION_IMAGE_METRIC_NAMES as _VALIDATION_IMAGE_METRIC_NAMES,
)
from virtual_staining.metrics import (
    VALIDATION_METRIC_TO_BASE,
    compute_standard_metrics,
)
from virtual_staining.models.io_contract import denormalize_model_output

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
            metrics = compute_standard_metrics(target_image, generated_image)
            values = {
                name: metrics[base_name] for name, base_name in VALIDATION_METRIC_TO_BASE.items()
            }
            for name, value in values.items():
                self._values[name].append(value)

    def mean(self) -> dict[str, float]:
        return {name: _finite_mean(values) for name, values in self._values.items()}


def normalized_tensor_batch_to_images(tensor: torch.Tensor) -> list[np.ndarray]:
    """Converts normalized NCHW tensors from [-1, 1] to NHWC arrays in [0, 1]."""
    if tensor.ndim != 4:
        raise ValueError("validation image metric tensors must be NCHW batches")
    images = denormalize_model_output(tensor.detach().to(device="cpu", dtype=torch.float32))
    images = images.permute(0, 2, 3, 1).contiguous().numpy()
    return [np.asarray(image, dtype=np.float32) for image in images]


def _finite_mean(values: Iterable[float]) -> float:
    finite_values = [value for value in values if np.isfinite(value)]
    if not finite_values:
        return float("nan")
    return float(np.mean(finite_values))
