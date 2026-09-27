from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
import torch

from virtual_staining.metrics import ResolvedMetric, compute_metrics
from virtual_staining.models.io_contract import denormalize_model_output


class ValidationImageMetricAccumulator:
    """Aggregates method-selected image metrics computed on [0, 1] NumPy arrays.

    ``metrics`` maps each validation column the method reports to the evaluation metric
    it reuses. Non-finite per-image results (infinite, undefined, unavailable) are left
    out of the finite mean.
    """

    def __init__(self, metrics: Mapping[str, ResolvedMetric]) -> None:
        self._metrics = dict(metrics)
        self._requested = tuple(self._metrics.values())
        self._values: dict[str, list[float]] = {name: [] for name in self._metrics}

    def add_batch(self, generated: torch.Tensor, target: torch.Tensor) -> None:
        if not self._metrics:
            return
        for generated_image, target_image in zip(
            _normalized_tensor_batch_to_images(generated),
            _normalized_tensor_batch_to_images(target),
            strict=True,
        ):
            results = compute_metrics(self._requested, target_image, generated_image)
            for name, metric in self._metrics.items():
                value = results[metric.name].value
                self._values[name].append(float("nan") if value is None else value)

    def mean(self) -> dict[str, float]:
        return {name: _finite_mean(values) for name, values in self._values.items()}


def _normalized_tensor_batch_to_images(tensor: torch.Tensor) -> list[np.ndarray]:
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
