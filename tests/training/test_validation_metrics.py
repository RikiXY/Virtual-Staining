from __future__ import annotations

import math

import torch

from virtual_staining.methods.builtin import PIX2PIX_VALIDATION_METRICS
from virtual_staining.training.validation_metrics import ValidationImageMetricAccumulator


def test_accumulator_reports_method_selected_metrics_with_finite_means() -> None:
    accumulator = ValidationImageMetricAccumulator(PIX2PIX_VALIDATION_METRICS)
    constant = torch.zeros(2, 3, 8, 8)

    accumulator.add_batch(constant, constant)
    means = accumulator.mean()

    assert list(means) == list(PIX2PIX_VALIDATION_METRICS)
    assert means["val_mae"] == 0.0
    assert means["val_ssim"] == 1.0
    # Identical images: PSNR is positive infinity and PCC undefined, so no finite mean.
    assert math.isnan(means["val_psnr"]) and math.isnan(means["val_pcc_gray"])


def test_accumulator_without_metrics_reports_nothing() -> None:
    accumulator = ValidationImageMetricAccumulator({})
    accumulator.add_batch(torch.zeros(1, 3, 8, 8), torch.zeros(1, 3, 8, 8))

    assert accumulator.mean() == {}
