from __future__ import annotations

import math

import torch

from virtual_staining.methods.builtin import PIX2PIX_IMAGE_METRICS, pix2pix_validation_metrics
from virtual_staining.training.validation_metrics import ValidationImageMetricAccumulator


def test_accumulator_reports_method_selected_metrics_with_finite_means() -> None:
    columns = {
        column: metric for column, (_, metric) in pix2pix_validation_metrics(("HE",)).items()
    }
    accumulator = ValidationImageMetricAccumulator(columns)
    constant = torch.zeros(2, 3, 8, 8)

    accumulator.add_batch(constant, constant)
    means = accumulator.mean()

    assert list(means) == [f"val_{name}__HE" for name in PIX2PIX_IMAGE_METRICS]
    assert means["val_mae__HE"] == 0.0
    assert means["val_ssim__HE"] == 1.0
    # Identical images: PSNR is positive infinity and PCC undefined, so no finite mean.
    assert math.isnan(means["val_psnr__HE"]) and math.isnan(means["val_pcc_gray__HE"])


def test_validation_columns_are_per_output_and_never_pooled() -> None:
    columns = pix2pix_validation_metrics(("PAS", "HE"))

    assert list(columns)[:2] == ["val_ssim__PAS", "val_ssim__HE"]
    assert {output for output, _ in columns.values()} == {"PAS", "HE"}
    assert not any(column.startswith("val_ssim") and "__" not in column for column in columns)


def test_accumulator_without_metrics_reports_nothing() -> None:
    accumulator = ValidationImageMetricAccumulator({})
    accumulator.add_batch(torch.zeros(1, 3, 8, 8), torch.zeros(1, 3, 8, 8))

    assert accumulator.mean() == {}
