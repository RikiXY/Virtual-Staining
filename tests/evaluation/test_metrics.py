from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from virtual_staining.evaluation.reports import METRIC_FIELDNAMES
from virtual_staining.metrics import (
    METRIC_SPECS,
    REPORT_METRIC_NAMES,
    compute_mae,
    compute_mse,
    compute_pcc,
    compute_pcc_gray,
    compute_pcc_rgb,
    compute_psnr,
    compute_rmse,
    compute_ssim,
    compute_standard_metrics,
    get_metric_thresholds,
    is_higher_better_metric,
)


def _rgb(value: float, h: int = 8, w: int = 8) -> np.ndarray:
    return np.full((h, w, 3), value, dtype=np.float32)


# ---------------------------------------------------------------------------
# Identical images - perfect scores
# ---------------------------------------------------------------------------


def test_mae_identical_images() -> None:
    img = _rgb(0.5)
    assert compute_mae(img, img) == 0.0


def test_rmse_identical_images() -> None:
    img = _rgb(0.5)
    assert compute_rmse(img, img) == 0.0


def test_mse_identical_images() -> None:
    img = _rgb(0.5)
    assert compute_mse(img, img) == 0.0


def test_psnr_identical_images() -> None:
    img = _rgb(0.5)
    assert compute_psnr(img, img) == float("inf")


def test_ssim_identical_images() -> None:
    img = _rgb(0.5)
    assert compute_ssim(img, img) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Near-inverted pair - poor scores
# ---------------------------------------------------------------------------


def test_mae_inverted_pair() -> None:
    a = _rgb(0.9)
    b = _rgb(0.1)
    assert compute_mae(a, b) == pytest.approx(0.8)


def test_rmse_inverted_pair() -> None:
    a = _rgb(0.9)
    b = _rgb(0.1)
    assert compute_rmse(a, b) == pytest.approx(0.8)


def test_mse_inverted_pair() -> None:
    a = _rgb(0.9)
    b = _rgb(0.1)
    assert compute_mse(a, b) == pytest.approx(0.64)


def test_ssim_inverted_pair_is_low() -> None:
    a = _rgb(0.9)
    b = _rgb(0.1)
    assert compute_ssim(a, b) < 0.5


# ---------------------------------------------------------------------------
# Known PSNR value
# ---------------------------------------------------------------------------


def test_psnr_known_value() -> None:
    # MSE = 0.1^2 = 0.01; PSNR = 20 * log10(1 / 0.1) = 20.0 dB.
    a = _rgb(0.0)
    b = _rgb(0.1)
    assert compute_psnr(a, b) == pytest.approx(20.0, abs=1e-4)


# ---------------------------------------------------------------------------
# Pearson correlation coefficient
# ---------------------------------------------------------------------------


def test_pcc_identical_non_constant_arrays() -> None:
    a = np.linspace(0.0, 1.0, 64, dtype=np.float32).reshape(8, 8)
    assert compute_pcc(a, a) == pytest.approx(1.0)


def test_pcc_inverse_arrays() -> None:
    a = np.linspace(0.0, 1.0, 64, dtype=np.float32).reshape(8, 8)
    b = 1.0 - a
    assert compute_pcc(a, b) == pytest.approx(-1.0)


def test_pcc_constant_array_is_nan() -> None:
    a = np.ones((8, 8), dtype=np.float32)
    b = np.zeros((8, 8), dtype=np.float32)
    assert np.isnan(compute_pcc(a, b))


def test_pcc_gray_and_rgb_mean() -> None:
    a = np.arange(8 * 8 * 3, dtype=np.float32).reshape(8, 8, 3)
    b = a.copy()
    pcc_r, pcc_g, pcc_b, pcc_rgb_mean = compute_pcc_rgb(a, b)

    assert compute_pcc_gray(a, b) == pytest.approx(1.0)
    assert pcc_r == pytest.approx(1.0)
    assert pcc_g == pytest.approx(1.0)
    assert pcc_b == pytest.approx(1.0)
    assert pcc_rgb_mean == pytest.approx(1.0)


def test_metric_specs_supply_direction_and_comparison_thresholds() -> None:
    assert set(METRIC_SPECS) >= {"ssim", "mae", "pcc_r"}
    assert is_higher_better_metric("ssim") is True
    assert is_higher_better_metric("mae") is False
    assert get_metric_thresholds("ssim") == [0.65, 0.75, 0.85]


def test_standard_metrics_and_report_orders_are_registry_contracts() -> None:
    image = np.arange(8 * 8 * 3, dtype=np.float32).reshape(8, 8, 3) / 192.0
    metrics = compute_standard_metrics(image, image)

    assert tuple(metrics) == tuple(METRIC_SPECS)
    assert REPORT_METRIC_NAMES == (
        "mae",
        "mse",
        "rmse",
        "psnr",
        "ssim",
        "pcc_gray",
        "pcc_r",
        "pcc_g",
        "pcc_b",
        "pcc_rgb_mean",
    )
    assert tuple(METRIC_FIELDNAMES[7:]) == REPORT_METRIC_NAMES


def test_metric_evaluator_output_must_match_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = METRIC_SPECS["ssim"]
    assert spec.evaluator_group is not None
    monkeypatch.setitem(
        METRIC_SPECS,
        "ssim",
        replace(
            spec,
            evaluator_group=lambda _target, _generated: {"ssim": 1.0, "unexpected": 2.0},
        ),
    )

    with pytest.raises(ValueError, match="output keys do not match registry"):
        compute_standard_metrics(_rgb(0.5), _rgb(0.5))


def test_unknown_metric_spec_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported metric 'unknown'"):
        is_higher_better_metric("unknown")
