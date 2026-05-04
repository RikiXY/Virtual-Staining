from __future__ import annotations

import numpy as np
import pytest

from virtual_staining.evaluation.metrics import (
    compute_mae,
    compute_rmse,
    compute_psnr,
    compute_ssim,
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
