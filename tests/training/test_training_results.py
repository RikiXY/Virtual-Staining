from __future__ import annotations

import pytest

from virtual_staining.training.results import EpochMetrics


def test_epoch_metrics_required_fields() -> None:
    metrics = EpochMetrics(loss_G=0.42, loss_D=0.18)
    assert metrics.loss_G == pytest.approx(0.42)
    assert metrics.loss_D == pytest.approx(0.18)


def test_epoch_metrics_optional_fields_default_none() -> None:
    metrics = EpochMetrics(loss_G=0.5, loss_D=0.3)
    assert metrics.loss_L1 is None
    assert metrics.loss_adv is None


def test_epoch_metrics_optional_fields_settable() -> None:
    metrics = EpochMetrics(loss_G=0.5, loss_D=0.3, loss_L1=0.1, loss_adv=0.05)
    assert metrics.loss_L1 == pytest.approx(0.1)
    assert metrics.loss_adv == pytest.approx(0.05)


def test_epoch_metrics_frozen() -> None:
    metrics = EpochMetrics(loss_G=0.5, loss_D=0.3)
    with pytest.raises((AttributeError, TypeError)):
        metrics.loss_G = 0.9  # type: ignore[misc]
