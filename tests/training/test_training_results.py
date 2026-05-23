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


def test_epoch_metrics_component_maps_default_empty() -> None:
    metrics = EpochMetrics(loss_G=0.5, loss_D=0.3)
    assert metrics.raw == {}
    assert metrics.weighted == {}
    assert metrics.current_weight == {}
    assert metrics.image == {}


def test_epoch_metrics_component_maps_settable() -> None:
    metrics = EpochMetrics(
        loss_G=0.5,
        loss_D=0.3,
        raw={"ssim": 0.2},
        weighted={"ssim": 0.4},
        current_weight={"ssim": 2.0},
        image={"val_ssim": 0.9},
    )
    assert metrics.raw["ssim"] == pytest.approx(0.2)
    assert metrics.weighted["ssim"] == pytest.approx(0.4)
    assert metrics.current_weight["ssim"] == pytest.approx(2.0)
    assert metrics.image["val_ssim"] == pytest.approx(0.9)


def test_epoch_metrics_frozen() -> None:
    metrics = EpochMetrics(loss_G=0.5, loss_D=0.3)
    with pytest.raises((AttributeError, TypeError)):
        metrics.loss_G = 0.9  # type: ignore[misc]
