from __future__ import annotations

from pathlib import Path

import pytest

from virtual_staining.data.results import DatasetBuildResult
from virtual_staining.training.results import EpochMetrics

# ---------------------------------------------------------------------------
# EpochMetrics
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# DatasetBuildResult
# ---------------------------------------------------------------------------


def test_dataset_build_result_fields() -> None:
    result = DatasetBuildResult(
        train_count=800,
        val_count=50,
        test_count=150,
        skipped_count=30,
        output_root=Path("/data/experiment"),
    )
    assert result.train_count == 800
    assert result.val_count == 50
    assert result.test_count == 150
    assert result.skipped_count == 30
    assert result.output_root == Path("/data/experiment")


def test_dataset_build_result_output_root_is_path() -> None:
    result = DatasetBuildResult(
        train_count=1,
        val_count=1,
        test_count=1,
        skipped_count=0,
        output_root=Path("/some/path"),
    )
    assert isinstance(result.output_root, Path)


def test_dataset_build_result_frozen() -> None:
    result = DatasetBuildResult(
        train_count=1,
        val_count=1,
        test_count=1,
        skipped_count=0,
        output_root=Path("/tmp"),
    )
    with pytest.raises((AttributeError, TypeError)):
        result.train_count = 999  # type: ignore[misc]
