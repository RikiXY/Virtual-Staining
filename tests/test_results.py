from __future__ import annotations

from pathlib import Path

import pytest

from virtual_staining.data.results import DatasetBuildResult
from virtual_staining.training.results import EpochMetrics


# ---------------------------------------------------------------------------
# EpochMetrics
# ---------------------------------------------------------------------------

def test_epoch_metrics_required_fields():
    m = EpochMetrics(loss_G=0.42, loss_D=0.18)
    assert m.loss_G == pytest.approx(0.42)
    assert m.loss_D == pytest.approx(0.18)


def test_epoch_metrics_optional_fields_default_none():
    m = EpochMetrics(loss_G=0.5, loss_D=0.3)
    assert m.loss_L1 is None
    assert m.loss_adv is None


def test_epoch_metrics_optional_fields_settable():
    m = EpochMetrics(loss_G=0.5, loss_D=0.3, loss_L1=0.1, loss_adv=0.05)
    assert m.loss_L1 == pytest.approx(0.1)
    assert m.loss_adv == pytest.approx(0.05)


def test_epoch_metrics_frozen():
    m = EpochMetrics(loss_G=0.5, loss_D=0.3)
    with pytest.raises((AttributeError, TypeError)):
        m.loss_G = 0.9  # type: ignore[misc]


# ---------------------------------------------------------------------------
# DatasetBuildResult
# ---------------------------------------------------------------------------

def test_dataset_build_result_fields():
    r = DatasetBuildResult(
        train_count=800,
        val_count=50,
        test_count=150,
        skipped_count=30,
        output_root=Path("/data/experiment"),
    )
    assert r.train_count == 800
    assert r.val_count == 50
    assert r.test_count == 150
    assert r.skipped_count == 30
    assert r.output_root == Path("/data/experiment")


def test_dataset_build_result_output_root_is_path():
    r = DatasetBuildResult(
        train_count=1,
        val_count=1,
        test_count=1,
        skipped_count=0,
        output_root=Path("/some/path"),
    )
    assert isinstance(r.output_root, Path)


def test_dataset_build_result_frozen():
    r = DatasetBuildResult(
        train_count=1,
        val_count=1,
        test_count=1,
        skipped_count=0,
        output_root=Path("/tmp"),
    )
    with pytest.raises((AttributeError, TypeError)):
        r.train_count = 999  # type: ignore[misc]
