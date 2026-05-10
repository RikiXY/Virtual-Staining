from __future__ import annotations

import math
from pathlib import Path

import pytest

from virtual_staining.evaluation.plotting import METRIC_NAMES, save_dataset_plots
from virtual_staining.evaluation.summaries import build_summary_rows


def _row(value: float) -> dict[str, object]:
    return {metric: value for metric in METRIC_NAMES}


def test_save_dataset_plots_creates_expected_files(tmp_path: Path) -> None:
    rows = [_row(0.5), _row(0.6), _row(0.7)]

    saved_paths = save_dataset_plots(rows, tmp_path)

    expected_names = {f"{metric}_histogram.png" for metric in METRIC_NAMES}
    expected_names.add("metrics_boxplot.png")

    assert {path.name for path in saved_paths} == expected_names
    assert all(path.is_file() for path in saved_paths)


# ---------------------------------------------------------------------------
# Non-finite value handling in plots
# ---------------------------------------------------------------------------


def test_save_dataset_plots_skips_inf_psnr_without_crashing(tmp_path: Path) -> None:
    """save_dataset_plots must not crash when PSNR is inf (identical images)."""
    row = dict(_row(0.5))
    row["psnr"] = float("inf")
    saved_paths = save_dataset_plots([row, _row(0.6)], tmp_path / "inf_psnr")
    assert all(p.is_file() for p in saved_paths)


def test_save_dataset_plots_skips_nan_pcc_without_crashing(tmp_path: Path) -> None:
    """save_dataset_plots must not crash when PCC metrics are nan (constant images)."""
    row = dict(_row(0.5))
    row["pcc_gray"] = float("nan")
    row["pcc_rgb_mean"] = float("nan")
    saved_paths = save_dataset_plots([row, _row(0.6)], tmp_path / "nan_pcc")
    assert all(p.is_file() for p in saved_paths)


def test_save_dataset_plots_all_non_finite_without_crashing(tmp_path: Path) -> None:
    """save_dataset_plots must not crash when every value for a metric is non-finite."""
    rows: list[dict[str, object]] = [{metric: float("inf") for metric in METRIC_NAMES}]
    saved_paths = save_dataset_plots(rows, tmp_path / "all_nonfinite")
    assert all(p.is_file() for p in saved_paths)


# ---------------------------------------------------------------------------
# build_summary_rows: non-finite count tracking
# ---------------------------------------------------------------------------


def test_build_summary_rows_tracks_non_finite_count() -> None:
    """build_summary_rows must count non-finite values and compute stats over finite ones."""
    rows = [_row(0.5), _row(0.6)]
    rows_with_inf = [dict(rows[0]), rows[1]]
    rows_with_inf[0]["psnr"] = float("inf")

    summary = build_summary_rows(rows_with_inf)
    psnr_row = next(r for r in summary if r["metric"] == "psnr")

    assert psnr_row["non_finite_count"] == 1
    assert psnr_row["finite_count"] == 1
    assert psnr_row["mean"] == pytest.approx(0.6)


def test_build_summary_rows_all_non_finite_returns_nan_stats() -> None:
    """build_summary_rows must return nan for stats when all values are non-finite."""
    rows: list[dict[str, object]] = [{metric: float("nan") for metric in METRIC_NAMES}]

    summary = build_summary_rows(rows)
    pcc_row = next(r for r in summary if r["metric"] == "pcc_gray")

    assert pcc_row["non_finite_count"] == 1
    assert pcc_row["finite_count"] == 0
    mean_val = pcc_row["mean"]
    assert isinstance(mean_val, float)
    assert math.isnan(mean_val)
