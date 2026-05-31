from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest

from virtual_staining.evaluation.summaries import (
    SUMMARY_METRIC_NAMES,
    build_weak_tail_rows,
    read_summary_csv,
    write_summary_csv,
    write_weak_tail_csv,
)
from virtual_staining.utils.metrics import DEFAULT_WEAK_TAIL_THRESHOLDS


def _make_rows(n: int) -> list[dict[str, object]]:
    return [
        {
            "sample_id": str(i),
            "mae": 0.05 * i,
            "mse": 0.01 * i,
            "rmse": 0.1 * i,
            "psnr": 30.0 + i,
            "ssim": 0.9 - 0.01 * i,
            "pcc_gray": 0.95,
            "pcc_rgb_mean": 0.93,
        }
        for i in range(n)
    ]


def test_write_summary_csv_creates_file(tmp_path: Path) -> None:
    path = write_summary_csv(_make_rows(5), tmp_path)
    assert path.exists()


def test_write_summary_csv_has_expected_columns(tmp_path: Path) -> None:
    path = write_summary_csv(_make_rows(3), tmp_path)
    with path.open() as f:
        fieldnames = csv.DictReader(f).fieldnames or []
    assert "metric" in fieldnames
    assert "mean" in fieldnames
    assert "std" in fieldnames
    assert "count" in fieldnames
    assert "finite_count" in fieldnames


def test_write_summary_csv_handles_psnr_inf(tmp_path: Path) -> None:
    rows: list[dict[str, object]] = [
        {
            "sample_id": "a",
            "psnr": float("inf"),
            "mae": 0.1,
            "mse": 0.01,
            "rmse": 0.1,
            "ssim": 0.9,
            "pcc_gray": 0.9,
            "pcc_rgb_mean": 0.9,
        },
        {
            "sample_id": "b",
            "psnr": 30.0,
            "mae": 0.2,
            "mse": 0.04,
            "rmse": 0.2,
            "ssim": 0.8,
            "pcc_gray": 0.8,
            "pcc_rgb_mean": 0.8,
        },
    ]
    path = write_summary_csv(rows, tmp_path)
    with path.open() as f:
        data = {row["metric"]: row for row in csv.DictReader(f)}
    assert data["psnr"]["count"] == "2"
    assert data["psnr"]["finite_count"] == "1"
    assert data["psnr"]["non_finite_count"] == "1"


def test_write_summary_csv_empty_rows_writes_nan_means(tmp_path: Path) -> None:
    path = write_summary_csv([], tmp_path)
    with path.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(SUMMARY_METRIC_NAMES)
    assert all(row["count"] == "0" for row in rows)
    assert all(math.isnan(float(row["mean"])) for row in rows)


def test_read_write_summary_csv_roundtrip(tmp_path: Path) -> None:
    path = write_summary_csv(_make_rows(3), tmp_path)
    summary = read_summary_csv(path)

    mae = summary["mae"]
    assert mae["count"] == 3.0
    assert mae["finite_count"] == 3.0
    assert mae["non_finite_count"] == 0.0
    assert mae["mean"] == pytest.approx(0.05)
    assert mae["median"] == pytest.approx(0.05)
    assert mae["std"] == pytest.approx(0.05)
    assert mae["min"] == pytest.approx(0.0)
    assert mae["max"] == pytest.approx(0.1)


def test_read_summary_csv_roundtrip_with_preamble(tmp_path: Path) -> None:
    path = write_summary_csv(
        _make_rows(2),
        tmp_path,
        num_targets_found=2,
        num_generated_found=2,
        num_pairs_evaluated=2,
        num_skipped=0,
    )

    summary = read_summary_csv(path)
    psnr = summary["psnr"]

    assert psnr["count"] == 2.0
    assert psnr["finite_count"] == 2.0
    assert psnr["non_finite_count"] == 0.0
    assert psnr["mean"] == pytest.approx(30.5)
    assert psnr["median"] == pytest.approx(30.5)
    assert psnr["std"] == pytest.approx(math.sqrt(0.5))
    assert psnr["min"] == pytest.approx(30.0)
    assert psnr["max"] == pytest.approx(31.0)


def test_default_weak_tail_thresholds_cover_core_metrics() -> None:
    assert DEFAULT_WEAK_TAIL_THRESHOLDS["ssim"] == 0.60
    assert DEFAULT_WEAK_TAIL_THRESHOLDS["mae"] == 0.08
    assert "rmse" in DEFAULT_WEAK_TAIL_THRESHOLDS
    assert "psnr" in DEFAULT_WEAK_TAIL_THRESHOLDS
    assert "pcc_gray" in DEFAULT_WEAK_TAIL_THRESHOLDS
    assert "pcc_rgb_mean" in DEFAULT_WEAK_TAIL_THRESHOLDS


def test_build_weak_tail_rows_counts_higher_is_better_values_below_threshold() -> None:
    rows: list[dict[str, object]] = [
        {"ssim": 0.70},
        {"ssim": 0.60},
        {"ssim": 0.59},
        {"ssim": 0.20},
    ]

    weak_tail = build_weak_tail_rows(rows, thresholds={"ssim": 0.60})
    ssim = weak_tail[0]

    assert ssim["metric"] == "ssim"
    assert ssim["direction"] == "higher_is_better"
    assert ssim["weak_rule"] == "<"
    assert ssim["threshold"] == pytest.approx(0.60)
    assert ssim["count"] == 4
    assert ssim["finite_count"] == 4
    assert ssim["weak_count"] == 2
    assert ssim["weak_share"] == pytest.approx(0.5)
    assert ssim["worst_value"] == pytest.approx(0.20)
    assert ssim["p05"] == pytest.approx(0.2585)


def test_build_weak_tail_rows_counts_lower_is_better_values_above_threshold() -> None:
    rows: list[dict[str, object]] = [
        {"mae": 0.07},
        {"mae": 0.08},
        {"mae": 0.09},
        {"mae": 0.20},
    ]

    weak_tail = build_weak_tail_rows(rows, thresholds={"mae": 0.08})
    mae = weak_tail[0]

    assert mae["metric"] == "mae"
    assert mae["direction"] == "lower_is_better"
    assert mae["weak_rule"] == ">"
    assert mae["count"] == 4
    assert mae["finite_count"] == 4
    assert mae["weak_count"] == 2
    assert mae["weak_share"] == pytest.approx(0.5)
    assert mae["worst_value"] == pytest.approx(0.20)
    assert mae["p95"] == pytest.approx(0.1835)


def test_build_weak_tail_rows_excludes_non_finite_values_from_denominator() -> None:
    rows: list[dict[str, object]] = [
        {"ssim": 0.50},
        {"ssim": float("inf")},
        {"ssim": float("nan")},
        {"ssim": 0.70},
    ]

    weak_tail = build_weak_tail_rows(rows, thresholds={"ssim": 0.60})
    ssim = weak_tail[0]

    assert ssim["count"] == 4
    assert ssim["finite_count"] == 2
    assert ssim["non_finite_count"] == 2
    assert ssim["weak_count"] == 1
    assert ssim["weak_share"] == pytest.approx(0.5)


def test_write_weak_tail_csv_empty_rows_writes_zero_counts(tmp_path: Path) -> None:
    path = write_weak_tail_csv([], tmp_path, thresholds={"ssim": 0.60})

    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert path.exists()
    assert len(rows) == 1
    assert rows[0]["metric"] == "ssim"
    assert rows[0]["count"] == "0"
    assert rows[0]["finite_count"] == "0"
    assert rows[0]["non_finite_count"] == "0"
    assert rows[0]["weak_count"] == "0"
    assert math.isnan(float(rows[0]["weak_share"]))
