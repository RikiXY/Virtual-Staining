from __future__ import annotations

import csv
import math
from pathlib import Path

from virtual_staining.evaluation.summaries import SUMMARY_METRIC_NAMES, write_summary_csv


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
