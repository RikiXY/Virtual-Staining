from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest

from virtual_staining.evaluation.reports import build_metric_row
from virtual_staining.evaluation.summaries import (
    SUMMARY_FIELDNAMES,
    read_summary_csv,
    write_summary_csv,
)
from virtual_staining.metrics import MetricResult

_NAMES = ["mae", "psnr", "pcc_gray", "ssim"]


def _row(i: int, **overrides: MetricResult) -> dict[str, object]:
    results = {
        "mae": MetricResult.of(0.05 * i),
        "psnr": MetricResult.of(30.0 + i),
        "pcc_gray": MetricResult.of(0.95),
        "ssim": MetricResult.of(0.9 - 0.01 * i),
        **overrides,
    }
    return build_metric_row(str(i), "t.png", "g.png", (8, 8, 3), results, set_id="S")


def _summary(tmp_path: Path, rows: list[dict[str, object]]) -> dict[str, dict[str, str]]:
    path = write_summary_csv(rows, _NAMES, tmp_path)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == SUMMARY_FIELDNAMES
        return {row["metric"]: row for row in reader}


def test_summary_follows_the_requested_metric_order(tmp_path: Path) -> None:
    assert list(_summary(tmp_path, [_row(0), _row(1)])) == _NAMES


def test_summary_counts_every_status_and_averages_finite_values_only(tmp_path: Path) -> None:
    rows = [
        _row(0, psnr=MetricResult.of(math.inf), pcc_gray=MetricResult.undefined("constant")),
        _row(1, ssim=MetricResult.unavailable("too small")),
        _row(2),
    ]

    summary = _summary(tmp_path, rows)

    psnr = summary["psnr"]
    assert (psnr["count"], psnr["finite_count"], psnr["positive_infinity_count"]) == ("3", "2", "1")
    assert float(psnr["finite_mean"]) == pytest.approx(31.5)
    assert (summary["pcc_gray"]["undefined_count"], summary["pcc_gray"]["finite_count"]) == (
        "1",
        "2",
    )
    assert summary["ssim"]["unavailable_count"] == "1"
    assert float(summary["ssim"]["finite_mean"]) == pytest.approx(0.89)
    for row in summary.values():
        assert int(row["count"]) == sum(
            int(row[f"{status}_count"])
            for status in ("finite", "positive_infinity", "undefined", "unavailable")
        )


def test_summary_without_finite_values_leaves_statistics_empty(tmp_path: Path) -> None:
    rows = [_row(0, pcc_gray=MetricResult.undefined("constant"))]

    summary = _summary(tmp_path, rows)

    assert summary["pcc_gray"]["finite_mean"] == ""
    assert math.isnan(read_summary_csv(tmp_path / "summary.csv")["pcc_gray"]["finite_mean"])


def test_read_write_summary_csv_roundtrip(tmp_path: Path) -> None:
    path = write_summary_csv([_row(0), _row(1), _row(2)], _NAMES, tmp_path)
    mae = read_summary_csv(path)["mae"]

    assert (mae["count"], mae["finite_count"], mae["undefined_count"]) == (3.0, 3.0, 0.0)
    assert mae["finite_mean"] == pytest.approx(0.05)
    assert mae["finite_median"] == pytest.approx(0.05)
    assert mae["finite_std"] == pytest.approx(0.05)
    assert (mae["finite_min"], mae["finite_max"]) == pytest.approx((0.0, 0.1))
