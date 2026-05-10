from __future__ import annotations

import csv
import math
import statistics
from pathlib import Path

from virtual_staining.utils.metrics import DEFAULT_METRICS

SUMMARY_METRIC_NAMES = list(DEFAULT_METRICS)


def metric_value(row: dict[str, object], metric: str) -> float:
    """Returns a metric value from a CSV-style row as a float."""
    value = row[metric]
    if isinstance(value, str | int | float):
        return float(value)
    raise TypeError(f"Metric '{metric}' must be a scalar value, got {type(value).__name__}.")


def build_summary_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Builds aggregated rows for summary.csv using finite values only."""
    summary_rows: list[dict[str, object]] = []

    for metric in SUMMARY_METRIC_NAMES:
        values = [metric_value(row, metric) for row in rows]
        finite = [v for v in values if math.isfinite(v)]
        non_finite_count = len(values) - len(finite)

        if finite:
            mean: float = statistics.mean(finite)
            median: float = statistics.median(finite)
            std: float = statistics.stdev(finite) if len(finite) > 1 else 0.0
            min_val: float = min(finite)
            max_val: float = max(finite)
        else:
            mean = median = std = min_val = max_val = float("nan")

        summary_rows.append(
            {
                "metric": metric,
                "count": len(values),
                "finite_count": len(finite),
                "non_finite_count": non_finite_count,
                "mean": mean,
                "median": median,
                "std": std,
                "min": min_val,
                "max": max_val,
            }
        )

    return summary_rows


def write_summary_csv(
    rows: list[dict[str, object]],
    output_dir: Path,
    filename: str = "summary.csv",
    *,
    num_targets_found: int | None = None,
    num_generated_found: int | None = None,
    num_pairs_evaluated: int | None = None,
    num_skipped: int | None = None,
) -> Path:
    """Writes a summary CSV for metric rows, preserving non-finite accounting."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    summary_rows = build_summary_rows(rows)
    fieldnames = [
        "metric",
        "count",
        "finite_count",
        "non_finite_count",
        "mean",
        "median",
        "std",
        "min",
        "max",
    ]

    with path.open("w", newline="", encoding="utf-8") as file:
        if None not in (
            num_targets_found,
            num_generated_found,
            num_pairs_evaluated,
            num_skipped,
        ):
            writer = csv.writer(file)
            writer.writerow(["num_targets_found", num_targets_found])
            writer.writerow(["num_generated_found", num_generated_found])
            writer.writerow(["num_pairs_evaluated", num_pairs_evaluated])
            writer.writerow(["num_skipped", num_skipped])
            writer.writerow([])
            writer.writerow(fieldnames)
            dict_writer = csv.DictWriter(file, fieldnames=fieldnames)
            dict_writer.writerows(summary_rows)
        else:
            dict_writer = csv.DictWriter(file, fieldnames=fieldnames)
            dict_writer.writeheader()
            dict_writer.writerows(summary_rows)

    return path
