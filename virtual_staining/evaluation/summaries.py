"""Dataset and grouped summaries of the requested per-image metrics, per model output.

Every summary row belongs to one ``output_name``; no statistic is ever pooled across
outputs. Every count column covers all evaluated samples of that output; ``finite_*``
statistics use only results whose status is ``finite``, so positive infinity, undefined
and unavailable results are counted but never averaged.
"""

from __future__ import annotations

import csv
import math
import random
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path

SUMMARY_FIELDNAMES = [
    "output_name",
    "metric",
    "count",
    "finite_count",
    "positive_infinity_count",
    "undefined_count",
    "unavailable_count",
    "finite_mean",
    "finite_median",
    "finite_std",
    "finite_min",
    "finite_max",
]
_STATUS_COUNTS = {
    "finite": "finite_count",
    "positive_infinity": "positive_infinity_count",
    "undefined": "undefined_count",
    "unavailable": "unavailable_count",
}


def finite_values(rows: Sequence[Mapping[str, object]], metric: str) -> list[float]:
    """Numbers of the rows whose ``<metric>_status`` is ``finite``."""
    return [float(str(row[metric])) for row in rows if row[f"{metric}_status"] == "finite"]


def rows_by_output(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, list[Mapping[str, object]]]:
    """Group per-image rows by ``output_name``, in first-appearance order."""
    grouped: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["output_name"]), []).append(row)
    return grouped


def _summary_row(rows: Sequence[Mapping[str, object]], metric: str) -> dict[str, object]:
    counts = dict.fromkeys(_STATUS_COUNTS.values(), 0)
    for row in rows:
        counts[_STATUS_COUNTS[str(row[f"{metric}_status"])]] += 1
    values = finite_values(rows, metric)
    stats: dict[str, object] = dict.fromkeys(
        ("finite_mean", "finite_median", "finite_std", "finite_min", "finite_max"), ""
    )
    if values:
        stats = {
            "finite_mean": statistics.mean(values),
            "finite_median": statistics.median(values),
            "finite_std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "finite_min": min(values),
            "finite_max": max(values),
        }
    return {"metric": metric, "count": len(rows), **counts, **stats}


def write_summary_csv(
    rows: Sequence[Mapping[str, object]],
    metric_names: Sequence[str],
    output_dir: Path,
    filename: str = "summary.csv",
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDNAMES)
        writer.writeheader()
        writer.writerows(
            {"output_name": output, **_summary_row(output_rows, metric)}
            for output, output_rows in rows_by_output(rows).items()
            for metric in metric_names
        )
    return path


def read_summary_csv(path: str | Path) -> dict[str, dict[str, dict[str, float]]]:
    """Summary statistics by output and metric; empty cells (no finite values) read as NaN."""
    summary_path = Path(path)
    if not summary_path.is_file():
        raise FileNotFoundError(f"Summary CSV not found: {summary_path}")
    summaries: dict[str, dict[str, dict[str, float]]] = {}
    with summary_path.open("r", newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            summaries.setdefault(row["output_name"], {})[row["metric"]] = {
                key: float(row[key]) if row[key] != "" else math.nan
                for key in SUMMARY_FIELDNAMES[2:]
            }
    return summaries


def read_per_image_metrics_csv(path: str | Path) -> list[dict[str, str]]:
    csv_path = Path(path)

    if not csv_path.is_file():
        raise FileNotFoundError(f"Per-image metrics CSV not found: {csv_path}")

    with csv_path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_grouped_summaries(
    rows: Sequence[Mapping[str, object]],
    metric_names: Sequence[str],
    set_rows: Mapping[str, Mapping[str, str]],
    output_dir: Path,
    *,
    bootstrap_iterations: int,
    bootstrap_seed: int,
) -> list[Path]:
    """Per-output, per-group finite means and a group-level bootstrap of their mean.

    ``<unit>_metrics.csv`` holds one row per (output, group) (set, specimen or patient,
    from the supplied slide-set metadata) with ``<m>_finite_count`` and
    ``<m>_finite_mean``. ``summary_<unit>.csv`` resamples one output's groups
    (``resampling_unit``) with replacement; only groups with a finite mean for that metric
    take part. Outputs are never pooled.
    """
    written: list[Path] = []
    by_output = rows_by_output(rows)
    for unit, field in (("set", "set_id"), ("specimen", "specimen_id"), ("patient", "patient_id")):
        groups: dict[tuple[str, str], list[Mapping[str, object]]] = {}
        incomplete = False
        for row in rows:
            set_id = str(row["set_id"])
            metadata = set_rows.get(set_id)
            group_id = set_id if unit == "set" else (metadata or {}).get(field, "")
            if not group_id:
                incomplete = True
                break
            groups.setdefault((str(row["output_name"]), group_id), []).append(row)
        if incomplete or not groups:
            continue
        unit_rows: list[dict[str, object]] = []
        for output in by_output:
            for group_id in sorted(group for name, group in groups if name == output):
                group_rows = groups[output, group_id]
                unit_row: dict[str, object] = {
                    "output_name": output,
                    "unit": unit,
                    "group_id": group_id,
                    "patch_count": len(group_rows),
                }
                for metric in metric_names:
                    values = finite_values(group_rows, metric)
                    unit_row[f"{metric}_finite_count"] = len(values)
                    unit_row[f"{metric}_finite_mean"] = statistics.mean(values) if values else ""
                unit_rows.append(unit_row)
        metrics_path = output_dir / f"{unit}_metrics.csv"
        with metrics_path.open("w", newline="", encoding="utf-8") as handle:
            fieldnames = ["output_name", "unit", "group_id", "patch_count"]
            for metric in metric_names:
                fieldnames += [f"{metric}_finite_count", f"{metric}_finite_mean"]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(unit_rows)
        written.append(metrics_path)
        summary_path = output_dir / f"summary_{unit}.csv"
        with summary_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "output_name",
                    "resampling_unit",
                    "metric",
                    "group_count",
                    "finite_mean",
                    "ci95_low",
                    "ci95_high",
                ],
            )
            writer.writeheader()
            for output, metric in ((o, m) for o in by_output for m in metric_names):
                values = [
                    float(str(row[f"{metric}_finite_mean"]))
                    for row in unit_rows
                    if row["output_name"] == output and row[f"{metric}_finite_mean"] != ""
                ]
                # Seeded per metric so adding or reordering metrics or outputs never moves
                # another CI.
                rng = random.Random(f"{bootstrap_seed}:{metric}")
                bootstrap = sorted(
                    statistics.mean(rng.choice(values) for _ in values)
                    for _ in range(bootstrap_iterations if values else 0)
                )
                writer.writerow(
                    {
                        "output_name": output,
                        "resampling_unit": unit,
                        "metric": metric,
                        "group_count": len(values),
                        "finite_mean": statistics.mean(values) if values else "",
                        "ci95_low": bootstrap[int(0.025 * (len(bootstrap) - 1))]
                        if bootstrap
                        else "",
                        "ci95_high": bootstrap[int(0.975 * (len(bootstrap) - 1))]
                        if bootstrap
                        else "",
                    }
                )
        written.append(summary_path)
    return written
