from __future__ import annotations

import csv
from pathlib import Path

from virtual_staining.metrics import REPORT_METRIC_NAMES

METRIC_FIELDNAMES = [
    "sample_id",
    "set_id",
    "target_path",
    "generated_path",
    "width",
    "height",
    "channels",
    *REPORT_METRIC_NAMES,
]


def build_metric_row(
    sample_id: str,
    target_path: str | Path,
    generated_path: str | Path,
    shape: tuple[int, int, int],
    metrics: dict[str, float],
    set_id: str,
) -> dict[str, object]:
    """Builds a standard row for per_image_metrics.csv."""
    height, width, channels = shape
    return {
        "sample_id": sample_id,
        "set_id": set_id,
        "target_path": str(target_path),
        "generated_path": str(generated_path),
        "width": width,
        "height": height,
        "channels": channels,
        **{name: metrics[name] for name in METRIC_FIELDNAMES[7:]},
    }


def write_per_image_metrics_csv(rows: list[dict[str, object]], output_path: str | Path) -> Path:
    """Writes the CSV with one row per evaluated pair."""
    path = Path(output_path)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=METRIC_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_skipped_csv(rows: list[dict[str, str]], output_path: str | Path) -> Path:
    """Writes the CSV of skipped samples with the corresponding reason."""
    path = Path(output_path)
    fieldnames = ["sample_id", "reason", "target_path", "generated_path"]

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return path


def write_single_case_csv(row: dict[str, object], output_path: str | Path) -> Path:
    """Writes the CSV produced by the single mode."""
    path = Path(output_path)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=METRIC_FIELDNAMES)
        writer.writeheader()
        writer.writerow(row)
    return path
