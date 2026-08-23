from __future__ import annotations

import csv
from pathlib import Path

METRIC_FIELDNAMES = [
    "sample_id",
    "pair_id",
    "target_path",
    "generated_path",
    "width",
    "height",
    "channels",
    "mae",
    "mse",
    "rmse",
    "psnr",
    "ssim",
    "pcc_gray",
    "pcc_r",
    "pcc_g",
    "pcc_b",
    "pcc_rgb_mean",
]


def build_metric_row(
    sample_id: str,
    target_path: str | Path,
    generated_path: str | Path,
    shape: tuple[int, int, int],
    metrics: dict[str, float],
    pair_id: str,
) -> dict[str, object]:
    """Builds a standard row for per_image_metrics.csv."""
    height, width, channels = shape
    return {
        "sample_id": sample_id,
        "pair_id": pair_id,
        "target_path": str(target_path),
        "generated_path": str(generated_path),
        "width": width,
        "height": height,
        "channels": channels,
        "mae": metrics["mae"],
        "mse": metrics["mse"],
        "rmse": metrics["rmse"],
        "psnr": metrics["psnr"],
        "ssim": metrics["ssim"],
        "pcc_gray": metrics["pcc_gray"],
        "pcc_r": metrics["pcc_r"],
        "pcc_g": metrics["pcc_g"],
        "pcc_b": metrics["pcc_b"],
        "pcc_rgb_mean": metrics["pcc_rgb_mean"],
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
