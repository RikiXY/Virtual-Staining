from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from virtual_staining.metrics import DEFAULT_METRICS

METRIC_NAMES = list(DEFAULT_METRICS)
PLOT_FIXED_BINS = 30
METRIC_PLOT_RANGES = {
    "mae": (0.0, 1.0),
    "mse": (0.0, 1.0),
    "rmse": (0.0, 1.0),
    "ssim": (0.0, 1.0),
    "pcc_gray": (-1.0, 1.0),
    "pcc_rgb_mean": (-1.0, 1.0),
    "pcc_r": (-1.0, 1.0),
    "pcc_g": (-1.0, 1.0),
    "pcc_b": (-1.0, 1.0),
    "psnr": (0.0, 60.0),
}


def _metric_value(row: dict[str, object], metric: str) -> float:
    value = row[metric]
    if isinstance(value, str | int | float):
        return float(value)
    raise TypeError(f"Metric '{metric}' must be a scalar value, got {type(value).__name__}.")


def _finite_metric_values(rows: list[dict[str, object]], metric: str) -> list[float]:
    """Returns metric values that are finite, intentionally skipping inf and nan."""
    return [v for row in rows if math.isfinite(v := _metric_value(row, metric))]


def get_metric_plot_range(metric: str) -> tuple[float, float]:
    """Returns the fixed range used in plots for a metric."""
    try:
        return METRIC_PLOT_RANGES[metric]
    except KeyError:
        raise ValueError(
            f"Unsupported metric '{metric}'. Supported metrics: {', '.join(METRIC_PLOT_RANGES)}"
        ) from None


def save_dataset_plots(rows: list[dict[str, object]], output_dir: str | Path) -> list[Path]:
    """Saves histograms with fixed axes and a final summary boxplot.

    Non-finite values (inf, nan) are intentionally excluded from all plots.
    """
    output_directory = Path(output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    for metric in METRIC_NAMES:
        values = _finite_metric_values(rows, metric)
        histogram_path = output_directory / f"{metric}_histogram.png"
        min_value, max_value = get_metric_plot_range(metric)
        bin_edges = np.linspace(min_value, max_value, PLOT_FIXED_BINS + 1)

        plt.figure(figsize=(6, 4))
        if values:
            weights = np.ones(len(values), dtype=float) / len(values)
            plt.hist(values, bins=bin_edges.tolist(), weights=weights)
        plt.title(f"{metric.upper()} Histogram")
        plt.xlabel(metric.upper())
        plt.ylabel("Share of samples")
        plt.xlim(min_value, max_value)
        plt.tight_layout()
        plt.savefig(histogram_path, dpi=200, bbox_inches="tight")
        plt.close()

        saved_paths.append(histogram_path)

    boxplot_path = output_directory / "metrics_boxplot.png"
    plt.figure(figsize=(8, 5))
    bp_data = [_finite_metric_values(rows, m) for m in METRIC_NAMES]
    bp_labels = [m.upper() for m in METRIC_NAMES]
    non_empty = [(d, lbl) for d, lbl in zip(bp_data, bp_labels, strict=True) if d]
    if non_empty:
        plot_data, plot_labels = zip(*non_empty, strict=True)
        plt.boxplot(list(plot_data), tick_labels=list(plot_labels))
    plt.title("Metrics Boxplot")
    plt.ylabel("Value")
    plt.tight_layout()
    plt.savefig(boxplot_path, dpi=200, bbox_inches="tight")
    plt.close()

    saved_paths.append(boxplot_path)
    return saved_paths
