from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from virtual_staining.utils.metrics import DEFAULT_METRICS
from virtual_staining.utils.metrics import get_metric_plot_range as default_metric_plot_range

METRIC_NAMES = list(DEFAULT_METRICS)
PLOT_FIXED_BINS = 30


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
    return default_metric_plot_range(metric)


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


def write_plots(rows: list[dict[str, object]], output_dir: str | Path) -> list[Path]:
    """Backward-compatible alias for dataset plot generation."""
    return save_dataset_plots(rows, output_dir)
