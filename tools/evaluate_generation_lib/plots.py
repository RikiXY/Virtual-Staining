from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from evaluate_generation_lib.core import METRIC_NAMES


YLIM = 0.5

PLOT_FIXED_RANGES = {
    "mae": (0.0, 1.0),
    "mse": (0.0, 1.0),
    "rmse": (0.0, 1.0),
    "ssim": (0.0, 1.0),
    "pcc_gray": (-1.0, 1.0),
    "pcc_rgb_mean": (-1.0, 1.0),
    "psnr": (0.0, 60.0),
}

PLOT_FIXED_BINS = 30


def get_metric_plot_range(metric: str) -> tuple[float, float]:
    """Restituisce il range fisso usato nei grafici per una metrica."""
    if metric not in PLOT_FIXED_RANGES:
        raise ValueError(f"Unsupported metric for plotting: {metric}")

    return PLOT_FIXED_RANGES[metric]


def save_dataset_plots(rows: list[dict[str, object]], output_dir: str | Path) -> list[Path]:
    """Salva istogrammi con assi fissi e un boxplot finale riassuntivo."""
    output_directory = Path(output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    for metric in METRIC_NAMES:
        values = [float(row[metric]) for row in rows]
        histogram_path = output_directory / f"{metric}_histogram.png"
        min_value, max_value = get_metric_plot_range(metric)
        bin_edges = np.linspace(min_value, max_value, PLOT_FIXED_BINS + 1)

        plt.figure(figsize=(6, 4))
        weights = np.ones(len(values), dtype=float) / len(values)
        plt.hist(values, bins=bin_edges, weights=weights)
        plt.title(f"{metric.upper()} Histogram")
        plt.xlabel(metric.upper())
        plt.ylabel("Share of samples")
        plt.xlim(min_value, max_value)
        plt.ylim(0.0, YLIM)
        plt.tight_layout()
        plt.savefig(histogram_path, dpi=200, bbox_inches="tight")
        plt.close()

        saved_paths.append(histogram_path)

    boxplot_path = output_directory / "metrics_boxplot.png"
    plt.figure(figsize=(8, 5))
    data = [[float(row[metric]) for row in rows] for metric in METRIC_NAMES]
    plt.boxplot(data, tick_labels=[metric.upper() for metric in METRIC_NAMES])
    plt.title("Metrics Boxplot")
    plt.ylabel("Value")
    plt.tight_layout()
    plt.savefig(boxplot_path, dpi=200, bbox_inches="tight")
    plt.close()

    saved_paths.append(boxplot_path)
    return saved_paths