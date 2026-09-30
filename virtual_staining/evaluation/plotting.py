from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from virtual_staining.evaluation.summaries import finite_values, rows_by_output
from virtual_staining.metrics import ResolvedMetric

PLOT_FIXED_BINS = 30


def histogram_edges(values: Sequence[float], plot_range: tuple[float, float] | None) -> np.ndarray:
    """Fixed bins over the definition's presentation range, else over the data."""
    if plot_range is not None:
        return np.linspace(plot_range[0], plot_range[1], PLOT_FIXED_BINS + 1)
    return np.histogram_bin_edges(values if len(values) else [0.0, 1.0], bins=PLOT_FIXED_BINS)


def save_dataset_plots(
    rows: Sequence[Mapping[str, object]],
    metrics: Sequence[ResolvedMetric],
    output_dir: str | Path,
) -> list[Path]:
    """Histogram per output and metric and one boxplot, over finite values only.

    Outputs are never mixed: each histogram belongs to one output
    (``<output>__<metric>_histogram.png``) and each boxplot box is labelled by output.
    """
    output_directory = Path(output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    grouped = rows_by_output(rows)

    for output, metric in ((o, m) for o in grouped for m in metrics):
        values = finite_values(grouped[output], metric.name)
        histogram_path = output_directory / f"{output}__{metric.name}_histogram.png"
        bin_edges = histogram_edges(values, metric.definition.plot_range)

        plt.figure(figsize=(6, 4))
        if values:
            weights = np.ones(len(values), dtype=float) / len(values)
            plt.hist(values, bins=bin_edges.tolist(), weights=weights)
        plt.title(f"{output}: {metric.name.upper()} Histogram (finite values)")
        plt.xlabel(metric.name.upper())
        plt.ylabel("Share of finite samples")
        plt.xlim(float(bin_edges[0]), float(bin_edges[-1]))
        plt.tight_layout()
        plt.savefig(histogram_path, dpi=200, bbox_inches="tight")
        plt.close()

        saved_paths.append(histogram_path)

    boxplot_path = output_directory / "metrics_boxplot.png"
    plt.figure(figsize=(8, 5))
    boxes = [(output, metric.name) for output in grouped for metric in metrics]
    bp_data = [finite_values(grouped[output], name) for output, name in boxes]
    bp_labels = [f"{output}\n{name.upper()}" for output, name in boxes]
    non_empty = [(d, lbl) for d, lbl in zip(bp_data, bp_labels, strict=True) if d]
    if non_empty:
        plot_data, plot_labels = zip(*non_empty, strict=True)
        plt.boxplot(list(plot_data), tick_labels=list(plot_labels))
    plt.title("Metrics Boxplot (finite values)")
    plt.ylabel("Value")
    plt.tight_layout()
    plt.savefig(boxplot_path, dpi=200, bbox_inches="tight")
    plt.close()

    saved_paths.append(boxplot_path)
    return saved_paths


def save_unpaired_feature_plot(
    generated: Mapping[str, Sequence[float]],
    reference: Mapping[str, Sequence[float]],
    path: Path,
) -> Path:
    """Overlay generated vs reference per-image feature histograms; diagnostic only."""
    features = list(generated)
    columns = 4
    rows = math.ceil(len(features) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(4 * columns, 3 * rows), squeeze=False)
    for ax, feature in zip(axes.flat, features, strict=False):
        bins = np.histogram_bin_edges([*generated[feature], *reference[feature]], bins=30)
        for label, values in (("generated", generated[feature]), ("reference", reference[feature])):
            weights = np.ones(len(values), dtype=float) / len(values)
            ax.hist(values, bins=bins.tolist(), weights=weights, alpha=0.5, label=label)
        ax.set_title(feature)
        ax.set_ylabel("Share of images")
    for ax in axes.flat[len(features) :]:
        ax.axis("off")
    axes.flat[0].legend()
    fig.suptitle("Per-image feature distributions: generated vs reference (not paired)")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
