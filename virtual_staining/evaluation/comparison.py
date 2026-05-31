from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from virtual_staining.evaluation.statistics import (
    PairedSummary,
    UnpairedComparison,
    UnpairedGroupStats,
    flatten_unpaired_group_stats,
)


def save_unpaired_group_statistics(
    group_a: UnpairedGroupStats,
    group_b: UnpairedGroupStats,
    output_dir: Path,
) -> None:
    """Save group_statistics.csv with one row per group."""
    rows = [
        flatten_unpaired_group_stats(group_a),
        flatten_unpaired_group_stats(group_b),
    ]
    pd.DataFrame(rows).to_csv(output_dir / "group_statistics.csv", index=False)


def save_unpaired_comparison_summary(
    group_a: UnpairedGroupStats,
    group_b: UnpairedGroupStats,
    comparison: UnpairedComparison,
    column: str,
    higher_is_better: bool,
    output_dir: Path,
) -> None:
    """Save comparison_summary.csv for the unpaired comparison."""
    row = {
        "mode": "unpaired",
        "metric": column,
        "direction": ("higher_is_better" if higher_is_better else "lower_is_better"),
        "label_a": group_a.label,
        "label_b": group_b.label,
        "n_a": group_a.n,
        "n_b": group_b.n,
        "mean_a": group_a.mean,
        "mean_b": group_b.mean,
        "median_a": group_a.median,
        "median_b": group_b.median,
        "iqr_a": group_a.iqr,
        "iqr_b": group_b.iqr,
        "mean_favors": comparison.mean_favors,
        "median_favors": comparison.median_favors,
        "threshold_favors": comparison.threshold_favors,
        "wasserstein_between_groups": comparison.wasserstein_between_groups,
        "ks_statistic": comparison.ks_statistic,
        "ks_pvalue": comparison.ks_pvalue,
        "mannwhitney_u": comparison.mannwhitney_u,
        "mannwhitney_pvalue": comparison.mannwhitney_pvalue,
        "better_label": comparison.better_label,
    }
    pd.DataFrame([row]).to_csv(output_dir / "comparison_summary.csv", index=False)


def save_unpaired_summary_json(
    group_a: UnpairedGroupStats,
    group_b: UnpairedGroupStats,
    comparison: UnpairedComparison,
    output_dir: Path,
) -> None:
    """Save a JSON summary of the unpaired comparison."""
    payload = {
        "group_a": asdict(group_a),
        "group_b": asdict(group_b),
        "comparison": asdict(comparison),
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_paired_summary_json(summary: PairedSummary, output_dir: Path) -> None:
    """Save a JSON summary of the paired comparison."""
    (output_dir / "summary.json").write_text(
        json.dumps(asdict(summary), indent=2), encoding="utf-8"
    )


def save_paired_comparison_summary(
    summary: PairedSummary,
    column: str,
    higher_is_better: bool,
    output_dir: Path,
) -> None:
    """Save comparison_summary.csv for the paired comparison."""
    row = {
        "mode": "paired",
        "metric": column,
        "direction": ("higher_is_better" if higher_is_better else "lower_is_better"),
        "label_a": summary.label_a,
        "label_b": summary.label_b,
        "n_pairs": summary.n_pairs,
        "tolerance": summary.tolerance,
        "mean_signed_delta": summary.mean_signed_delta,
        "median_signed_delta": summary.median_signed_delta,
        "share_b_better": summary.share_b_better,
        "share_a_better": summary.share_a_better,
        "share_equal": summary.share_equal,
        "wilcoxon_statistic": summary.wilcoxon_statistic,
        "wilcoxon_pvalue": summary.wilcoxon_pvalue,
        "better_label": summary.better_label,
    }
    pd.DataFrame([row]).to_csv(output_dir / "comparison_summary.csv", index=False)


def save_paired_sample_deltas(
    merged: pd.DataFrame,
    higher_is_better: bool,
    tolerance: float,
    label_a: str,
    label_b: str,
    output_dir: Path,
) -> None:
    """Save a sample-by-sample paired comparison CSV."""
    raw_delta = merged["value_b"].to_numpy(dtype=float) - merged["value_a"].to_numpy(dtype=float)
    signed_delta = raw_delta if higher_is_better else -raw_delta
    result = merged.copy()
    result["raw_delta_b_minus_a"] = raw_delta
    result["signed_delta"] = signed_delta
    result["winner"] = np.where(
        signed_delta > tolerance,
        label_b,
        np.where(
            signed_delta < -tolerance,
            label_a,
            "equal",
        ),
    )
    result.to_csv(output_dir / "paired_sample_deltas.csv", index=False)


def save_paired_multi_metric_reports(
    sample_deltas: pd.DataFrame,
    metric_summary: pd.DataFrame,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Save wide per-sample deltas and per-metric delta summary CSVs."""
    sample_path = output_dir / "paired_sample_deltas_all_metrics.csv"
    summary_path = output_dir / "paired_metric_delta_summary.csv"
    sample_deltas.to_csv(sample_path, index=False)
    metric_summary.to_csv(summary_path, index=False)
    return sample_path, summary_path


def save_unpaired_report_txt(
    group_a: UnpairedGroupStats,
    group_b: UnpairedGroupStats,
    comparison: UnpairedComparison,
    column: str,
    higher_is_better: bool,
    output_dir: Path,
) -> None:
    """Save report.txt for the unpaired comparison."""
    lines = [
        f"Metric: {column}",
        f"Direction: {'higher is better' if higher_is_better else 'lower is better'}",
        "",
        (
            f"{group_a.label}: n={group_a.n}, mean={group_a.mean:.6f}, "
            f"median={group_a.median:.6f}, IQR={group_a.iqr:.6f}"
        ),
        (
            f"{group_b.label}: n={group_b.n}, mean={group_b.mean:.6f}, "
            f"median={group_b.median:.6f}, IQR={group_b.iqr:.6f}"
        ),
        "",
        f"Mean favors: {comparison.mean_favors}",
        f"Median favors: {comparison.median_favors}",
        f"Threshold favors: {comparison.threshold_favors}",
        f"Wasserstein between groups: {comparison.wasserstein_between_groups:.6f}",
        f"KS statistic: {comparison.ks_statistic:.6f}",
        f"KS p-value: {comparison.ks_pvalue:.6g}",
        f"Mann-Whitney U: {comparison.mannwhitney_u:.6f}",
        f"Mann-Whitney p-value: {comparison.mannwhitney_pvalue:.6g}",
        "",
        f"Overall comparison favors: {comparison.better_label}",
    ]
    (output_dir / "report.txt").write_text("\n".join(lines), encoding="utf-8")


def save_paired_report_txt(
    summary: PairedSummary,
    column: str,
    higher_is_better: bool,
    output_dir: Path,
) -> None:
    """Save report.txt for the paired comparison."""
    lines = [
        f"Metric: {column}",
        f"Direction: {'higher is better' if higher_is_better else 'lower is better'}",
        f"Paired samples: {summary.n_pairs}",
        f"Tolerance: {summary.tolerance:.6f}",
        "",
        f"Mean signed delta: {summary.mean_signed_delta:.6f}",
        f"Median signed delta: {summary.median_signed_delta:.6f}",
        f"Share {summary.label_b} better: {summary.share_b_better:.6f}",
        f"Share {summary.label_a} better: {summary.share_a_better:.6f}",
        f"Share equal: {summary.share_equal:.6f}",
        f"Wilcoxon statistic: {summary.wilcoxon_statistic:.6f}",
        f"Wilcoxon p-value: {summary.wilcoxon_pvalue:.6g}",
        "",
        f"Overall paired comparison favors: {summary.better_label}",
    ]
    (output_dir / "report.txt").write_text("\n".join(lines), encoding="utf-8")


def ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build the empirical cumulative distribution function of the sample."""
    x = np.sort(values)
    y = np.arange(1, values.size + 1) / values.size
    return x, y


def plot_distribution_histogram(
    a: np.ndarray,
    b: np.ndarray,
    edges: np.ndarray,
    label_a: str,
    label_b: str,
    column: str,
    output_dir: Path,
) -> None:
    """Save the comparison histogram between two distributions."""
    plt.figure(figsize=(9, 5))
    bins = edges.tolist()
    plt.hist(a, bins=bins, density=True, alpha=0.45, label=label_a)
    plt.hist(b, bins=bins, density=True, alpha=0.45, label=label_b)
    plt.xlabel(column)
    plt.ylabel("Density")
    plt.title(f"Histogram comparison - {column}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "histogram_comparison.png", dpi=180)
    plt.close()


def plot_distribution_ecdf(
    a: np.ndarray,
    b: np.ndarray,
    label_a: str,
    label_b: str,
    column: str,
    output_dir: Path,
) -> None:
    """Save the comparison between the empirical cumulative distributions."""
    xa, ya = ecdf(a)
    xb, yb = ecdf(b)

    plt.figure(figsize=(9, 5))
    plt.step(xa, ya, where="post", label=label_a)
    plt.step(xb, yb, where="post", label=label_b)
    plt.xlabel(column)
    plt.ylabel("ECDF")
    plt.title(f"ECDF comparison - {column}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "ecdf_comparison.png", dpi=180)
    plt.close()


def plot_paired_delta_histogram(signed_delta: np.ndarray, column: str, output_dir: Path) -> None:
    """Save the histogram of signed deltas for the paired comparison."""
    plt.figure(figsize=(9, 5))
    plt.hist(signed_delta, bins=30)
    plt.axvline(0.0, linestyle="--", linewidth=1)
    plt.xlabel(f"Signed delta of {column}")
    plt.ylabel("Count")
    plt.title(f"Paired signed delta histogram - {column}")
    plt.tight_layout()
    plt.savefig(output_dir / "paired_delta_histogram.png", dpi=180)
    plt.close()


def plot_paired_scatter(
    merged: pd.DataFrame,
    label_a: str,
    label_b: str,
    column: str,
    output_dir: Path,
) -> None:
    """Save the paired scatter plot A vs B with a parity diagonal."""
    values_a = merged["value_a"].to_numpy(dtype=float)
    values_b = merged["value_b"].to_numpy(dtype=float)
    min_value = min(float(values_a.min()), float(values_b.min()))
    max_value = max(float(values_a.max()), float(values_b.max()))

    plt.figure(figsize=(6, 6))
    plt.scatter(values_a, values_b, s=12, alpha=0.45)
    plt.plot([min_value, max_value], [min_value, max_value], linestyle="--", linewidth=1)
    plt.xlabel(f"{label_a} {column}")
    plt.ylabel(f"{label_b} {column}")
    plt.title(f"Paired scatter - {column}")
    plt.tight_layout()
    plt.savefig(output_dir / "paired_scatter.png", dpi=180)
    plt.close()
