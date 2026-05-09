from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from virtual_staining.evaluation.statistics import (
    PairedSummary,
    UnpairedComparison,
    UnpairedGroupStats,
    align_paired_frames,
    compute_paired_summary,
    compute_unpaired_comparison,
    compute_unpaired_group_stats,
    flatten_unpaired_group_stats,
    load_metric_values,
    resolve_comparison_inputs,
    resolve_comparison_output_dir,
    resolve_metric_direction,
    resolve_plot_range,
    resolve_thresholds,
)
from virtual_staining.utils.cli import print_info, print_section, style
from virtual_staining.utils.metrics import color_metric_value


def color_distance(value: float, good: float, warn: float) -> str:
    """Colours a distance: the smaller, the better."""
    if value <= good:
        return style(f"{value:.6f}", "green")
    if value <= warn:
        return style(f"{value:.6f}", "yellow")
    if value <= warn * 1.5:
        return style(f"{value:.6f}", "orange")
    return style(f"{value:.6f}", "red")


def color_pvalue(value: float) -> str:
    """Colours a p-value according to the strength of evidence for a difference."""
    if value < 0.001:
        return style(f"{value:.6g}", "green")
    if value < 0.01:
        return style(f"{value:.6g}", "yellow")
    if value < 0.05:
        return style(f"{value:.6g}", "orange")
    return style(f"{value:.6g}", "red")


def color_share(value: float) -> str:
    """Colours a share between 0 and 1."""
    if value >= 0.70:
        return style(f"{value:.6f}", "green")
    if value >= 0.40:
        return style(f"{value:.6f}", "yellow")
    if value >= 0.20:
        return style(f"{value:.6f}", "orange")
    return style(f"{value:.6f}", "red")


def color_signed_delta(value: float) -> str:
    """Colours a signed delta: positive favours B, negative favours A."""
    if value > 0:
        return style(f"{value:.6f}", "green")
    if value < 0:
        return style(f"{value:.6f}", "red")
    return style(f"{value:.6f}", "yellow")


# ==========================
# Section dedicated to the parser
# ==========================
def add_direction_arguments(parser: argparse.ArgumentParser) -> None:
    """Adds optional metric direction overrides."""
    parser.add_argument(
        "--higher-is-better",
        action="store_true",
        help="Override the default metric direction for metrics like SSIM and PSNR.",
    )
    parser.add_argument(
        "--lower-is-better",
        action="store_true",
        help="Override the default metric direction for metrics like MAE and RMSE.",
    )


def add_common_comparison_arguments(parser: argparse.ArgumentParser) -> None:
    """Adds the arguments common to both comparison modes."""
    parser.add_argument(
        "--run-a",
        type=Path,
        default=None,
        help=("First run directory. The script reads RUN_A/evaluation/per_image_metrics.csv."),
    )
    parser.add_argument(
        "--run-b",
        type=Path,
        default=None,
        help=("Second run directory. The script reads RUN_B/evaluation/per_image_metrics.csv."),
    )
    parser.add_argument(
        "--csv-a",
        default=None,
        help=(
            "First CSV file or directory containing per_image_metrics.csv. "
            "Advanced alternative to --run-a."
        ),
    )
    parser.add_argument(
        "--csv-b",
        default=None,
        help=(
            "Second CSV file or directory containing per_image_metrics.csv. "
            "Advanced alternative to --run-b."
        ),
    )
    parser.add_argument(
        "--label-a",
        default=None,
        help=(
            "Label shown in reports and plots for the first group. "
            "If omitted, inferred from --run-a or --csv-a."
        ),
    )
    parser.add_argument(
        "--label-b",
        default=None,
        help=(
            "Label shown in reports and plots for the second group. "
            "If omitted, inferred from --run-b or --csv-b."
        ),
    )
    parser.add_argument(
        "--column",
        default="ssim",
        help="Metric column to compare.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory where outputs will be saved. If omitted, outputs go under "
            "results/comparisons/RUN_A_vs_RUN_B/MODE_METRIC/."
        ),
    )
    add_direction_arguments(parser)


def add_unpaired_subparser(subparsers: Any) -> None:
    """Adds the subcommand for unpaired distributions."""
    parser = subparsers.add_parser(
        "unpaired",
        help="Compare two independent metric distributions.",
        description=(
            "Compare two independent metric distributions from per-image CSV files. "
            "Useful when the two runs do not share exactly the same samples."
        ),
    )
    add_common_comparison_arguments(parser)
    parser.add_argument(
        "--min-value",
        type=float,
        default=None,
        help=(
            "Minimum metric value used for shared histogram bins. "
            "If omitted, inferred from metric defaults."
        ),
    )
    parser.add_argument(
        "--max-value",
        type=float,
        default=None,
        help=(
            "Maximum metric value used for shared histogram bins. "
            "If omitted, inferred from metric defaults."
        ),
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=30,
        help="Number of bins for the common histogram.",
    )
    parser.add_argument(
        "--thresholds",
        nargs="*",
        type=float,
        default=None,
        help=(
            "Thresholds used for share-above or share-below statistics. "
            "If omitted, inferred from metric defaults."
        ),
    )
    parser.set_defaults(func=run_unpaired)


def add_paired_subparser(subparsers: Any) -> None:
    """Adds the subcommand for paired distributions on the same samples."""
    parser = subparsers.add_parser(
        "paired",
        help="Compare two paired metric distributions on the same samples.",
        description=(
            "Compare two paired metric distributions by aligning rows on the same sample_id. "
            "Useful when the two runs share the same test samples."
        ),
    )
    add_common_comparison_arguments(parser)

    parser.add_argument(
        "--sample-id-column",
        default="sample_id",
        help="Column used to align the two CSV files.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.0,
        help="Tolerance below which two values are considered equal.",
    )

    parser.add_argument(
        "--min-value",
        type=float,
        default=None,
        help=(
            "Minimum metric value used for shared histogram bins. "
            "If omitted, inferred from metric defaults."
        ),
    )
    parser.add_argument(
        "--max-value",
        type=float,
        default=None,
        help=(
            "Maximum metric value used for shared histogram bins. "
            "If omitted, inferred from metric defaults."
        ),
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=30,
        help="Number of bins for the comparison histogram.",
    )

    parser.set_defaults(func=run_paired)


def build_parser() -> argparse.ArgumentParser:
    """Builds the main parser and registers the available subcommands."""
    parser = argparse.ArgumentParser(
        prog="python tools/compare_distributions.py",
        description=(
            "Compare metric distributions from per-image CSV files. "
            "Supports both unpaired and paired comparisons."
        ),
        epilog=(
            "Examples:\n"
            "  python tools/compare_distributions.py unpaired \\\n"
            "      --run-a local_workspace/results/run_a \\\n"
            "      --run-b local_workspace/results/run_b \\\n"
            "      --column ssim\n"
            "\n"
            "  python tools/compare_distributions.py paired \\\n"
            "      --run-a local_workspace/results/L1-25 \\\n"
            "      --run-b local_workspace/results/L1-31 \\\n"
            "      --column ssim\n"
            "\n"
            "  python tools/compare_distributions.py paired \\\n"
            "      --csv-a custom_a/per_image_metrics.csv \\\n"
            "      --csv-b custom_b/per_image_metrics.csv \\\n"
            "      --label-a custom_a \\\n"
            "      --label-b custom_b \\\n"
            "      --column ssim \\\n"
            "      --output-dir local_workspace/results/comparisons/custom_paired_ssim\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
        add_help=True,
    )
    subparsers = parser.add_subparsers(dest="mode")
    add_unpaired_subparser(subparsers)
    add_paired_subparser(subparsers)
    return parser


def validate_direction(args: argparse.Namespace) -> None:
    """Resolves metric direction, allowing explicit CLI overrides."""
    try:
        args.resolved_higher_is_better = resolve_metric_direction(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


# =======================================
# Section dedicated to output writing
# =======================================
def save_unpaired_group_statistics(
    group_a: UnpairedGroupStats,
    group_b: UnpairedGroupStats,
    output_dir: Path,
) -> None:
    """Saves group_statistics.csv with one row per group."""
    rows = [
        flatten_unpaired_group_stats(group_a),
        flatten_unpaired_group_stats(group_b),
    ]
    pd.DataFrame(rows).to_csv(output_dir / "group_statistics.csv", index=False)


def save_unpaired_comparison_summary(
    group_a: UnpairedGroupStats,
    group_b: UnpairedGroupStats,
    comparison: UnpairedComparison,
    args: argparse.Namespace,
    output_dir: Path,
) -> None:
    """Saves comparison_summary.csv for the unpaired comparison."""
    row = {
        "mode": "unpaired",
        "metric": args.column,
        "direction": ("higher_is_better" if args.resolved_higher_is_better else "lower_is_better"),
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
    """Saves a JSON summary of the unpaired comparison."""
    payload = {
        "group_a": asdict(group_a),
        "group_b": asdict(group_b),
        "comparison": asdict(comparison),
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_paired_summary_json(summary: PairedSummary, output_dir: Path) -> None:
    """Saves a JSON summary of the paired comparison."""
    (output_dir / "summary.json").write_text(
        json.dumps(asdict(summary), indent=2), encoding="utf-8"
    )


def save_paired_comparison_summary(
    summary: PairedSummary,
    args: argparse.Namespace,
    output_dir: Path,
) -> None:
    """Saves comparison_summary.csv for the paired comparison."""
    row = {
        "mode": "paired",
        "metric": args.column,
        "direction": ("higher_is_better" if args.resolved_higher_is_better else "lower_is_better"),
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
    args: argparse.Namespace,
    output_dir: Path,
) -> None:
    """Saves a sample-by-sample paired comparison CSV."""
    raw_delta = merged["value_b"].to_numpy(dtype=float) - merged["value_a"].to_numpy(dtype=float)
    signed_delta = raw_delta if args.resolved_higher_is_better else -raw_delta
    result = merged.copy()
    result["raw_delta_b_minus_a"] = raw_delta
    result["signed_delta"] = signed_delta
    result["winner"] = np.where(
        signed_delta > args.tolerance,
        args.resolved_label_b,
        np.where(
            signed_delta < -args.tolerance,
            args.resolved_label_a,
            "equal",
        ),
    )
    result.to_csv(output_dir / "paired_sample_deltas.csv", index=False)


def save_unpaired_report_txt(
    group_a: UnpairedGroupStats,
    group_b: UnpairedGroupStats,
    comparison: UnpairedComparison,
    args: argparse.Namespace,
    output_dir: Path,
) -> None:
    """Saves report.txt for the unpaired comparison."""
    lines = [
        f"Metric: {args.column}",
        f"Direction: {'higher is better' if args.resolved_higher_is_better else 'lower is better'}",
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
    summary: PairedSummary, args: argparse.Namespace, output_dir: Path
) -> None:
    """Saves report.txt for the paired comparison."""
    lines = [
        f"Metric: {args.column}",
        f"Direction: {'higher is better' if args.resolved_higher_is_better else 'lower is better'}",
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


# ===========================
# Section dedicated to plots
# ===========================
def ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Builds the empirical cumulative distribution function of the sample."""
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
    """Saves the comparison histogram between two distributions."""
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
    """Saves the comparison between the empirical cumulative distributions."""
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
    """Saves the histogram of signed deltas for the paired comparison."""
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
    """Saves the paired scatter plot A vs B with a parity diagonal."""
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


# ====================================
# Section dedicated to the text report
# ====================================
def print_unpaired_group_summary(group: UnpairedGroupStats, metric_name: str) -> None:
    """Prints the CLI summary of an unpaired group."""
    print_section(f"Group {group.label}")
    print_info("Samples", str(group.n))
    print_info("Mean", color_metric_value(metric_name, group.mean))
    print_info("Median", color_metric_value(metric_name, group.median))
    print_info("IQR", color_distance(group.iqr, 0.05, 0.10))

    for key, value in group.threshold_shares.items():
        print_info(key, color_share(value))


def print_unpaired_cli_summary(
    group_a: UnpairedGroupStats,
    group_b: UnpairedGroupStats,
    comparison: UnpairedComparison,
    args: argparse.Namespace,
    output_dir: Path,
) -> None:
    """Prints the CLI summary of the unpaired comparison."""
    print_section("Input")
    print_info("Metric", args.column)
    print_info(
        "Direction",
        "higher is better" if args.resolved_higher_is_better else "lower is better",
    )
    print_info("Output dir", str(output_dir))

    print_unpaired_group_summary(group_a, args.column)
    print_unpaired_group_summary(group_b, args.column)

    print_section("Distribution comparison")
    comparison_color = "green" if comparison.better_label != "tie" else "yellow"
    print_info(
        "Mean favors",
        style(comparison.mean_favors, comparison_color)
        if comparison.mean_favors != "tie"
        else style("tie", "yellow"),
    )
    print_info(
        "Median favors",
        style(comparison.median_favors, comparison_color)
        if comparison.median_favors != "tie"
        else style("tie", "yellow"),
    )
    print_info(
        "Threshold favors",
        style(comparison.threshold_favors, comparison_color)
        if comparison.threshold_favors != "tie"
        else style("tie", "yellow"),
    )
    print_info(
        "Wasserstein between groups",
        color_distance(comparison.wasserstein_between_groups, 0.03, 0.08),
    )
    print_info("KS statistic", color_distance(comparison.ks_statistic, 0.08, 0.18))
    print_info("KS p-value", color_pvalue(comparison.ks_pvalue))
    print_info("Mann-Whitney U", f"{comparison.mannwhitney_u:.6f}")
    print_info("Mann-Whitney p-value", color_pvalue(comparison.mannwhitney_pvalue))

    print_section("Conclusion")
    print(
        style(
            f"Overall unpaired comparison favors: {comparison.better_label}",
            "bold",
            comparison_color,
        )
    )
    print(style(f"Saved outputs to: {output_dir}", "bold", "magenta"))


def print_paired_cli_summary(
    summary: PairedSummary, args: argparse.Namespace, output_dir: Path
) -> None:
    """Prints the CLI summary of the paired comparison."""
    print_section("Input")
    print_info("Metric", args.column)
    print_info(
        "Direction",
        "higher is better" if args.resolved_higher_is_better else "lower is better",
    )
    print_info("Output dir", str(output_dir))

    print_section("Paired comparison")
    print_info("Paired samples", str(summary.n_pairs))
    print_info("Tolerance", f"{summary.tolerance:.6f}")
    print_info("Mean signed delta", color_signed_delta(summary.mean_signed_delta))
    print_info("Median signed delta", color_signed_delta(summary.median_signed_delta))
    print_info(f"Share {summary.label_b} better", color_share(summary.share_b_better))
    print_info(f"Share {summary.label_a} better", color_share(summary.share_a_better))
    print_info("Share equal", color_share(summary.share_equal))
    print_info("Wilcoxon statistic", f"{summary.wilcoxon_statistic:.6f}")
    print_info("Wilcoxon p-value", color_pvalue(summary.wilcoxon_pvalue))

    conclusion_color = "green" if summary.better_label != "tie" else "yellow"
    print_section("Conclusion")
    print(
        style(
            f"Overall paired comparison favors: {summary.better_label}",
            "bold",
            conclusion_color,
        )
    )
    print(style(f"Saved outputs to: {output_dir}", "bold", "magenta"))


# =====================================
# Section dedicated to the main flow
# =====================================
def run_unpaired(args: argparse.Namespace) -> None:
    """Runs the complete flow for the comparison between unpaired distributions."""
    resolve_comparison_inputs(args)
    output_dir = resolve_comparison_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    values_a = load_metric_values(args.resolved_csv_a, args.column)
    values_b = load_metric_values(args.resolved_csv_b, args.column)
    thresholds = resolve_thresholds(args)
    higher_is_better = args.resolved_higher_is_better

    group_a = compute_unpaired_group_stats(
        values=values_a,
        label=args.resolved_label_a,
        thresholds=thresholds,
        higher_is_better=higher_is_better,
    )
    group_b = compute_unpaired_group_stats(
        values=values_b,
        label=args.resolved_label_b,
        thresholds=thresholds,
        higher_is_better=higher_is_better,
    )
    comparison = compute_unpaired_comparison(
        a=values_a,
        b=values_b,
        group_a=group_a,
        group_b=group_b,
        higher_is_better=higher_is_better,
    )

    save_unpaired_group_statistics(group_a, group_b, output_dir)
    save_unpaired_comparison_summary(group_a, group_b, comparison, args, output_dir)
    save_unpaired_summary_json(group_a, group_b, comparison, output_dir)
    save_unpaired_report_txt(group_a, group_b, comparison, args, output_dir)

    min_value, max_value = resolve_plot_range(args)
    edges = np.linspace(min_value, max_value, args.bins + 1)
    plot_distribution_histogram(
        values_a,
        values_b,
        edges,
        args.resolved_label_a,
        args.resolved_label_b,
        args.column,
        output_dir,
    )
    plot_distribution_ecdf(
        values_a,
        values_b,
        args.resolved_label_a,
        args.resolved_label_b,
        args.column,
        output_dir,
    )
    print_unpaired_cli_summary(group_a, group_b, comparison, args, output_dir)


def run_paired(args: argparse.Namespace) -> None:
    """Runs the complete flow for the paired comparison on the same samples."""
    resolve_comparison_inputs(args)
    output_dir = resolve_comparison_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    higher_is_better = args.resolved_higher_is_better

    merged = align_paired_frames(
        csv_a=args.resolved_csv_a,
        csv_b=args.resolved_csv_b,
        sample_id_column=args.sample_id_column,
        metric_column=args.column,
    )
    summary = compute_paired_summary(
        merged=merged,
        label_a=args.resolved_label_a,
        label_b=args.resolved_label_b,
        tolerance=args.tolerance,
        higher_is_better=higher_is_better,
    )

    save_paired_comparison_summary(summary, args, output_dir)
    save_paired_sample_deltas(merged, args, output_dir)
    save_paired_summary_json(summary, output_dir)
    save_paired_report_txt(summary, args, output_dir)

    raw_delta = merged["value_b"].to_numpy(dtype=float) - merged["value_a"].to_numpy(dtype=float)
    signed_delta = raw_delta if higher_is_better else -raw_delta
    values_a = merged["value_a"].to_numpy(dtype=float)
    values_b = merged["value_b"].to_numpy(dtype=float)

    min_value, max_value = resolve_plot_range(args)
    edges = np.linspace(min_value, max_value, args.bins + 1)

    plot_distribution_histogram(
        values_a,
        values_b,
        edges,
        args.resolved_label_a,
        args.resolved_label_b,
        args.column,
        output_dir,
    )

    plot_distribution_ecdf(
        values_a,
        values_b,
        args.resolved_label_a,
        args.resolved_label_b,
        args.column,
        output_dir,
    )
    plot_paired_delta_histogram(signed_delta, args.column, output_dir)
    plot_paired_scatter(
        merged,
        args.resolved_label_a,
        args.resolved_label_b,
        args.column,
        output_dir,
    )
    print_paired_cli_summary(summary, args, output_dir)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_help()
        return

    validate_direction(args)
    args.func(args)


if __name__ == "__main__":
    main()
