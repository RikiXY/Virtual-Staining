from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from virtual_staining.evaluation.comparison import (
    plot_distribution_ecdf,
    plot_distribution_histogram,
    plot_paired_delta_histogram,
    plot_paired_scatter,
    save_paired_comparison_summary,
    save_paired_report_txt,
    save_paired_sample_deltas,
    save_paired_summary_json,
    save_unpaired_comparison_summary,
    save_unpaired_group_statistics,
    save_unpaired_report_txt,
    save_unpaired_summary_json,
)
from virtual_staining.evaluation.statistics import (
    PairedSummary,
    UnpairedComparison,
    UnpairedGroupStats,
    align_paired_frames,
    compute_paired_summary,
    compute_unpaired_comparison,
    compute_unpaired_group_stats,
    load_metric_values,
    resolve_comparison_inputs,
    resolve_comparison_output_dir,
    resolve_plot_range,
    resolve_thresholds,
)
from virtual_staining.utils.cli import print_info, print_section, style
from virtual_staining.utils.metrics import color_metric_value


def color_distance(value: float, good: float, warn: float) -> str:
    """Colour a distance: the smaller, the better."""
    if value <= good:
        return style(f"{value:.6f}", "green")
    if value <= warn:
        return style(f"{value:.6f}", "yellow")
    if value <= warn * 1.5:
        return style(f"{value:.6f}", "orange")
    return style(f"{value:.6f}", "red")


def color_pvalue(value: float) -> str:
    """Colour a p-value according to the strength of evidence for a difference."""
    if value < 0.001:
        return style(f"{value:.6g}", "green")
    if value < 0.01:
        return style(f"{value:.6g}", "yellow")
    if value < 0.05:
        return style(f"{value:.6g}", "orange")
    return style(f"{value:.6g}", "red")


def color_share(value: float) -> str:
    """Colour a share between 0 and 1."""
    if value >= 0.70:
        return style(f"{value:.6f}", "green")
    if value >= 0.40:
        return style(f"{value:.6f}", "yellow")
    if value >= 0.20:
        return style(f"{value:.6f}", "orange")
    return style(f"{value:.6f}", "red")


def color_signed_delta(value: float) -> str:
    """Colour a signed delta: positive favours B, negative favours A."""
    if value > 0:
        return style(f"{value:.6f}", "green")
    if value < 0:
        return style(f"{value:.6f}", "red")
    return style(f"{value:.6f}", "yellow")


def print_unpaired_group_summary(group: UnpairedGroupStats, metric_name: str) -> None:
    """Print the CLI summary of an unpaired group."""
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
    """Print the CLI summary of the unpaired comparison."""
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
    """Print the CLI summary of the paired comparison."""
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


def compare_unpaired(args: argparse.Namespace) -> None:
    """Run the complete flow for the comparison between unpaired distributions."""
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


def compare_paired(args: argparse.Namespace) -> None:
    """Run the complete flow for the paired comparison on the same samples."""
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
