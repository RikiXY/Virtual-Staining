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


def compare_unpaired(
    args: argparse.Namespace,
) -> tuple[UnpairedGroupStats, UnpairedGroupStats, UnpairedComparison, Path]:
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

    return group_a, group_b, comparison, output_dir


def compare_paired(args: argparse.Namespace) -> tuple[PairedSummary, Path]:
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

    return summary, output_dir
