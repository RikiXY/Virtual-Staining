from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from common.cli_style import (
    color_distance,
    color_metric,
    color_pvalue,
    color_share,
    color_signed_delta,
    print_info,
    print_section,
    style,
)
from compare_distributions_lib.core import (
    PairedSummary,
    UnpairedComparison,
    UnpairedGroupStats,
    align_paired_frames,
    compute_paired_summary,
    compute_unpaired_comparison,
    compute_unpaired_group_stats,
    is_higher_better_metric,
    load_metric_values,
    resolve_plot_range,
    resolve_thresholds,
    save_paired_comparison_summary,
    save_paired_sample_deltas,
    save_unpaired_comparison_summary,
    save_unpaired_group_statistics,
    resolve_comparison_inputs,
    resolve_output_dir,
)
from compare_distributions_lib.plots import (
    plot_distribution_ecdf,
    plot_distribution_histogram,
    plot_paired_delta_histogram,
    plot_paired_scatter,
)


def add_common_comparison_arguments(parser: argparse.ArgumentParser) -> None:
    """Aggiunge gli argomenti comuni alle due modalità di confronto."""
    parser.add_argument(
        "--run-a",
        type=Path,
        default=None,
        help=(
            "First run directory. The script will read "
            "RUN_A/evaluation/per_image_metrics.csv."
        ),
    )
    parser.add_argument(
        "--run-b",
        type=Path,
        default=None,
        help=(
            "Second run directory. The script will read "
            "RUN_B/evaluation/per_image_metrics.csv."
        ),
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
            "Directory where outputs will be saved. If omitted, the script saves to "
            "results/comparisons/RUN_A_vs_RUN_B/MODE_METRIC/."
        ),
    )


def add_unpaired_subparser(subparsers: Any) -> None:
    """Aggiunge il sottocomando per distribuzioni non appaiate."""
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
        help="Minimum metric value used for shared histogram bins. If omitted, inferred from metric config.",
    )
    parser.add_argument(
        "--max-value",
        type=float,
        default=None,
        help="Maximum metric value used for shared histogram bins. If omitted, inferred from metric config.",
    )
    parser.add_argument(
        "--thresholds",
        nargs="*",
        type=float,
        default=None,
        help="Thresholds used for share-above or share-below statistics. If omitted, inferred from metric config.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=30,
        help="Number of bins for the common histogram.",
    )
    parser.set_defaults(func=run_unpaired)


def add_paired_subparser(subparsers: Any) -> None:
    """Aggiunge il sottocomando per distribuzioni appaiate sullo stesso sample."""
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
        help="Minimum plausible metric value used for shared histogram bins. If omitted, inferred from metric config.",
    )
    parser.add_argument(
        "--max-value",
        type=float,
        default=None,
        help="Maximum plausible metric value used for shared histogram bins. If omitted, inferred from metric config.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=30,
        help="Number of bins for the comparison histogram.",
    )
    parser.set_defaults(func=run_paired)


def build_parser() -> argparse.ArgumentParser:
    """Costruisce il parser principale e registra i sottocomandi disponibili."""
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


def print_unpaired_group_summary(group: UnpairedGroupStats, metric_name: str) -> None:
    """Stampa il riepilogo CLI di un gruppo non appaiato."""
    print_section(f"Group {group.label}")
    print_info("Samples", str(group.n))
    print_info("Mean", color_metric(metric_name, group.mean))
    print_info("Median", color_metric(metric_name, group.median))
    print_info("IQR", color_distance(group.iqr, 0.05, 0.10))

    for key, value in group.threshold_shares.items():
        print_info(key, color_share(value))

def print_automatic_decision_disclaimer() -> None:
    """Stampa il disclaimer relativo alla decisione automatica."""
    print_info(
        "*",
        (
            "Automatic score-based suggestion, not a definitive statistical conclusion. "
            "Interpret it together with the detailed tables, plots and metric values."
        ),
    )
    
    
def print_unpaired_cli_summary(
    group_a: UnpairedGroupStats,
    group_b: UnpairedGroupStats,
    comparison: UnpairedComparison,
    args: argparse.Namespace,
    output_dir: Path,
) -> None:
    """Stampa in CLI il riepilogo del confronto unpaired."""
    print_section("Input")
    print_info("Metric", args.column)
    print_info("Direction", "higher is better" if args.resolved_higher_is_better else "lower is better")
    print_info("Output dir", str(output_dir))

    print_unpaired_group_summary(group_a, args.column)
    print_unpaired_group_summary(group_b, args.column)

    print_section("Distribution comparison")
    comparison_color = "green" if comparison.better_label != "tie" else "yellow"

    print_info("Score A", f"{comparison.score_a:.2f}")
    print_info("Score B", f"{comparison.score_b:.2f}")
    print_info("Score diff", f"{comparison.score_diff:.2f}")

    print_info(
        "Signed quantile shift",
        color_signed_delta(comparison.signed_quantile_shift),
    )
    print_info("Quantile shift favors", comparison.quantile_shift_favors)
    print_info("Threshold favors", comparison.threshold_favors)
    print_info("Worst tail favors", comparison.worst_tail_favors)
    print_info(
        "Common language B better",
        color_share(comparison.common_language_b_better),
    )
    print_info("Common language favors", comparison.common_language_favors)

    print_info("Wasserstein between groups", color_distance(comparison.wasserstein_between_groups, 0.03, 0.08))
    print_info("KS statistic", color_distance(comparison.ks_statistic, 0.08, 0.18))
    print_info("KS p-value", color_pvalue(comparison.ks_pvalue))
    print_info("Mann-Whitney U", f"{comparison.mannwhitney_u:.6f}")

    print_section("Conclusion")
    print(style(f"Overall better*: {comparison.better_label}", "bold", comparison_color))
    print_automatic_decision_disclaimer()
    print(style(f"Decision strength: {comparison.decision_strength}", "bold", comparison_color))
    print_info("Reason", comparison.reason)
    print(style(f"Saved outputs to: {output_dir}", "bold", "magenta"))


def print_paired_cli_summary(
    summary: PairedSummary,
    args: argparse.Namespace,
    output_dir: Path,
) -> None:
    """Stampa in CLI il riepilogo del confronto paired."""
    print_section("Input")
    print_info("Metric", args.column)
    print_info("Direction", "higher is better" if args.resolved_higher_is_better else "lower is better")
    print_info("Output dir", str(output_dir))

    print_section("Paired comparison")
    print_info("Paired samples", str(summary.n_pairs))
    print_info("Tolerance", f"{summary.tolerance:.6f}")
    print_info("Score A", f"{summary.score_a:.2f}")
    print_info("Score B", f"{summary.score_b:.2f}")
    print_info("Score diff", f"{summary.score_diff:.2f}")
    print_info("Mean signed delta", color_signed_delta(summary.mean_signed_delta))
    print_info("Median signed delta", color_signed_delta(summary.median_signed_delta))
    print_info("Q10 signed delta", color_signed_delta(summary.q10_signed_delta))
    print_info("Q90 signed delta", color_signed_delta(summary.q90_signed_delta))
    print_info(f"Share {summary.label_b} better", color_share(summary.share_b_better))
    print_info(f"Share {summary.label_a} better", color_share(summary.share_a_better))
    print_info("Share equal", color_share(summary.share_equal))
    print_info("Wilcoxon statistic", f"{summary.wilcoxon_statistic:.6f}")
    print_info("Wilcoxon p-value", color_pvalue(summary.wilcoxon_pvalue))
    print_info("Median delta favors", summary.median_delta_favors)
    print_info("Share improvement favors", summary.share_improvement_favors)
    print_info("Worst delta favors", summary.worst_delta_favors)
    print_info("Mean delta favors", summary.mean_delta_favors)
    print_info("Wilcoxon favors", summary.wilcoxon_favors)

    conclusion_color = "green" if summary.better_label != "tie" else "yellow"

    print_section("Conclusion")
    print(style(f"Overall better*: {summary.better_label}", "bold", conclusion_color))
    print_automatic_decision_disclaimer()
    print(style(f"Decision strength: {summary.decision_strength}", "bold", conclusion_color))
    print_info("Reason", summary.reason)
    print(style(f"Saved outputs to: {output_dir}", "bold", "magenta"))


def run_unpaired(args: argparse.Namespace) -> None:
    """Esegue il flusso completo per il confronto tra distribuzioni non appaiate."""
    resolve_comparison_inputs(args)

    output_dir = resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    values_a = load_metric_values(args.resolved_csv_a, args.column)
    values_b = load_metric_values(args.resolved_csv_b, args.column)

    higher_is_better = args.resolved_higher_is_better
    thresholds = resolve_thresholds(args)

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

    min_value, max_value = resolve_plot_range(args)
    edges = np.linspace(min_value, max_value, args.bins + 1)

    save_unpaired_group_statistics(group_a, group_b, output_dir)
    save_unpaired_comparison_summary(group_a, group_b, comparison, args, output_dir)

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
    """Esegue il flusso completo per il confronto paired sullo stesso sample."""
    resolve_comparison_inputs(args)

    output_dir = resolve_output_dir(args)
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
    plot_paired_delta_histogram(
        signed_delta, 
        args.column, 
        output_dir
    )
    plot_paired_scatter(
        merged, 
        args.resolved_label_a, 
        args.resolved_label_b, 
        args.column, 
        output_dir
    )

    print_paired_cli_summary(summary, args, output_dir)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_help()
        return

    args.resolved_higher_is_better = is_higher_better_metric(args.column)
    args.func(args)


if __name__ == "__main__":
    main()