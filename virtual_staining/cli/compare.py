from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from virtual_staining.applications.compare import compare_paired, compare_unpaired
from virtual_staining.evaluation.statistics import (
    PairedSummary,
    UnpairedComparison,
    UnpairedGroupStats,
    resolve_metric_direction,
)
from virtual_staining.utils.console import print_info, print_section, style
from virtual_staining.utils.metrics import color_metric_value


def color_distance(value: float, good: float, warn: float) -> str:
    if value <= good:
        return style(f"{value:.6f}", "green")
    if value <= warn:
        return style(f"{value:.6f}", "yellow")
    if value <= warn * 1.5:
        return style(f"{value:.6f}", "orange")
    return style(f"{value:.6f}", "red")


def color_pvalue(value: float) -> str:
    if value < 0.001:
        return style(f"{value:.6g}", "green")
    if value < 0.01:
        return style(f"{value:.6g}", "yellow")
    if value < 0.05:
        return style(f"{value:.6g}", "orange")
    return style(f"{value:.6g}", "red")


def color_share(value: float) -> str:
    if value >= 0.70:
        return style(f"{value:.6f}", "green")
    if value >= 0.40:
        return style(f"{value:.6f}", "yellow")
    if value >= 0.20:
        return style(f"{value:.6f}", "orange")
    return style(f"{value:.6f}", "red")


def color_signed_delta(value: float) -> str:
    if value > 0:
        return style(f"{value:.6f}", "green")
    if value < 0:
        return style(f"{value:.6f}", "red")
    return style(f"{value:.6f}", "yellow")


def print_unpaired_group_summary(group: UnpairedGroupStats, metric_name: str) -> None:
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


def add_direction_arguments(parser: argparse.ArgumentParser) -> None:
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
    parser.set_defaults(mode="unpaired")


def add_paired_subparser(subparsers: Any) -> None:
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
    parser.set_defaults(mode="paired")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vs-compare",
        description=(
            "Compare metric distributions from per-image CSV files. "
            "Supports both unpaired and paired comparisons."
        ),
        epilog=(
            "Examples:\n"
            "  vs-compare unpaired \\\n"
            "      --run-a local_workspace/results/run_a \\\n"
            "      --run-b local_workspace/results/run_b \\\n"
            "      --column ssim\n"
            "\n"
            "  vs-compare paired \\\n"
            "      --run-a local_workspace/results/L1-25 \\\n"
            "      --run-b local_workspace/results/L1-31 \\\n"
            "      --column ssim\n"
            "\n"
            "  vs-compare paired \\\n"
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


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.mode is None:
        parser.print_help()
        return

    try:
        args.resolved_higher_is_better = resolve_metric_direction(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    if args.mode == "unpaired":
        group_a, group_b, comparison, output_dir = compare_unpaired(args)
        print_unpaired_cli_summary(group_a, group_b, comparison, args, output_dir)
    elif args.mode == "paired":
        summary, output_dir = compare_paired(args)
        print_paired_cli_summary(summary, args, output_dir)
    else:
        raise SystemExit(f"Unsupported comparison mode: {args.mode}")
