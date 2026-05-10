from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from virtual_staining.applications.compare import compare_paired, compare_unpaired
from virtual_staining.evaluation.statistics import resolve_metric_direction


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
        compare_unpaired(args)
    elif args.mode == "paired":
        compare_paired(args)
    else:
        raise SystemExit(f"Unsupported comparison mode: {args.mode}")
