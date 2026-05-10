from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from virtual_staining.applications.compare_panels import run_from_metrics, run_single


def add_single_subparser(subparsers: Any) -> None:
    single_parser = subparsers.add_parser(
        "single",
        help="Create one comparison panel from source/generated/target images.",
        description="Create one comparison panel from source/generated/target images. "
        "Supported image extensions: .tif, .tiff, .png.",
    )
    single_parser.add_argument(
        "--source-image",
        type=Path,
        required=True,
        help="Path to the real source image.",
    )
    single_parser.add_argument(
        "--target-image",
        type=Path,
        required=True,
        help="Path to the real target image.",
    )
    single_parser.add_argument(
        "--generated-image",
        type=Path,
        required=True,
        help="Path to the generated image.",
    )
    single_parser.add_argument(
        "--save-path",
        type=Path,
        default=None,
        help=(
            "Path where the comparison panel will be saved. If omitted, the script "
            "tries to infer .../results/NAME_RUN/comparisons from --generated-image."
        ),
    )
    single_parser.add_argument(
        "--with-diagnostics",
        action="store_true",
        help="Also save single-case diagnostic plots alongside the comparison panel.",
    )
    single_parser.set_defaults(func=run_single)


def add_from_metrics_subparser(subparsers: Any) -> None:
    metrics_parser = subparsers.add_parser(
        "from-metrics",
        help="Generate representative comparison panels from evaluation CSV files.",
        description="Generate representative comparison panels from evaluation CSV files.",
    )
    metrics_parser.add_argument(
        "--run-path",
        type=Path,
        required=True,
        help="Path to a run directory like local_workspace/results/NAME_RUN.",
    )
    metrics_parser.add_argument(
        "--hide-graphs-path",
        action="store_true",
        help="Do not print the full list of saved aggregated graph paths.",
    )
    metrics_parser.set_defaults(func=run_from_metrics)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vs-compare-panels",
        description=(
            "Create side-by-side comparison panels for paired histology images, "
            "or generate representative panels from evaluation CSV files. "
            "Supported image extensions: .tif, .tiff, .png."
        ),
        epilog=(
            "Examples:\n"
            "  vs-compare-panels single\n"
            "      --source-image local_workspace/datasets/your_run/dataset_test/00512_09216_source.tif\n"  # noqa: E501
            "      --generated-image local_workspace/results/your_run/output_test/00512_09216_target_generated.tif\n"  # noqa: E501
            "      --target-image local_workspace/datasets/your_run/dataset_test/00512_09216_target.tif\n"  # noqa: E501
            "      --with-diagnostics\n"
            "\n"
            "  vs-compare-panels from-metrics\n"
            "      --run-path local_workspace/results/your_run\n\n"
            "Use 'vs-compare-panels <command> --help' to see the options "
            "for a specific command."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
        add_help=True,
    )
    subparsers = parser.add_subparsers(dest="mode")
    add_single_subparser(subparsers)
    add_from_metrics_subparser(subparsers)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return
    args.func(args)
