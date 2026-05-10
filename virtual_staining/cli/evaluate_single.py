from __future__ import annotations

import argparse
from typing import Any

from virtual_staining.applications.evaluate_single import run_dataset, run_single


def add_single_subparser(subparsers: Any) -> None:
    single_parser = subparsers.add_parser(
        "single",
        help="Evaluate one target/generated image pair.",
        description=(
            "Compute MAE, MSE, RMSE, PSNR, SSIM and PCC for one target/generated pair. "
            "Supported image extensions: .tif, .tiff, .png."
        ),
    )
    single_parser.add_argument(
        "--target-image",
        dest="target",
        type=str,
        required=True,
        help="Path to the target image.",
    )
    single_parser.add_argument(
        "--generated-image",
        dest="generated",
        type=str,
        required=True,
        help="Path to the generated image.",
    )
    single_parser.add_argument(
        "--output-dir",
        dest="output_dir",
        type=str,
        default=None,
        help=(
            "Directory where evaluation outputs will be saved. If omitted, the script "
            "tries to infer .../results/NAME_RUN/evaluation from the generated path."
        ),
    )
    single_parser.set_defaults(func=run_single)


def add_dataset_subparser(subparsers: Any) -> None:
    dataset_parser = subparsers.add_parser(
        "dataset",
        help="Evaluate all matching target/generated pairs in two folders.",
        description=(
            "Compute MAE, MSE, RMSE, PSNR, SSIM and PCC for all matching pairs in a dataset. "
            "Supported image extensions: .tif, .tiff, .png."
        ),
    )
    dataset_parser.add_argument(
        "--config",
        type=str,
        default="config/runs/example.yaml",
        help="path to the run config YAML (default: config/runs/example.yaml)",
    )
    dataset_parser.add_argument(
        "--hide-graphs-path",
        action="store_true",
        help="Do not print the full list of saved graph paths.",
    )
    dataset_parser.set_defaults(func=run_dataset)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vs-evaluate-single",
        description=(
            "Evaluate generated images against target images. "
            "Supported image extensions: .tif, .tiff, .png."
        ),
        epilog=(
            "Examples:\n"
            "  vs-evaluate-single single\n"
            "      --target-image local_workspace/datasets/your_run/dataset_test/00512_09216_target.tif\n"  # noqa: E501
            "      --generated-image local_workspace/results/your_run/output_test/00512_09216_target_generated.tif\n"  # noqa: E501
            "\n"
            "  vs-evaluate-single dataset\n"
            "      --config config/runs/example.yaml\n\n"
            "Use 'vs-evaluate-single <command> --help' to see the options "
            "for a specific command."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
        add_help=True,
    )
    subparsers = parser.add_subparsers(dest="mode")
    add_single_subparser(subparsers)
    add_dataset_subparser(subparsers)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return
    args.func(args)
