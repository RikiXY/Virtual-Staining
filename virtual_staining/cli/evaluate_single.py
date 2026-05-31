from __future__ import annotations

import argparse
import sys
from pathlib import Path

from virtual_staining.applications.evaluate_single import (
    EvaluateSingleRequest,
    SingleEvalResult,
    _resolve_output_dir,
    evaluate_single,
)
from virtual_staining.evaluation.io import extract_single_sample_id
from virtual_staining.utils.console import print_info, print_section, style
from virtual_staining.utils.metrics import color_metric

REMOVED_DATASET_MODE_MESSAGE = (
    "vs-evaluate-single dataset/config mode was removed. "
    "Use 'vs-evaluate --config <run.yaml>' for run-level evaluation, or "
    "'vs-evaluate-single --target-image ... --generated-image ...' "
    "for one target/generated pair."
)
REMOVED_SINGLE_SUBCOMMAND_MESSAGE = (
    "The 'single' subcommand was removed. "
    "Use 'vs-evaluate-single --target-image ... --generated-image ...' instead."
)


def _print_single_result(result: SingleEvalResult) -> None:
    height, width, channels = result.shape
    print_section("Single-pair evaluation")
    print_info("Target", str(result.target))
    print_info("Generated", str(result.generated))
    print_info("Shape", f"{width}x{height}x{channels}")
    print()
    print_info("MAE", color_metric("mae", result.metrics["mae"]))
    print_info("MSE", color_metric("mse", result.metrics["mse"]))
    print_info("RMSE", color_metric("rmse", result.metrics["rmse"]))
    print_info("PSNR", color_metric("psnr", result.metrics["psnr"]))
    print_info("SSIM", color_metric("ssim", result.metrics["ssim"]))
    print_info("PCC gray", color_metric("pcc_gray", result.metrics["pcc_gray"]))
    print_info("PCC RGB mean", color_metric("pcc_rgb_mean", result.metrics["pcc_rgb_mean"]))


def _build_request(args: argparse.Namespace) -> EvaluateSingleRequest:
    target_path = Path(args.target)
    generated_path = Path(args.generated)
    sample_id = extract_single_sample_id(target_path, generated_path)
    output_dir = _resolve_output_dir(args.output_dir, generated_path)
    return EvaluateSingleRequest(
        target_dir=target_path.parent,
        generated_dir=generated_path.parent,
        output_dir=output_dir,
        sample_id=sample_id,
    )


def _cmd_evaluate(args: argparse.Namespace) -> None:
    request = _build_request(args)
    result = evaluate_single(request)
    _print_single_result(result)
    print_section("Saved files")
    print_info("Single evaluation CSV", style(str(result.single_case_csv), "bold", "magenta"))


def _add_image_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--target-image",
        dest="target",
        type=str,
        required=True,
        help="Path to the target image.",
    )
    parser.add_argument(
        "--generated-image",
        dest="generated",
        type=str,
        required=True,
        help="Path to the generated image.",
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        type=str,
        default=None,
        help=(
            "Directory where evaluation outputs will be saved. If omitted, the script "
            "tries to infer .../results/NAME_RUN/evaluation from the generated path."
        ),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vs-evaluate-single",
        description=(
            "Evaluate one generated image against one target image. "
            "Supported image extensions: .tif, .tiff, .png."
        ),
        epilog=(
            "Examples:\n"
            "  vs-evaluate-single\n"
            "      --target-image local_workspace/datasets/your_run/splits/test/00512_09216_target.tif\n"  # noqa: E501
            "      --generated-image local_workspace/results/your_run/artifacts/output_test/00512_09216_target_generated.tif\n"  # noqa: E501
        ),
        formatter_class=argparse.RawTextHelpFormatter,
        add_help=True,
    )
    _add_image_arguments(parser)
    return parser


def _reject_removed_invocation(
    parser: argparse.ArgumentParser,
    argv: list[str],
) -> None:
    if argv and argv[0] == "dataset":
        parser.error(REMOVED_DATASET_MODE_MESSAGE)
    if argv and argv[0] == "single":
        parser.error(REMOVED_SINGLE_SUBCOMMAND_MESSAGE)
    if "--config" in argv or "--hide-graphs-path" in argv:
        parser.error(REMOVED_DATASET_MODE_MESSAGE)


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    parsed_argv = list(sys.argv[1:] if argv is None else argv)
    _reject_removed_invocation(parser, parsed_argv)
    args = parser.parse_args(parsed_argv)
    _cmd_evaluate(args)
