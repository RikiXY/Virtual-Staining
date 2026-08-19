from __future__ import annotations

import argparse
from pathlib import Path

from virtual_staining.applications.evaluate_single import SingleEvalResult, evaluate_pair
from virtual_staining.applications.pipeline import run_stage
from virtual_staining.cli._common import add_log_level_argument, configure_logging
from virtual_staining.cli._output import color_metric, print_info, print_section, style


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vs evaluate",
        description="Evaluate a configured run or one target/generated image pair.",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--config", help="Path to the run config YAML.")
    mode.add_argument(
        "--pair",
        nargs=2,
        type=Path,
        metavar=("TARGET", "GENERATED"),
        help="Target and generated image paths to evaluate.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for --pair (inferred from GENERATED when omitted).",
    )
    add_log_level_argument(parser)
    return parser


def _print_pair_result(result: SingleEvalResult) -> None:
    height, width, channels = result.shape
    print_section("Single-pair evaluation")
    print_info("Target", str(result.target))
    print_info("Generated", str(result.generated))
    print_info("Shape", f"{width}x{height}x{channels}")
    for metric in ("mae", "mse", "rmse", "psnr", "ssim", "pcc_gray", "pcc_rgb_mean"):
        print_info(metric.upper().replace("_", " "), color_metric(metric, result.metrics[metric]))
    print_section("Saved files")
    print_info("Single evaluation CSV", style(str(result.single_case_csv), "bold", "magenta"))


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.output_dir is not None and args.pair is None:
        parser.error("--output-dir requires --pair")
    configure_logging(args.log_level)
    if args.config is not None:
        run_stage(Path(args.config).resolve(), "evaluate")
        return
    target, generated = args.pair
    _print_pair_result(evaluate_pair(target, generated, args.output_dir))


if __name__ == "__main__":
    main()
