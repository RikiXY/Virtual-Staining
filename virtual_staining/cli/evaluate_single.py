from __future__ import annotations

import argparse
import math
import statistics
from pathlib import Path
from typing import Any

from virtual_staining.applications.evaluate_single import (
    DatasetEvalResult,
    EvaluateSingleRequest,
    SingleEvalResult,
    _resolve_output_dir,
    evaluate_single,
)
from virtual_staining.config.run import RunConfig
from virtual_staining.evaluation.io import extract_single_sample_id
from virtual_staining.evaluation.summaries import metric_value
from virtual_staining.experiment.run_paths import RunPaths
from virtual_staining.utils.console import print_info, print_section, style
from virtual_staining.utils.metrics import DEFAULT_METRICS, color_metric

METRIC_NAMES = list(DEFAULT_METRICS)


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


def _print_dataset_summary(result: DatasetEvalResult) -> None:
    print_section("Dataset evaluation")
    print_info("Targets found", str(len(result.target_files)))
    print_info("Generated found", str(len(result.generated_files)))

    pairs_color = "green" if result.per_image_rows else "red"
    skipped_color = "green" if not result.skipped_rows else "yellow"
    print_info("Pairs evaluated", style(str(len(result.per_image_rows)), pairs_color))
    print_info("Skipped", style(str(len(result.skipped_rows)), skipped_color))

    if result.per_image_rows:
        print_section("Metric summary")
        for metric in METRIC_NAMES:
            values = [metric_value(row, metric) for row in result.per_image_rows]
            finite = [v for v in values if math.isfinite(v)]
            if not finite:
                continue
            print_info(
                f"{metric.upper()} mean", color_metric(metric, float(statistics.mean(finite)))
            )
            print_info(
                f"{metric.upper()} median",
                color_metric(metric, float(statistics.median(finite))),
            )

    print_section("Saved files")
    print_info("Evaluation dir", style(str(result.output_dir), "bold", "magenta"))


def _build_single_request(args: argparse.Namespace) -> EvaluateSingleRequest:
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


def _build_dataset_request(args: argparse.Namespace) -> EvaluateSingleRequest:
    config = RunConfig.from_yaml(Path(args.config).resolve())
    eval_cfg = config.evaluation
    run_root = config.project.run_root
    paths = RunPaths(run_root)
    target_dir = (
        eval_cfg.target_dir
        if eval_cfg and eval_cfg.target_dir
        else config.project.split_dir("test")
    )
    generated_dir = (
        eval_cfg.generated_dir if eval_cfg and eval_cfg.generated_dir else paths.output_test_dir
    )
    output_dir = (
        eval_cfg.output_dir if eval_cfg and eval_cfg.output_dir else run_root / "evaluation"
    )
    save_graphs = eval_cfg.save_graphs if eval_cfg else False
    return EvaluateSingleRequest(
        target_dir=target_dir,
        generated_dir=generated_dir,
        output_dir=output_dir,
        sample_id=None,
        save_graphs=save_graphs,
    )


def _cmd_single(args: argparse.Namespace) -> None:
    request = _build_single_request(args)
    result = evaluate_single(request)
    assert isinstance(result, SingleEvalResult)
    _print_single_result(result)
    print_section("Saved files")
    print_info("Single evaluation CSV", style(str(result.single_case_csv), "bold", "magenta"))


def _cmd_dataset(args: argparse.Namespace) -> None:
    request = _build_dataset_request(args)
    result = evaluate_single(request)
    assert isinstance(result, DatasetEvalResult)
    print_section("Saved files")
    print_info("Per-image metrics", str(result.per_image_csv))
    print_info("Summary", str(result.summary_csv))
    print_info("Skipped samples", str(result.skipped_csv))
    if result.plot_paths and not args.hide_graphs_path:
        for plot_path in result.plot_paths:
            print_info("Graph", str(plot_path))
    _print_dataset_summary(result)


def _add_single_subparser(subparsers: Any) -> None:
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
    single_parser.set_defaults(func=_cmd_single)


def _add_dataset_subparser(subparsers: Any) -> None:
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
        required=True,
        help="Path to the run config YAML.",
    )
    dataset_parser.add_argument(
        "--hide-graphs-path",
        action="store_true",
        help="Do not print the full list of saved graph paths.",
    )
    dataset_parser.set_defaults(func=_cmd_dataset)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vs-evaluate-single",
        description=(
            "Evaluate generated images against target images. "
            "Supported image extensions: .tif, .tiff, .png."
        ),
        epilog=(
            "Examples:\n"
            "  vs-evaluate-single single\n"
            "      --target-image local_workspace/datasets/your_run/splits/test/00512_09216_target.tif\n"  # noqa: E501
            "      --generated-image local_workspace/results/your_run/artifacts/output_test/00512_09216_target_generated.tif\n"  # noqa: E501
            "\n"
            "  vs-evaluate-single dataset\n"
            "      --config config/runs/example.yaml\n\n"
            "Use 'vs-evaluate-single <command> --help' to see the options "
            "for a specific command."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
        add_help=True,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to the run config YAML. Runs the dataset evaluation mode.",
    )
    parser.add_argument(
        "--hide-graphs-path",
        action="store_true",
        help="Do not print the full list of saved graph paths for config-driven dataset mode.",
    )
    subparsers = parser.add_subparsers(dest="mode")
    _add_single_subparser(subparsers)
    _add_dataset_subparser(subparsers)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.mode == "single":
        args.func(args)
        return
    if args.config is not None:
        _cmd_dataset(args)
        return
    if not hasattr(args, "func"):
        parser.error("either --config or an evaluate-single mode is required")
    args.func(args)
