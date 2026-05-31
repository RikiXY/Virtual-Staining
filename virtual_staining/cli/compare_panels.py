from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from virtual_staining.applications.compare_panels import (
    ComparePanelsRequest,
    FromMetricsResult,
    SinglePanelResult,
    compare_panels,
)
from virtual_staining.config.run import RunConfig
from virtual_staining.utils.console import print_info, print_section, style
from virtual_staining.utils.metrics import color_for_metric


def _print_single_summary(result: SinglePanelResult) -> None:
    print_section("Single comparison")
    print_info("Saved comparison image", style(str(result.saved_path), "green"))
    for diagnostic_path in result.diagnostic_paths:
        print_info("Saved diagnostic plot", style(str(diagnostic_path), "magenta"))


def _print_metric_based_selection(
    metric_name: str, ranked_rows: dict[str, list[dict[str, str]]]
) -> None:
    print_section(f"Metric {metric_name.upper()}")
    for kind, rows in ranked_rows.items():
        for rank, row in enumerate(rows, start=1):
            metric_value = float(row[metric_name])
            sample_id = row["sample_id"]
            color = color_for_metric(metric_name, metric_value)
            print_info(
                f"{kind.upper()} sample #{rank}",
                style(f"{sample_id} | value={metric_value:.6f}", color),
            )


def _print_metric_run_header(result: FromMetricsResult) -> None:
    print_section("Metric-based representative comparisons")
    print_info("Run path", str(result.run_path))
    print_info("Metrics found", ", ".join(result.available_metrics))


def _print_metric_saved_files(result: FromMetricsResult) -> None:
    print_section("Saved files")
    print_info("Metric-based comparisons", style(str(result.metrics_dir), "bold", "magenta"))


def _cmd_single(args: argparse.Namespace) -> None:
    request = ComparePanelsRequest(
        mode="single",
        source_image=args.source_image,
        generated_image=args.generated_image,
        target_image=args.target_image,
        save_path=args.save_path,
        with_diagnostics=args.with_diagnostics,
    )
    result = compare_panels(request)
    assert isinstance(result, SinglePanelResult)
    _print_single_summary(result)


def _print_from_metrics_summary(result: FromMetricsResult, hide_graphs_path: bool) -> None:
    _print_metric_run_header(result)
    for metric_name in result.available_metrics:
        ranked_rows = result.per_metric_ranked_rows.get(metric_name) or {
            kind: [row] for kind, row in result.per_metric_representative_rows[metric_name].items()
        }
        _print_metric_based_selection(metric_name, ranked_rows)
    if not hide_graphs_path:
        print_section("Saved aggregated panels")
        for aggregated_path in result.saved_aggregated_paths:
            print_info("Saved aggregated panel", str(aggregated_path))
    _print_metric_saved_files(result)


def _cmd_from_metrics(args: argparse.Namespace) -> None:
    request = ComparePanelsRequest(
        mode="from_metrics",
        run_path=args.run_path,
        metrics=tuple(args.metrics) if args.metrics is not None else None,
        kinds=tuple(args.kinds),
        top_k=args.top_k,
    )
    result = compare_panels(request)
    assert isinstance(result, FromMetricsResult)
    _print_from_metrics_summary(result, args.hide_graphs_path)


def _add_single_subparser(subparsers: Any) -> None:
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
    single_parser.set_defaults(func=_cmd_single)


def _add_from_metrics_subparser(subparsers: Any) -> None:
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
    metrics_parser.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        help="Optional metric names to render. Defaults to all supported metrics in summary.csv.",
    )
    metrics_parser.add_argument(
        "--kinds",
        nargs="+",
        choices=("best", "median", "worst"),
        default=("best", "median", "worst"),
        help="Representative kinds to render. Defaults to best median worst.",
    )
    metrics_parser.add_argument(
        "--top-k",
        type=int,
        default=1,
        help="Number of ranked samples to render per metric/kind. Defaults to 1.",
    )
    metrics_parser.set_defaults(func=_cmd_from_metrics)


def _build_parser() -> argparse.ArgumentParser:
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
            "      --source-image local_workspace/datasets/your_run/splits/test/00512_09216_source.tif\n"  # noqa: E501
            "      --generated-image local_workspace/results/your_run/artifacts/output_test/00512_09216_target_generated.tif\n"  # noqa: E501
            "      --target-image local_workspace/datasets/your_run/splits/test/00512_09216_target.tif\n"  # noqa: E501
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
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to run config YAML. Uses the config's compare_panels section.",
    )
    subparsers = parser.add_subparsers(dest="mode")
    _add_single_subparser(subparsers)
    _add_from_metrics_subparser(subparsers)
    return parser


def _run_from_config(config_path: Path) -> None:
    config = RunConfig.from_yaml(config_path.resolve())
    panels_cfg = config.compare_panels
    if panels_cfg is None:
        raise SystemExit("Config has no 'compare_panels' section.")

    if panels_cfg.mode == "single":
        if (
            panels_cfg.source_image is None
            or panels_cfg.generated_image is None
            or panels_cfg.target_image is None
        ):
            raise SystemExit(
                "compare_panels.single requires source_image, generated_image, and target_image."
            )
        request = ComparePanelsRequest(
            mode="single",
            source_image=panels_cfg.source_image,
            generated_image=panels_cfg.generated_image,
            target_image=panels_cfg.target_image,
            save_path=panels_cfg.save_path,
            with_diagnostics=panels_cfg.with_diagnostics,
        )
        result = compare_panels(request)
        assert isinstance(result, SinglePanelResult)
        _print_single_summary(result)
        return

    request = ComparePanelsRequest(
        mode="from_metrics",
        run_path=(
            panels_cfg.run_path if panels_cfg.run_path is not None else config.project.run_root
        ),
        metrics=panels_cfg.metrics,
        kinds=panels_cfg.kinds,
        top_k=panels_cfg.top_k,
    )
    result = compare_panels(request)
    assert isinstance(result, FromMetricsResult)
    _print_from_metrics_summary(result, panels_cfg.hide_graphs_path)


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.config is not None:
        _run_from_config(args.config)
        return
    if not hasattr(args, "func"):
        parser.error("either --config or a compare-panels mode is required")
    args.func(args)
