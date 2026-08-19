from __future__ import annotations

import argparse
from pathlib import Path

from virtual_staining.applications.organize import (
    DEFAULT_METRICS,
    OrganizeRequest,
    OrganizeResult,
    organize,
    organize_from_config,
)
from virtual_staining.cli._output import print_info, print_section, style


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vs-organize",
        description=(
            "Organize generated, target and source images by metric ranking. "
            "By default, the script reads RUN/evaluation/per_image_metrics.csv "
            "and writes to RUN/evaluation/sorted_by_metrics/."
        ),
        epilog=(
            "Examples:\n"
            "  vs-organize \\\n"
            "      --run-path local_workspace/results/RUN_NAME \\\n"
            "      --top-k 20\n"
            "\n"
            "  vs-organize \\\n"
            "      --metrics-csv local_workspace/results/RUN_NAME/evaluation/per_image_metrics.csv \\\n"  # noqa: E501
            "      --output-dir local_workspace/results/RUN_NAME/evaluation/sorted_by_metrics \\\n"
            "      --top-k 20 \\\n"
            "      --mode hardlink\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to run config YAML. Uses the config's organize section.",
    )

    parser.add_argument(
        "--run-path",
        type=Path,
        default=None,
        help=(
            "Path to a run directory like local_workspace/results/RUN_NAME. "
            "The script will read RUN/evaluation/per_image_metrics.csv."
        ),
    )
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        default=None,
        help="Path to per_image_metrics.csv. Advanced alternative to --run-path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory where sorted metric folders will be created. "
            "If omitted, defaults to RUN/evaluation/sorted_by_metrics/."
        ),
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=DEFAULT_METRICS,
        help="Metrics to use for sorting.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of best/worst samples to export for each metric.",
    )
    parser.add_argument(
        "--mode",
        choices=["hardlink", "symlink", "copy"],
        default="hardlink",
        help="How to place files in the output folders.",
    )
    parser.add_argument(
        "--include-all-ranked",
        action="store_true",
        help="Also create a full ranked folder for each metric.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing links/files if present.",
    )

    return parser


def _print_result(result: OrganizeResult) -> None:
    print_section("Organize outputs by metric")
    print_info("Metrics CSV", str(result.metrics_csv))
    print_info("Output dir", style(str(result.output_dir), "bold", "magenta"))
    print_info("Mode", result.mode)
    print_info("Top K", str(result.top_k))
    print_info("Image columns", ", ".join(result.image_columns))
    for summary in result.metric_summaries:
        print_info("Organized metric", style(str(summary["metric"]), "green"))
    if result.summary_csv is not None:
        print_info("Summary CSV", str(result.summary_csv))
    print_section("Done")
    print_info("Output written to", style(str(result.output_dir), "bold", "magenta"))


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.config is not None:
        try:
            result = organize_from_config(args.config)
        except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
            raise SystemExit(str(exc)) from exc
        _print_result(result)
        return
    if args.run_path is None and args.metrics_csv is None:
        parser.error("either --config, --run-path, or --metrics-csv is required")
    try:
        result = organize(
            OrganizeRequest(
                run_path=args.run_path,
                metrics_csv=args.metrics_csv,
                output_dir=args.output_dir,
                top_k=args.top_k,
                metrics=tuple(args.metrics),
                mode=args.mode,
                overwrite=args.overwrite,
                include_all_ranked=args.include_all_ranked,
            )
        )
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    _print_result(result)
