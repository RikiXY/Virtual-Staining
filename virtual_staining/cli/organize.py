from __future__ import annotations

import argparse
from pathlib import Path

from virtual_staining.applications.organize import run_organize
from virtual_staining.utils.metrics import DEFAULT_METRICS


def build_parser() -> argparse.ArgumentParser:
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


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_organize(args)
