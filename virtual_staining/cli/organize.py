from __future__ import annotations

import argparse
from pathlib import Path

from virtual_staining.applications.organize import OrganizeRequest, organize
from virtual_staining.utils.metrics import DEFAULT_METRICS


def _resolve_run_path(run_path: str | Path) -> Path:
    path = Path(run_path).resolve()
    if not path.is_dir():
        raise NotADirectoryError(f"Run directory not found: {path}")
    return path


def _infer_run_path_from_metrics_csv(metrics_csv: str | Path) -> Path | None:
    path = Path(metrics_csv).resolve()
    if path.name == "per_image_metrics.csv" and path.parent.name == "evaluation":
        return path.parent.parent
    return None


def _resolve_metrics_csv(args: argparse.Namespace) -> Path:
    if args.metrics_csv is not None:
        metrics_csv = args.metrics_csv.resolve()
        if not metrics_csv.is_file():
            raise FileNotFoundError(f"CSV not found: {metrics_csv}")
        return metrics_csv

    if args.run_path is None:
        raise ValueError("You must provide either --run-path or --metrics-csv.")

    run_path = _resolve_run_path(args.run_path)
    metrics_csv = run_path / "evaluation" / "per_image_metrics.csv"

    if not metrics_csv.is_file():
        raise FileNotFoundError(f"Could not find per_image_metrics.csv. Expected: {metrics_csv}")

    return metrics_csv


def _resolve_output_dir(args: argparse.Namespace, metrics_csv: Path) -> Path:
    if args.output_dir is not None:
        return args.output_dir.resolve()

    if args.run_path is not None:
        return _resolve_run_path(args.run_path) / "evaluation" / "sorted_by_metrics"

    inferred_run_path = _infer_run_path_from_metrics_csv(metrics_csv)

    if inferred_run_path is not None:
        return inferred_run_path / "evaluation" / "sorted_by_metrics"

    raise ValueError("Could not infer output directory. Please provide --output-dir explicitly.")


def _build_request(args: argparse.Namespace) -> OrganizeRequest:
    metrics_csv = _resolve_metrics_csv(args)
    output_dir = _resolve_output_dir(args, metrics_csv)
    return OrganizeRequest(
        metrics_csv=metrics_csv,
        output_dir=output_dir,
        top_k=args.top_k,
        metrics=tuple(args.metrics),
        mode=args.mode,
        overwrite=args.overwrite,
        include_all_ranked=args.include_all_ranked,
    )


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
    parser = _build_parser()
    args = parser.parse_args(argv)
    request = _build_request(args)
    organize(request)
