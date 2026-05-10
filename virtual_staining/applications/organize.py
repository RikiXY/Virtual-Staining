from __future__ import annotations

import argparse
from pathlib import Path

from virtual_staining.evaluation.ranking import organize_by_metrics


def resolve_run_path(run_path: str | Path) -> Path:
    """Validates and resolves a run directory."""
    path = Path(run_path).resolve()

    if not path.is_dir():
        raise NotADirectoryError(f"Run directory not found: {path}")

    return path


def resolve_metrics_csv(args: argparse.Namespace) -> Path:
    """Resolves the metrics CSV from --run-path or --metrics-csv."""
    if args.metrics_csv is not None:
        metrics_csv = args.metrics_csv.resolve()

        if not metrics_csv.is_file():
            raise FileNotFoundError(f"CSV not found: {metrics_csv}")

        return metrics_csv

    if args.run_path is None:
        raise ValueError("You must provide either --run-path or --metrics-csv.")

    run_path = resolve_run_path(args.run_path)
    metrics_csv = run_path / "evaluation" / "per_image_metrics.csv"

    if not metrics_csv.is_file():
        raise FileNotFoundError(f"Could not find per_image_metrics.csv. Expected: {metrics_csv}")

    return metrics_csv


def infer_run_path_from_metrics_csv(metrics_csv: str | Path) -> Path | None:
    """Tries to recover the run directory from a metrics CSV path."""
    path = Path(metrics_csv).resolve()

    if path.name == "per_image_metrics.csv" and path.parent.name == "evaluation":
        return path.parent.parent

    return None


def resolve_output_dir(args: argparse.Namespace, metrics_csv: Path) -> Path:
    """Resolves the output directory."""
    if args.output_dir is not None:
        return args.output_dir.resolve()

    if args.run_path is not None:
        return resolve_run_path(args.run_path) / "evaluation" / "sorted_by_metrics"

    inferred_run_path = infer_run_path_from_metrics_csv(metrics_csv)

    if inferred_run_path is not None:
        return inferred_run_path / "evaluation" / "sorted_by_metrics"

    raise ValueError("Could not infer output directory. Please provide --output-dir explicitly.")


def run_organize(args: argparse.Namespace) -> None:
    """Orchestrates the full organize flow."""
    metrics_csv = resolve_metrics_csv(args)
    output_dir = resolve_output_dir(args, metrics_csv)
    output_dir.mkdir(parents=True, exist_ok=True)
    organize_by_metrics(
        csv_path=metrics_csv,
        output_dir=output_dir,
        top_n=args.top_k,
        metrics=list(args.metrics),
        mode=args.mode,
        overwrite=args.overwrite,
        include_all_ranked=args.include_all_ranked,
    )
