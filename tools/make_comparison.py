from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from virtual_staining.evaluation.panels import (
    METRIC_SELECTION_ORDER,
    DiagnosticEntry,
    build_metric_case_artifacts,
    extract_generated_sample_id,
    save_comparison_panel,
    save_diagnostic_plots,
    save_metric_diagnostics_summary,
    select_representative_rows,
    write_metric_selection_summary,
)
from virtual_staining.evaluation.summaries import read_per_image_metrics_csv, read_summary_csv
from virtual_staining.utils.cli import print_info, print_section, style
from virtual_staining.utils.metrics import color_for_metric

# ==========================
# Section dedicated to the parser
# ==========================


def add_single_subparser(subparsers: Any) -> None:
    """Adds the subcommand for comparing a single pair."""
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
    """Adds the subcommand for representative panels from CSV files."""
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
    """Builds the main parser and registers the available subcommands."""
    parser = argparse.ArgumentParser(
        prog="python tools/make_comparison.py",
        description=(
            "Create side-by-side comparison panels for paired histology images, "
            "or generate representative panels from evaluation CSV files. "
            "Supported image extensions: .tif, .tiff, .png."
        ),
        epilog=(
            "Examples:\n"
            "  python tools/make_comparison.py single\n"
            "      --source-image local_workspace/datasets/your_run/dataset_test/00512_09216_source.tif\n"  # noqa: E501
            "      --generated-image local_workspace/results/your_run/output_test/00512_09216_target_generated.tif\n"  # noqa: E501
            "      --target-image local_workspace/datasets/your_run/dataset_test/00512_09216_target.tif\n"  # noqa: E501
            "      --with-diagnostics\n"
            "\n"
            "  python tools/make_comparison.py from-metrics\n"
            "      --run-path local_workspace/results/your_run\n\n"
            "Use 'python tools/make_comparison.py <command> --help' to see the options "
            "for a specific command."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
        add_help=True,
    )
    subparsers = parser.add_subparsers(dest="mode")
    add_single_subparser(subparsers)
    add_from_metrics_subparser(subparsers)
    return parser


def infer_run_dir_from_generated_path(generated_path: str | Path) -> Path:
    """Tries to derive the run directory from a generated path inside results/."""
    path = Path(generated_path).resolve()
    base = path.parent if path.is_file() else path
    parts = base.parts

    if "results" not in parts:
        raise ValueError(
            "Could not infer run directory from generated path. Expected a path like "
            ".../results/NAME_RUN/output_test/..."
        )

    results_index = parts.index("results")

    if results_index + 1 >= len(parts):
        raise ValueError(
            "Could not infer NAME_RUN from generated path. Expected a path like "
            ".../results/NAME_RUN/output_test/..."
        )

    run_dir = Path(*parts[: results_index + 2])

    if run_dir.parent.name != "results":
        raise ValueError(
            "Could not infer a valid run directory inside results/. "
            "Please provide --save-path explicitly."
        )

    return run_dir


def infer_default_save_path(generated_image: str | Path) -> Path:
    """Builds the default save path for a single comparison."""
    generated_path = Path(generated_image)
    sample_id = extract_generated_sample_id(generated_path)
    run_dir = infer_run_dir_from_generated_path(generated_path)
    return run_dir / "comparisons" / f"{sample_id}_comparison.png"


def infer_diagnostics_dir(save_path: str | Path) -> Path:
    """Derives the diagnostics directory from the panel save path."""
    save_path = Path(save_path)
    return save_path.parent / "diagnostics"


def infer_case_diagnostics_dir(save_path: str | Path, generated_image: str | Path) -> Path:
    """Derives the diagnostics directory for the individual sample."""
    diagnostics_dir = infer_diagnostics_dir(save_path)
    sample_id = extract_generated_sample_id(generated_image)
    return diagnostics_dir / sample_id


# ====================================
# Section dedicated to the text report
# ====================================


def print_single_summary(saved_path: Path, diagnostic_paths: list[Path]) -> None:
    """Prints the final summary of the single mode."""
    print_section("Single comparison")
    print_info("Saved comparison image", style(str(saved_path), "green"))

    for diagnostic_path in diagnostic_paths:
        print_info("Saved diagnostic plot", style(str(diagnostic_path), "magenta"))


def print_metric_based_selection(
    metric_name: str, representative_rows: dict[str, dict[str, str]]
) -> None:
    """Prints the representative samples chosen for a metric."""
    print_section(f"Metric {metric_name.upper()}")

    for kind, row in representative_rows.items():
        metric_value = float(row[metric_name])
        sample_id = row["sample_id"]
        color = color_for_metric(metric_name, metric_value)
        print_info(
            f"{kind.upper()} sample",
            style(f"{sample_id} | value={metric_value:.6f}", color),
        )


def print_metric_run_header(run_path: Path, available_metrics: list[str]) -> None:
    """Prints the general header for the from-metrics mode."""
    print_section("Metric-based representative comparisons")
    print_info("Run path", str(run_path))
    print_info("Metrics found", ", ".join(available_metrics))


def print_metric_saved_files(metrics_dir: Path) -> None:
    """Prints the final summary of files saved in from-metrics mode."""
    print_section("Saved files")
    print_info("Metric-based comparisons", style(str(metrics_dir), "bold", "magenta"))


# =====================================
# Section dedicated to the main flow
# =====================================


def run_single(args: argparse.Namespace) -> None:
    """Runs the complete flow for comparing a single pair."""
    if args.save_path is not None:
        save_path = args.save_path
    else:
        save_path = infer_default_save_path(args.generated_image)

    saved_path = save_comparison_panel(
        source_path=args.source_image,
        generated_path=args.generated_image,
        target_path=args.target_image,
        save_path=save_path,
    )

    diagnostic_paths: list[Path] = []

    if args.with_diagnostics:
        diagnostics_dir = infer_case_diagnostics_dir(
            save_path=saved_path,
            generated_image=args.generated_image,
        )
        diagnostic_paths = save_diagnostic_plots(
            source_path=args.source_image,
            generated_path=args.generated_image,
            target_path=args.target_image,
            save_dir=diagnostics_dir,
        )

    print_single_summary(saved_path, diagnostic_paths)


def run_from_metrics(args: argparse.Namespace) -> None:
    """Runs the complete flow for comparisons selected from the CSV files."""
    run_path = args.run_path.resolve()
    evaluation_dir = run_path / "evaluation"
    summary_csv = evaluation_dir / "summary.csv"
    per_image_csv = evaluation_dir / "per_image_metrics.csv"
    summary_rows = read_summary_csv(summary_csv)
    per_image_rows = read_per_image_metrics_csv(per_image_csv)
    metrics_dir = run_path / "comparisons" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    selection_summary_rows: list[dict[str, object]] = []
    saved_aggregated_paths: list[Path] = []
    available_metrics = [metric for metric in METRIC_SELECTION_ORDER if metric in summary_rows]

    if not available_metrics:
        raise ValueError(
            f"No supported metrics found in {summary_csv}. "
            f"Expected one of: {', '.join(METRIC_SELECTION_ORDER)}"
        )

    print_metric_run_header(run_path, available_metrics)

    for metric_name in available_metrics:
        metric_summary = summary_rows[metric_name]
        metric_dir = metrics_dir / metric_name
        metric_dir.mkdir(parents=True, exist_ok=True)
        representative_rows = select_representative_rows(
            metric_name,
            metric_summary,
            per_image_rows,
        )
        metric_selection_rows: list[dict[str, object]] = []
        metric_diagnostic_entries: list[DiagnosticEntry] = []

        print_metric_based_selection(metric_name, representative_rows)

        for kind, row in representative_rows.items():
            selection_row, diagnostic_entry = build_metric_case_artifacts(
                metric_name=metric_name,
                kind=kind,
                row=row,
                metric_summary=metric_summary,
                metric_dir=metric_dir,
            )
            selection_summary_rows.append(selection_row)
            metric_selection_rows.append(selection_row)
            metric_diagnostic_entries.append(diagnostic_entry)

        write_metric_selection_summary(metric_selection_rows, metric_dir / "selection_summary.csv")
        kind_order = {"best": 0, "median": 1, "worst": 2}
        metric_diagnostic_entries.sort(key=lambda entry: kind_order[entry["kind"]])
        aggregated_paths = save_metric_diagnostics_summary(
            metric_name=metric_name,
            metric_dir=metric_dir,
            diagnostic_entries=metric_diagnostic_entries,
        )
        saved_aggregated_paths.extend(aggregated_paths)

    if not args.hide_graphs_path:
        print_section("Saved aggregated panels")
        for aggregated_path in saved_aggregated_paths:
            print_info("Saved aggregated panel", str(aggregated_path))

    write_metric_selection_summary(
        selection_summary_rows,
        metrics_dir / "metrics_selection_summary.csv",
    )
    print_metric_saved_files(metrics_dir)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()
