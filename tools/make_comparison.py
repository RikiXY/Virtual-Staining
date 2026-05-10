from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from virtual_staining.evaluation.panels import (
    METRIC_SELECTION_ORDER,
    DiagnosticEntry,
    build_selection_summary_row,
    extract_generated_sample_id,
    infer_source_path_from_row,
    save_comparison_panel,
    save_diagnostic_plots,
    save_metric_diagnostics_summary,
    select_representative_rows,
    write_metric_selection_summary,
)
from virtual_staining.utils.cli import print_info, print_section, style
from virtual_staining.utils.metrics import color_for_metric, is_higher_better_metric

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
# Section dedicated to CSV reading
# ====================================


def read_summary_csv(path: str | Path) -> dict[str, dict[str, float]]:
    """Reads summary.csv and returns the aggregate statistics per metric."""
    summary_path = Path(path)

    if not summary_path.is_file():
        raise FileNotFoundError(f"Summary CSV not found: {summary_path}")

    rows: dict[str, dict[str, float]] = {}

    with summary_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        header_found = False

        for row in reader:
            if not row:
                continue

            if row[0] == "metric":
                header_found = True
                continue

            if not header_found:
                continue

            metric_name = row[0].strip().lower()
            rows[metric_name] = {
                "count": float(row[1]),
                "mean": float(row[2]),
                "median": float(row[3]),
                "std": float(row[4]),
                "min": float(row[5]),
                "max": float(row[6]),
            }

    return rows


def read_per_image_metrics_csv(path: str | Path) -> list[dict[str, str]]:
    """Reads per_image_metrics.csv and returns all rows as dictionaries."""
    csv_path = Path(path)

    if not csv_path.is_file():
        raise FileNotFoundError(f"Per-image metrics CSV not found: {csv_path}")

    with csv_path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


# ==============================================
# Section dedicated to sample selection
# ==============================================


def build_metric_kind_row_title(
    metric_name: str, kind: str, sample_id: str, metric_value: float
) -> str:
    """Builds the row title for aggregated panels."""
    return f"{metric_name.upper()} | {kind.upper()} | sample={sample_id} | value={metric_value:.6f}"


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


def build_metric_case_artifacts(
    metric_name: str,
    kind: str,
    row: dict[str, str],
    metric_summary: dict[str, float],
    metric_dir: Path,
) -> tuple[dict[str, object], DiagnosticEntry]:
    """Builds and saves the artefacts for a representative case."""
    sample_id = row["sample_id"]
    metric_value = float(row[metric_name])

    if kind == "best":
        summary_key = "max" if is_higher_better_metric(metric_name) else "min"
    elif kind == "worst":
        summary_key = "min" if is_higher_better_metric(metric_name) else "max"
    elif kind == "median":
        summary_key = "median"
    else:
        raise ValueError(f"Unsupported representative kind: {kind}")

    target_value = float(metric_summary[summary_key])
    source_path = infer_source_path_from_row(row)
    generated_path = Path(row["generated_path"])
    target_path = Path(row["target_path"])
    comparison_path = metric_dir / f"{kind}_{sample_id}_comparison.png"
    saved_path = save_comparison_panel(
        source_path=source_path,
        generated_path=generated_path,
        target_path=target_path,
        save_path=comparison_path,
        suptitle=(
            f"{metric_name.upper()} | {kind.upper()} | "
            f"sample={sample_id} | value={metric_value:.6f}"
        ),
    )

    diagnostics_case_dir = metric_dir / "diagnostics" / f"{kind}_{sample_id}"
    diagnostic_paths = save_diagnostic_plots(
        source_path=source_path,
        generated_path=generated_path,
        target_path=target_path,
        save_dir=diagnostics_case_dir,
    )
    diagnostic_paths_by_name = {path.name: path for path in diagnostic_paths}
    diagnostic_entry: DiagnosticEntry = {
        "kind": kind,
        "sample_id": sample_id,
        "metric_value": metric_value,
        "comparison_path": saved_path,
        "error_histogram_path": diagnostic_paths_by_name[f"{sample_id}_error_histogram.png"],
        "intensity_overlay_histogram_path": diagnostic_paths_by_name[
            f"{sample_id}_intensity_overlay_histogram.png"
        ],
        "target_vs_generated_scatter_by_channel_path": diagnostic_paths_by_name[
            f"{sample_id}_target_vs_generated_scatter_by_channel.png"
        ],
    }
    selection_row = build_selection_summary_row(
        metric_name=metric_name,
        kind=kind,
        sample_id=sample_id,
        metric_value=metric_value,
        target_value=target_value,
        source_path=source_path,
        target_path=target_path,
        generated_path=generated_path,
        comparison_path=saved_path,
    )
    return selection_row, diagnostic_entry


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
