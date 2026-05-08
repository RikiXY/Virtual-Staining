from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from common.cli_style import print_info, print_section
from make_comparison_lib.core import (
    HIGHER_IS_BETTER,
    METRIC_SELECTION_ORDER,
    build_selection_summary_row,
    infer_case_diagnostics_dir,
    infer_default_save_path,
    infer_source_path_from_row,
    print_metric_based_selection,
    print_metric_run_header,
    print_metric_saved_files,
    print_single_summary,
    read_per_image_metrics_csv,
    read_summary_csv,
    select_representative_rows,
    write_metric_selection_summary,
)
from make_comparison_lib.plots import (
    save_comparison_panel,
    save_diagnostic_plots,
    save_metric_diagnostics_summary,
)


def add_single_subparser(subparsers: Any) -> None:
    """Aggiunge il sottocomando per il confronto di una singola coppia."""
    single_parser = subparsers.add_parser(
        "single",
        help="Create one comparison panel from source/generated/target images.",
        description=(
            "Create one comparison panel from source/generated/target images. "
            "Supported image extensions: .tif, .tiff, .png."
        ),
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
    """Aggiunge il sottocomando per i pannelli rappresentativi da CSV."""
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
    """Costruisce il parser principale e registra i sottocomandi disponibili."""
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
            "      --source-image local_workspace/datasets/your_run/dataset_test/00512_09216_source.tif\n"
            "      --generated-image local_workspace/results/your_run/output_test/00512_09216_target_generated.tif\n"
            "      --target-image local_workspace/datasets/your_run/dataset_test/00512_09216_target.tif\n"
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


def run_single(args: argparse.Namespace) -> None:
    """Esegue il flusso completo per il confronto di una singola coppia."""
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
) -> tuple[dict[str, object], dict[str, object]]:
    """Costruisce e salva gli artefatti relativi a un caso rappresentativo."""
    sample_id = row["sample_id"]
    metric_value = float(row[metric_name])

    if kind == "best":
        summary_key = "max" if HIGHER_IS_BETTER[metric_name] else "min"
    elif kind == "worst":
        summary_key = "min" if HIGHER_IS_BETTER[metric_name] else "max"
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
        suptitle=None,
    )

    diagnostics_case_dir = metric_dir / "diagnostics" / f"{kind}_{sample_id}"
    diagnostic_paths = save_diagnostic_plots(
        source_path=source_path,
        generated_path=generated_path,
        target_path=target_path,
        save_dir=diagnostics_case_dir,
    )

    diagnostic_paths_by_name = {path.name: path for path in diagnostic_paths}

    diagnostic_entry = {
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
    """Esegue il flusso completo per i confronti selezionati a partire dai CSV."""
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
        metric_diagnostic_entries: list[dict[str, object]] = []

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

        write_metric_selection_summary(
            metric_selection_rows,
            metric_dir / "selection_summary.csv",
        )

        kind_order = {"best": 0, "median": 1, "worst": 2}
        metric_diagnostic_entries.sort(key=lambda entry: kind_order[entry["kind"]])

        aggregated_paths = save_metric_diagnostics_summary(
            metric_name=metric_name,
            metric_dir=metric_dir,
            diagnostic_entries=metric_diagnostic_entries,
        )

        saved_aggregated_paths.extend(aggregated_paths)

    write_metric_selection_summary(
        selection_summary_rows,
        metrics_dir / "metrics_selection_summary.csv",
    )

    if not args.hide_graphs_path:
        print_section("Saved aggregated panels")
        for aggregated_path in saved_aggregated_paths:
            print_info("Saved aggregated panel", str(aggregated_path))

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