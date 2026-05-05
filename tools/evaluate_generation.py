from __future__ import annotations

import argparse
from typing import Any

from common.cli_style import print_info, print_section, style
from evaluate_generation_lib.core import (
    build_metric_row,
    build_summary_rows,
    collect_image_files,
    evaluate_pair,
    extract_single_sample_id,
    print_dataset_summary,
    print_single_result,
    resolve_output_dir,
    write_per_image_metrics_csv,
    write_single_case_csv,
    write_skipped_csv,
    write_summary_csv,
)
from evaluate_generation_lib.plots import save_dataset_plots


def add_single_subparser(subparsers: Any) -> None:
    """Aggiunge il sottocomando per la valutazione di una singola coppia."""
    single_parser = subparsers.add_parser(
        "single",
        help="Evaluate one target/generated image pair.",
        description=(
            "Compute MAE, RMSE, PSNR, SSIM, MSE, and PCC for one target/generated pair. "
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
    single_parser.set_defaults(func=run_single)


def add_dataset_subparser(subparsers: Any) -> None:
    """Aggiunge il sottocomando per la valutazione di un intero dataset."""
    dataset_parser = subparsers.add_parser(
        "dataset",
        help="Evaluate all matching target/generated pairs in two folders.",
        description=(
            "Compute MAE, RMSE, PSNR, SSIM, MSE, and PCC for all matching pairs in a dataset. "
            "Supported image extensions: .tif, .tiff, .png."
        ),
    )
    dataset_parser.add_argument(
        "--target-dir",
        dest="target_dir",
        type=str,
        required=True,
        help=(
            "Directory containing target images with filename stem ending in '_target'. "
            "Supported extensions: .tif, .tiff, .png."
        ),
    )
    dataset_parser.add_argument(
        "--generated-dir",
        dest="generated_dir",
        type=str,
        required=True,
        help=(
            "Directory containing generated images with filename stem ending in '_target_generated'. "
            "Supported extensions: .tif, .tiff, .png."
        ),
    )
    dataset_parser.add_argument(
        "--output-dir",
        dest="output_dir",
        type=str,
        default=None,
        help=(
            "Directory where evaluation outputs will be saved. If omitted, the script "
            "tries to infer .../results/NAME_RUN/evaluation from the generated directory."
        ),
    )
    dataset_parser.add_argument(
        "--save-graphs",
        dest="save_graphs",
        action="store_true",
        help="Save aggregate plots for the evaluated dataset.",
    )
    dataset_parser.add_argument(
        "--hide-graphs-path",
        dest="hide_graphs_path",
        action="store_true",
        help="Do not print the full list of saved graph paths.",
    )
    dataset_parser.set_defaults(func=run_dataset)


def build_parser() -> argparse.ArgumentParser:
    """Costruisce il parser principale e registra i sottocomandi disponibili."""
    parser = argparse.ArgumentParser(
        prog="python tools/evaluate_generation.py",
        description=(
            "Evaluate generated images against target images. "
            "Supported image extensions: .tif, .tiff, .png."
        ),
        epilog=(
            "Examples:\n"
            "  python tools/evaluate_generation.py single\n"
            "      --target-image local_workspace/datasets/your_run/dataset_test/00512_09216_target.tif\n"
            "      --generated-image local_workspace/results/your_run/output_test/00512_09216_target_generated.tif\n"
            "\n"
            "  python tools/evaluate_generation.py dataset\n"
            "      --target-dir local_workspace/datasets/your_run/dataset_test\n"
            "      --generated-dir local_workspace/results/your_run/output_test\n"
            "      --save-graphs\n\n"
            "Use 'python tools/evaluate_generation.py <command> --help' to see the options "
            "for a specific command."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
        add_help=True,
    )

    subparsers = parser.add_subparsers(dest="mode")
    add_single_subparser(subparsers)
    add_dataset_subparser(subparsers)
    return parser


def run_single(args: argparse.Namespace) -> None:
    """Esegue il flusso completo per la modalità single."""
    sample_id = extract_single_sample_id(args.target, args.generated)
    metrics, shape = evaluate_pair(args.target, args.generated)
    print_single_result(args.target, args.generated, metrics, shape)

    output_dir = resolve_output_dir(args.output_dir, args.generated)
    individual_cases_dir = output_dir / "individual_cases"
    individual_cases_dir.mkdir(parents=True, exist_ok=True)

    row = build_metric_row(sample_id, args.target, args.generated, shape, metrics)
    single_case_csv = individual_cases_dir / f"{sample_id}_evaluation.csv"
    write_single_case_csv(row, single_case_csv)

    print_section("Saved files")
    print_info("Single evaluation CSV", style(str(single_case_csv), "bold", "magenta"))


def run_dataset(args: argparse.Namespace) -> None:
    """Esegue il flusso completo per la modalità dataset."""
    target_files = collect_image_files(args.target_dir, "_target", "Target")
    generated_files = collect_image_files(args.generated_dir, "_target_generated", "Generated")

    output_dir = resolve_output_dir(args.output_dir, args.generated_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_sample_ids = sorted(set(target_files) | set(generated_files))
    per_image_rows: list[dict[str, object]] = []
    skipped_rows: list[dict[str, str]] = []

    for sample_id in all_sample_ids:
        target_path = target_files.get(sample_id)
        generated_path = generated_files.get(sample_id)

        if target_path is None:
            skipped_rows.append(
                {
                    "sample_id": sample_id,
                    "reason": "missing_target",
                    "target_path": "",
                    "generated_path": str(generated_path),
                }
            )
            continue

        if generated_path is None:
            skipped_rows.append(
                {
                    "sample_id": sample_id,
                    "reason": "missing_generated",
                    "target_path": str(target_path),
                    "generated_path": "",
                }
            )
            continue

        try:
            metrics, shape = evaluate_pair(target_path, generated_path)
        except Exception as exc:
            skipped_rows.append(
                {
                    "sample_id": sample_id,
                    "reason": str(exc),
                    "target_path": str(target_path),
                    "generated_path": str(generated_path),
                }
            )
            continue

        per_image_rows.append(
            build_metric_row(sample_id, target_path, generated_path, shape, metrics)
        )

    per_image_csv = output_dir / "per_image_metrics.csv"
    skipped_csv = output_dir / "skipped.csv"
    summary_csv = output_dir / "summary.csv"

    write_per_image_metrics_csv(per_image_rows, per_image_csv)
    write_skipped_csv(skipped_rows, skipped_csv)

    summary_rows: list[dict[str, object]] = []
    if per_image_rows:
        summary_rows = build_summary_rows(per_image_rows)

    write_summary_csv(
        summary_rows=summary_rows,
        output_path=summary_csv,
        num_targets_found=len(target_files),
        num_generated_found=len(generated_files),
        num_pairs_evaluated=len(per_image_rows),
        num_skipped=len(skipped_rows),
    )

    print_dataset_summary(
        target_files=target_files,
        generated_files=generated_files,
        per_image_rows=per_image_rows,
        skipped_rows=skipped_rows,
    )

    print_section("Saved files")
    print_info("Evaluation dir", style(str(output_dir), "bold", "magenta"))
    print_info("Per-image metrics", str(per_image_csv))
    print_info("Summary", str(summary_csv))
    print_info("Skipped samples", str(skipped_csv))

    if args.save_graphs and per_image_rows:
        plot_paths = save_dataset_plots(per_image_rows, output_dir)

        if not args.hide_graphs_path:
            for plot_path in plot_paths:
                print_info("Graph", str(plot_path))


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()