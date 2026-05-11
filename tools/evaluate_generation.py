from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from tqdm import tqdm

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
    dataset_parser.add_argument(
        "--workers",
        type=int,
        default=get_default_workers(),
        help=(
            "Number of parallel workers used to evaluate image pairs. "
            "Use 1 to disable parallel evaluation."
        ),
    )
    dataset_parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the progress bar during dataset evaluation.",
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


def get_default_workers() -> int:
    """Restituisce un numero prudente di worker paralleli."""
    cpu_count = os.cpu_count() or 1

    if cpu_count <= 2:
        return 1

    return max(1, min(cpu_count - 1, 8))


def evaluate_pair_task(task: tuple[str, str, str]) -> dict[str, object]:
    """
    Valuta una singola coppia target/generated.

    La funzione è definita a livello globale per essere compatibile
    con ProcessPoolExecutor anche su Windows.
    """
    sample_id, target_path_str, generated_path_str = task

    target_path = Path(target_path_str)
    generated_path = Path(generated_path_str)

    try:
        metrics, shape = evaluate_pair(target_path, generated_path)

        return {
            "ok": True,
            "row": build_metric_row(
                sample_id=sample_id,
                target_path=target_path,
                generated_path=generated_path,
                shape=shape,
                metrics=metrics,
            ),
        }

    except Exception as exc:
        return {
            "ok": False,
            "row": {
                "sample_id": sample_id,
                "reason": str(exc),
                "target_path": str(target_path),
                "generated_path": str(generated_path),
            },
        }


def run_dataset(args: argparse.Namespace) -> None:
    """Esegue il flusso completo per la modalità dataset."""
    print_section("Dataset evaluation")
    print_info("Status", "Preparing image pairs...")

    target_files = collect_image_files(args.target_dir, "_target", "Target")
    generated_files = collect_image_files(
        args.generated_dir,
        "_target_generated",
        "Generated",
    )

    output_dir = resolve_output_dir(args.output_dir, args.generated_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_sample_ids = sorted(set(target_files) | set(generated_files))

    per_image_rows: list[dict[str, object]] = []
    skipped_rows: list[dict[str, str]] = []
    evaluation_tasks: list[tuple[str, str, str]] = []

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

        evaluation_tasks.append(
            (
                sample_id,
                str(target_path),
                str(generated_path),
            )
        )

    requested_workers = max(1, int(args.workers))
    max_workers = min(requested_workers, len(evaluation_tasks)) if evaluation_tasks else 1

    print_info("Targets found", str(len(target_files)))
    print_info("Generated found", str(len(generated_files)))
    print_info("Pairs to evaluate", str(len(evaluation_tasks)))
    print_info("Initial skipped", str(len(skipped_rows)))
    print_info(
        "Status",
        f"Evaluating image pairs using {max_workers} worker(s). This may take a while...",
    )

    if evaluation_tasks:
        if max_workers == 1:
            iterator = evaluation_tasks

            if not args.no_progress:
                iterator = tqdm(
                    evaluation_tasks,
                    desc="Evaluating pairs",
                    unit="pair",
                )

            for task in iterator:
                result = evaluate_pair_task(task)
                if result["ok"]:
                    per_image_rows.append(result["row"])
                else:
                    skipped_rows.append(result["row"])

        else:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(evaluate_pair_task, task)
                    for task in evaluation_tasks
                ]

                iterator = as_completed(futures)

                if not args.no_progress:
                    iterator = tqdm(
                        iterator,
                        total=len(futures),
                        desc="Evaluating pairs",
                        unit="pair",
                    )

                for future in iterator:
                    try:
                        result = future.result()
                    except Exception as exc:
                        skipped_rows.append(
                            {
                                "sample_id": "unknown",
                                "reason": f"worker_error: {exc}",
                                "target_path": "",
                                "generated_path": "",
                            }
                        )
                        continue

                    if result["ok"]:
                        per_image_rows.append(result["row"])
                    else:
                        skipped_rows.append(result["row"])

    per_image_rows.sort(key=lambda row: str(row["sample_id"]))
    skipped_rows.sort(key=lambda row: str(row["sample_id"]))

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
        print_info("Status", "Saving aggregate plots...")

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