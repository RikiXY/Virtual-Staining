from __future__ import annotations

import argparse
import math
import statistics
from pathlib import Path
from typing import Any

from virtual_staining.config import (
    _TOP_LEVEL_KEYS,
    load_yaml_mapping,
    parse_bool_strict,
    reject_unknown_keys,
    section_with_shared_fields,
)
from virtual_staining.evaluation.evaluator import evaluate_pairs
from virtual_staining.evaluation.plotting import write_plots
from virtual_staining.evaluation.reports import (
    build_metric_row,
    write_single_case_csv,
    write_skipped_csv,
)
from virtual_staining.evaluation.summaries import (
    build_summary_rows as _build_summary_rows,
)
from virtual_staining.evaluation.summaries import (
    metric_value,
)
from virtual_staining.evaluation.summaries import (
    write_summary_csv as _write_summary_csv,
)
from virtual_staining.utils.cli import print_info, print_section, style
from virtual_staining.utils.metrics import (
    DEFAULT_METRICS,
    color_metric,
)

_EVALUATION_KEYS: frozenset[str] = frozenset(
    {
        # shared fields
        "dataset_root",
        "results_path",
        "run_name",
        # section-specific
        "save_graphs",
        "hide_graphs_path",
        "target_dir",
        "generated_dir",
        "output_dir",
    }
)

METRIC_NAMES = list(DEFAULT_METRICS)


def _optional_path(data: dict[str, Any], key: str, default: Path) -> Path:
    value = data.get(key)
    if value is None:
        return default
    return Path(value)


def apply_dataset_config(args: argparse.Namespace) -> argparse.Namespace:
    raw_data = load_yaml_mapping(args.config)
    if "evaluation" in raw_data:
        reject_unknown_keys(raw_data, _TOP_LEVEL_KEYS, "top level")
    data = section_with_shared_fields(
        raw_data, "evaluation", {"dataset_root", "results_path", "run_name"}
    )
    reject_unknown_keys(data, _EVALUATION_KEYS, "evaluation")

    dataset_root = Path(data["dataset_root"])
    run_root = Path(data["results_path"]) / data["run_name"]
    args.target_dir = str(_optional_path(data, "target_dir", dataset_root / "dataset_test"))
    args.generated_dir = str(_optional_path(data, "generated_dir", run_root / "output_test"))
    args.output_dir = str(_optional_path(data, "output_dir", run_root / "evaluation"))
    args.save_graphs = parse_bool_strict(data.get("save_graphs", True), "save_graphs")
    args.hide_graphs_path = parse_bool_strict(
        data.get("hide_graphs_path", getattr(args, "hide_graphs_path", False)),
        "hide_graphs_path",
    )
    return args


# ==========================
# Section dedicated to the parser
# ==========================


def add_single_subparser(subparsers: Any) -> None:
    """Adds the subcommand for evaluating a single pair."""
    single_parser = subparsers.add_parser(
        "single",
        help="Evaluate one target/generated image pair.",
        description=(
            "Compute MAE, MSE, RMSE, PSNR, SSIM and PCC for one target/generated pair. "
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
    """Adds the subcommand for evaluating an entire dataset."""
    dataset_parser = subparsers.add_parser(
        "dataset",
        help="Evaluate all matching target/generated pairs in two folders.",
        description=(
            "Compute MAE, MSE, RMSE, PSNR, SSIM and PCC for all matching pairs in a dataset. "
            "Supported image extensions: .tif, .tiff, .png."
        ),
    )
    dataset_parser.add_argument(
        "--config",
        type=str,
        default="config/runs/example.yaml",
        help="path to the run config YAML (default: config/runs/example.yaml)",
    )
    dataset_parser.add_argument(
        "--hide-graphs-path",
        action="store_true",
        help="Do not print the full list of saved graph paths.",
    )
    dataset_parser.set_defaults(func=run_dataset)


def build_parser() -> argparse.ArgumentParser:
    """Builds the main parser and registers the available subcommands."""
    parser = argparse.ArgumentParser(
        prog="python tools/evaluate_generation.py",
        description=(
            "Evaluate generated images against target images. "
            "Supported image extensions: .tif, .tiff, .png."
        ),
        epilog=(
            "Examples:\n"
            "  python tools/evaluate_generation.py single\n"
            "      --target-image local_workspace/datasets/your_run/dataset_test/00512_09216_target.tif\n"  # noqa: E501
            "      --generated-image local_workspace/results/your_run/output_test/00512_09216_target_generated.tif\n"  # noqa: E501
            "\n"
            "  python tools/evaluate_generation.py dataset\n"
            "      --config config/runs/example.yaml\n\n"
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


def extract_sample_id(path: str | Path, suffix: str, label: str = "File") -> str:
    """Extracts the sample id by removing the expected suffix from the filename."""
    name = Path(path).stem

    if not name.endswith(suffix):
        raise ValueError(f"{label} file does not end with '{suffix}': {path}")

    return name[: -len(suffix)]


def extract_single_sample_id(target_path: str | Path, generated_path: str | Path) -> str:
    """Checks that target and generated belong to the same sample."""
    target_id = extract_sample_id(target_path, "_target", "Target")
    generated_id = extract_sample_id(generated_path, "_target_generated", "Generated")

    if target_id != generated_id:
        raise ValueError(
            "Target and generated files refer to different sample ids. "
            f"Got '{target_id}' and '{generated_id}'."
        )

    return target_id


def collect_image_files(directory_path: str | Path, suffix: str, label: str) -> dict[str, Path]:
    """Collects valid files from a directory, indexed by sample id."""
    from virtual_staining.utils.image_io import VALID_IMAGE_EXTENSIONS

    directory = Path(directory_path)

    if not directory.is_dir():
        raise NotADirectoryError(f"{label} directory not found: {directory}")

    files: dict[str, Path] = {}

    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in VALID_IMAGE_EXTENSIONS:
            continue
        if not path.stem.endswith(suffix):
            continue

        sample_id = extract_sample_id(path, suffix, label)
        files[sample_id] = path

    return files


def infer_default_output_dir(generated_path: str | Path) -> Path:
    """Tries to derive results/NAME_RUN/evaluation from a path inside the run."""
    path = Path(generated_path).resolve()
    base = path.parent if path.is_file() else path
    parts = base.parts

    if "results" not in parts:
        raise ValueError(
            "Could not infer output directory from generated path. Expected the generated "
            "data to be inside a path like .../results/NAME_RUN/... Please provide "
            "--output-dir explicitly."
        )

    results_index = parts.index("results")

    if results_index + 1 >= len(parts):
        raise ValueError(
            "Could not infer NAME_RUN from generated path. Expected a path like "
            ".../results/NAME_RUN/... Please provide --output-dir explicitly."
        )

    run_dir = Path(*parts[: results_index + 2])

    if run_dir.name == "results":
        raise ValueError(
            "Could not infer NAME_RUN from generated path. Expected a path like "
            ".../results/NAME_RUN/... Please provide --output-dir explicitly."
        )

    if run_dir.parent.name != "results":
        raise ValueError(
            "Could not infer a valid run directory inside results/. Please provide "
            "--output-dir explicitly."
        )

    return run_dir / "evaluation"


def resolve_output_dir(output_dir: str | None, generated_path: str | Path) -> Path:
    """Uses the explicit output path if provided, otherwise tries to infer it."""
    if output_dir is not None:
        return Path(output_dir)
    return infer_default_output_dir(generated_path)


def build_summary_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Builds the aggregated rows for summary.csv."""
    return _build_summary_rows(rows)


def write_summary_csv(
    rows: list[dict[str, object]],
    output_dir: Path,
    num_targets_found: int,
    num_generated_found: int,
    num_pairs_evaluated: int,
    num_skipped: int,
) -> Path:
    """Writes the summary CSV with global counts and per-metric statistics."""
    return _write_summary_csv(
        rows,
        output_dir,
        num_targets_found=num_targets_found,
        num_generated_found=num_generated_found,
        num_pairs_evaluated=num_pairs_evaluated,
        num_skipped=num_skipped,
    )


# ====================================
# Section dedicated to the text report
# ====================================


def print_single_result(
    target_path: str | Path,
    generated_path: str | Path,
    metrics: dict[str, float],
    shape: tuple[int, int, int],
) -> None:
    """Prints the summary of a single evaluation to the terminal."""
    height, width, channels = shape

    print_section("Single-pair evaluation")
    print_info("Target", str(target_path))
    print_info("Generated", str(generated_path))
    print_info("Shape", f"{width}x{height}x{channels}")
    print()
    print_info("MAE", color_metric("mae", metrics["mae"]))
    print_info("MSE", color_metric("mse", metrics["mse"]))
    print_info("RMSE", color_metric("rmse", metrics["rmse"]))
    print_info("PSNR", color_metric("psnr", metrics["psnr"]))
    print_info("SSIM", color_metric("ssim", metrics["ssim"]))
    print_info("PCC gray", color_metric("pcc_gray", metrics["pcc_gray"]))
    print_info("PCC RGB mean", color_metric("pcc_rgb_mean", metrics["pcc_rgb_mean"]))


def print_dataset_summary(
    target_files: dict[str, Path],
    generated_files: dict[str, Path],
    per_image_rows: list[dict[str, object]],
    skipped_rows: list[dict[str, str]],
    output_dir: Path,
) -> None:
    """Prints a final summary of the dataset mode."""
    print_section("Dataset evaluation")
    print_info("Targets found", str(len(target_files)))
    print_info("Generated found", str(len(generated_files)))

    pairs_color = "green" if per_image_rows else "red"
    skipped_color = "green" if not skipped_rows else "yellow"
    print_info("Pairs evaluated", style(str(len(per_image_rows)), pairs_color))
    print_info("Skipped", style(str(len(skipped_rows)), skipped_color))

    if per_image_rows:
        print_section("Metric summary")
        for metric in METRIC_NAMES:
            values = [metric_value(row, metric) for row in per_image_rows]
            finite = [v for v in values if math.isfinite(v)]
            if not finite:
                continue
            print_info(
                f"{metric.upper()} mean", color_metric(metric, float(statistics.mean(finite)))
            )
            print_info(
                f"{metric.upper()} median",
                color_metric(metric, float(statistics.median(finite))),
            )

    print_section("Saved files")
    print_info("Evaluation dir", style(str(output_dir), "bold", "magenta"))


# =====================================
# Section dedicated to the main flow
# =====================================


def run_single(args: argparse.Namespace) -> None:
    """Runs the complete flow for the single mode."""
    from virtual_staining.evaluation.metrics import evaluate_pair

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
    """Runs the complete flow for the dataset mode."""
    args = apply_dataset_config(args)
    target_files = collect_image_files(args.target_dir, "_target", "Target")
    generated_files = collect_image_files(args.generated_dir, "_target_generated", "Generated")

    output_dir = resolve_output_dir(args.output_dir, args.generated_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_sample_ids = sorted(set(target_files) | set(generated_files))
    skipped_rows: list[dict[str, str]] = []
    paired_samples: list[tuple[Path, Path, str]] = []

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

        paired_samples.append((target_path, generated_path, sample_id))

    evaluation = evaluate_pairs(paired_samples, output_dir)
    per_image_rows = evaluation.rows
    skipped_rows.extend(evaluation.skipped_rows)

    per_image_csv = output_dir / "per_image_metrics.csv"
    skipped_csv = output_dir / "skipped.csv"
    summary_csv = write_summary_csv(
        per_image_rows,
        output_dir,
        num_targets_found=len(target_files),
        num_generated_found=len(generated_files),
        num_pairs_evaluated=len(per_image_rows),
        num_skipped=len(skipped_rows),
    )
    write_skipped_csv(skipped_rows, skipped_csv)

    print_section("Saved files")
    print_info("Per-image metrics", str(per_image_csv))
    print_info("Summary", str(summary_csv))
    print_info("Skipped samples", str(skipped_csv))

    if args.save_graphs and per_image_rows:
        plot_paths = write_plots(per_image_rows, output_dir)
        if not args.hide_graphs_path:
            for plot_path in plot_paths:
                print_info("Graph", str(plot_path))

    print_dataset_summary(target_files, generated_files, per_image_rows, skipped_rows, output_dir)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()
