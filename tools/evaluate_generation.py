from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path
from typing import Any

from utils.cli import ANSI, use_color, style, print_section, print_info
from utils.image_io import VALID_IMAGE_EXTENSIONS, load_rgb_image, to_float01
from utils.metrics import color_metric

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

try:
    from skimage.metrics import structural_similarity
except ImportError as exc:
    raise ImportError(
        "Missing dependency: scikit-image. Install it with:\n"
        "pip install scikit-image"
    ) from exc

METRIC_NAMES = ["mae", "rmse", "psnr", "ssim"]

METRIC_FIELDNAMES = [
    "sample_id",
    "target_path",
    "generated_path",
    "width",
    "height",
    "channels",
    "mae",
    "rmse",
    "psnr",
    "ssim",
]

PLOT_FIXED_RANGES = {
    "mae": (0.0, 1.0),
    "rmse": (0.0, 1.0),
    "ssim": (0.0, 1.0),
    # PSNR has no finite theoretical maximum. For visual comparison we use
    # a fixed practical window, so that histograms remain comparable.
    "psnr": (0.0, 60.0),
}

PLOT_FIXED_BINS = 30



# ==========================
# Section dedicated to the parser
# ==========================

def add_single_subparser(subparsers: Any) -> None:
    """Adds the subcommand for evaluating a single pair."""
    single_parser = subparsers.add_parser(
        "single",
        help="Evaluate one target/generated image pair.",
        description=(
            "Compute MAE, RMSE, PSNR and SSIM for one target/generated pair. "
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
            "Compute MAE, RMSE, PSNR and SSIM for all matching pairs in a dataset. "
            "Supported image extensions: .tif, .tiff, .png."
        ),
    )
    dataset_parser.add_argument(
        "--target-dir",
        dest="target_dir",
        type=str,
        required=True,
        help=("Directory containing target images with filename stem ending in '_target'. "
            "Supported extensions: .tif, .tiff, .png."
        ),
    )
    dataset_parser.add_argument(
        "--generated-dir",
        dest="generated_dir",
        type=str,
        required=True,
        help=("Directory containing generated images with filename stem ending in '_target_generated'. "
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


def validate_same_shape(target: np.ndarray, generated: np.ndarray) -> None:
    """Verifies that target and generated have exactly the same shape."""
    if target.shape != generated.shape:
        raise ValueError(
            "Target and generated images must have the same shape. "
            f"Got {target.shape} and {generated.shape}."
        )


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


# ==========================================
# Section dedicated to metric computation
# ==========================================

def compute_mae(target: np.ndarray, generated: np.ndarray) -> float:
    """Computes the Mean Absolute Error on normalised images."""
    return float(np.mean(np.abs(target - generated)))


def compute_rmse(target: np.ndarray, generated: np.ndarray) -> float:
    """Computes the Root Mean Squared Error on normalised images."""
    return float(np.sqrt(np.mean((target - generated) ** 2)))


def compute_psnr(target: np.ndarray, generated: np.ndarray) -> float:
    """Computes the PSNR assuming normalised images in the [0,1] range."""
    mse = float(np.mean((target - generated) ** 2))

    if mse == 0.0:
        return float("inf")

    return float(20.0 * np.log10(1.0 / np.sqrt(mse)))


def compute_ssim(target: np.ndarray, generated: np.ndarray) -> float:
    """Computes the SSIM on normalised RGB images."""
    try:
        return float(
            structural_similarity(
                target,
                generated,
                channel_axis=2,
                data_range=1.0,
            )
        )
    except TypeError:
        return float(
            structural_similarity(
                target,
                generated,
                multichannel=True,
                data_range=1.0,
            )
        )


def evaluate_pair(
    target_path: str | Path,
    generated_path: str | Path,
) -> tuple[dict[str, float], tuple[int, int, int]]:
    """Computes the four metrics for a target/generated pair."""
    target = load_rgb_image(target_path)
    generated = load_rgb_image(generated_path)

    validate_same_shape(target, generated)
    shape = target.shape

    target_float = to_float01(target)
    generated_float = to_float01(generated)

    metrics = {
        "mae": compute_mae(target_float, generated_float),
        "rmse": compute_rmse(target_float, generated_float),
        "psnr": compute_psnr(target_float, generated_float),
        "ssim": compute_ssim(target_float, generated_float),
    }

    return metrics, shape


def build_summary_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Builds the aggregated rows for summary.csv."""
    summary_rows: list[dict[str, object]] = []

    for metric in METRIC_NAMES:
        values = [float(row[metric]) for row in rows]
        summary_rows.append(
            {
                "metric": metric,
                "count": len(values),
                "mean": statistics.mean(values),
                "median": statistics.median(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                "min": min(values),
                "max": max(values),
            }
        )

    return summary_rows


# =======================================
# Section dedicated to CSV writing
# =======================================

def build_metric_row(
    sample_id: str,
    target_path: str | Path,
    generated_path: str | Path,
    shape: tuple[int, int, int],
    metrics: dict[str, float],
) -> dict[str, object]:
    """Builds a standard row for per_image_metrics.csv."""
    height, width, channels = shape
    return {
        "sample_id": sample_id,
        "target_path": str(target_path),
        "generated_path": str(generated_path),
        "width": width,
        "height": height,
        "channels": channels,
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "psnr": metrics["psnr"],
        "ssim": metrics["ssim"],
    }

def write_per_image_metrics_csv(rows: list[dict[str, object]], output_path: str | Path) -> None:
    """Writes the CSV with one row per evaluated pair."""
    with Path(output_path).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=METRIC_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def write_skipped_csv(rows: list[dict[str, str]], output_path: str | Path) -> None:
    """Writes the CSV of skipped samples with the corresponding reason."""
    fieldnames = ["sample_id", "reason", "target_path", "generated_path"]

    with Path(output_path).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_single_case_csv(row: dict[str, object], output_path: str | Path) -> None:
    """Writes the CSV produced by the single mode."""
    with Path(output_path).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=METRIC_FIELDNAMES)
        writer.writeheader()
        writer.writerow(row)


def write_summary_csv(
    summary_rows: list[dict[str, object]],
    output_path: str | Path,
    num_targets_found: int,
    num_generated_found: int,
    num_pairs_evaluated: int,
    num_skipped: int,
) -> None:
    """Writes the summary CSV with global counts and per-metric statistics."""
    fieldnames = ["metric", "count", "mean", "median", "std", "min", "max"]

    with Path(output_path).open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["num_targets_found", num_targets_found])
        writer.writerow(["num_generated_found", num_generated_found])
        writer.writerow(["num_pairs_evaluated", num_pairs_evaluated])
        writer.writerow(["num_skipped", num_skipped])
        writer.writerow([])
        writer.writerow(fieldnames)

        dict_writer = csv.DictWriter(file, fieldnames=fieldnames)
        dict_writer.writerows(summary_rows)


# ===========================
# Section dedicated to plots
# ===========================


def get_metric_plot_range(metric: str) -> tuple[float, float]:
    """Returns the fixed range used in plots for a metric."""
    if metric not in PLOT_FIXED_RANGES:
        raise ValueError(f"Unsupported metric for plotting: {metric}")
    return PLOT_FIXED_RANGES[metric]

def save_dataset_plots(rows: list[dict[str, object]], output_dir: str | Path) -> list[Path]:
    """Saves histograms with fixed axes and a final summary boxplot."""
    output_directory = Path(output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    for metric in METRIC_NAMES:
        values = [float(row[metric]) for row in rows]
        histogram_path = output_directory / f"{metric}_histogram.png"
        min_value, max_value = get_metric_plot_range(metric)
        bin_edges = np.linspace(min_value, max_value, PLOT_FIXED_BINS + 1)

        plt.figure(figsize=(6, 4))
        weights = np.ones(len(values), dtype=float) / len(values)
        plt.hist(values, bins=bin_edges, weights=weights)
        plt.title(f"{metric.upper()} Histogram")
        plt.xlabel(metric.upper())
        plt.ylabel("Share of samples")
        plt.xlim(min_value, max_value)
        plt.tight_layout()
        plt.savefig(histogram_path, dpi=200, bbox_inches="tight")
        plt.close()

        saved_paths.append(histogram_path)

    boxplot_path = output_directory / "metrics_boxplot.png"
    plt.figure(figsize=(8, 5))
    data = [[float(row[metric]) for row in rows] for metric in METRIC_NAMES]
    plt.boxplot(data, tick_labels=[metric.upper() for metric in METRIC_NAMES])
    plt.title("Metrics Boxplot")
    plt.ylabel("Value")
    plt.tight_layout()
    plt.savefig(boxplot_path, dpi=200, bbox_inches="tight")
    plt.close()

    saved_paths.append(boxplot_path)
    return saved_paths


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
    print_info("RMSE", color_metric("rmse", metrics["rmse"]))
    print_info("PSNR", color_metric("psnr", metrics["psnr"]))
    print_info("SSIM", color_metric("ssim", metrics["ssim"]))


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
            values = [float(row[metric]) for row in per_image_rows]
            print_info(f"{metric.upper()} mean", color_metric(metric, float(np.mean(values))))
            print_info(f"{metric.upper()} median", color_metric(metric, float(np.median(values))))

    print_section("Saved files")
    print_info("Evaluation dir", style(str(output_dir), "bold", "magenta"))




# =====================================
# Section dedicated to the main flow
# =====================================

def run_single(args: argparse.Namespace) -> None:
    """Runs the complete flow for the single mode."""
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

    print_section("Saved files")
    print_info("Per-image metrics", str(per_image_csv))
    print_info("Summary", str(summary_csv))
    print_info("Skipped samples", str(skipped_csv))

    if args.save_graphs and per_image_rows:
        plot_paths = save_dataset_plots(per_image_rows, output_dir)
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
