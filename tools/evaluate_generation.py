from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

try:
    from skimage.metrics import structural_similarity
except ImportError as exc:
    raise ImportError(
        "Missing dependency: scikit-image. Install it with:\n"
        "pip install scikit-image"
    ) from exc


def add_single_subparser(subparsers: Any) -> None:
    single_parser = subparsers.add_parser(
        "single",
        help="Evaluate one target/generated image pair.",
        description="Compute MAE, RMSE, PSNR and SSIM for one target/generated pair.",
    )
    single_parser.add_argument(
        "target",
        type=str,
        help="Path to the target image.",
    )
    single_parser.add_argument(
        "generated",
        type=str,
        help="Path to the generated image.",
    )
    single_parser.set_defaults(func=run_single)


def add_dataset_subparser(subparsers: Any) -> None:
    dataset_parser = subparsers.add_parser(
        "dataset",
        help="Evaluate all matching target/generated pairs in two folders.",
        description="Compute MAE, RMSE, PSNR and SSIM for all matching pairs in a dataset.",
    )
    dataset_parser.add_argument(
        "target_dir",
        type=str,
        help="Directory containing target images.",
    )
    dataset_parser.add_argument(
        "generated_dir",
        type=str,
        help="Directory containing generated images.",
    )
    dataset_parser.add_argument(
        "output_dir",
        type=str,
        help="Directory where CSV results will be saved.",
    )
    dataset_parser.set_defaults(func=run_dataset)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python tools/evaluate_generation.py",
        description="Evaluate generated images against target images.",
        add_help=True,
    )

    subparsers = parser.add_subparsers(dest="mode")
    add_single_subparser(subparsers)
    add_dataset_subparser(subparsers)

    return parser


def load_rgb_image(path: str | Path) -> np.ndarray:
    image_path = Path(path)

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as exc:
        raise RuntimeError(f"Could not open image: {image_path}") from exc

    return np.array(image)


def validate_same_shape(target: np.ndarray, generated: np.ndarray) -> None:
    if target.shape != generated.shape:
        raise ValueError(
            "Target and generated images must have the same shape. "
            f"Got {target.shape} and {generated.shape}."
        )


def to_float01(image: np.ndarray) -> np.ndarray:
    return image.astype(np.float32) / 255.0


def compute_mae(target: np.ndarray, generated: np.ndarray) -> float:
    return float(np.mean(np.abs(target - generated)))


def compute_rmse(target: np.ndarray, generated: np.ndarray) -> float:
    return float(np.sqrt(np.mean((target - generated) ** 2)))


def compute_psnr(target: np.ndarray, generated: np.ndarray) -> float:
    mse = float(np.mean((target - generated) ** 2))

    if mse == 0.0:
        return float("inf")

    return float(20.0 * np.log10(1.0 / np.sqrt(mse)))


def compute_ssim(target: np.ndarray, generated: np.ndarray) -> float:
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
    target = load_rgb_image(target_path)
    generated = load_rgb_image(generated_path)

    validate_same_shape(target, generated)
    shape = target.shape

    target = to_float01(target)
    generated = to_float01(generated)

    metrics = {
        "mae": compute_mae(target, generated),
        "rmse": compute_rmse(target, generated),
        "psnr": compute_psnr(target, generated),
        "ssim": compute_ssim(target, generated),
    }

    return metrics, shape


def print_single_result(
    target_path: str | Path,
    generated_path: str | Path,
    metrics: dict[str, float],
    shape: tuple[int, int, int],
) -> None:
    height, width, channels = shape

    print(f"Target:     {target_path}")
    print(f"Generated:  {generated_path}")
    print(f"Shape:      {width}x{height}x{channels}")
    print()
    print(f"MAE:   {metrics['mae']:.6f}")
    print(f"RMSE:  {metrics['rmse']:.6f}")
    print(f"PSNR:  {metrics['psnr']:.4f}")
    print(f"SSIM:  {metrics['ssim']:.4f}")


def extract_target_sample_id(path: str | Path) -> str:
    name = Path(path).stem
    suffix = "_target"

    if not name.endswith(suffix):
        raise ValueError(f"Target file does not end with '{suffix}': {path}")

    return name[: -len(suffix)]


def extract_generated_sample_id(path: str | Path) -> str:
    name = Path(path).stem
    suffix = "_target_generated"

    if not name.endswith(suffix):
        raise ValueError(f"Generated file does not end with '{suffix}': {path}")

    return name[: -len(suffix)]


def collect_target_files(target_dir: str | Path) -> dict[str, Path]:
    directory = Path(target_dir)

    if not directory.is_dir():
        raise NotADirectoryError(f"Target directory not found: {directory}")

    target_files: dict[str, Path] = {}
    for path in sorted(directory.glob("*_target.tif")):
        sample_id = extract_target_sample_id(path)
        target_files[sample_id] = path

    return target_files


def collect_generated_files(generated_dir: str | Path) -> dict[str, Path]:
    directory = Path(generated_dir)

    if not directory.is_dir():
        raise NotADirectoryError(f"Generated directory not found: {directory}")

    generated_files: dict[str, Path] = {}
    for path in sorted(directory.glob("*_target_generated.tif")):
        sample_id = extract_generated_sample_id(path)
        generated_files[sample_id] = path

    return generated_files


def write_per_image_metrics_csv(rows: list[dict[str, object]], output_path: str | Path) -> None:
    fieldnames = [
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

    with Path(output_path).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_skipped_csv(rows: list[dict[str, str]], output_path: str | Path) -> None:
    fieldnames = ["sample_id", "reason", "target_path", "generated_path"]

    with Path(output_path).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_summary_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    metrics = ["mae", "rmse", "psnr", "ssim"]
    summary_rows: list[dict[str, object]] = []

    for metric in metrics:
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


def write_summary_csv(
    summary_rows: list[dict[str, object]],
    output_path: str | Path,
    num_targets_found: int,
    num_generated_found: int,
    num_pairs_evaluated: int,
    num_skipped: int,
) -> None:
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


def run_single(args: argparse.Namespace) -> None:
    metrics, shape = evaluate_pair(args.target, args.generated)
    print_single_result(args.target, args.generated, metrics, shape)


def run_dataset(args: argparse.Namespace) -> None:
    target_files = collect_target_files(args.target_dir)
    generated_files = collect_generated_files(args.generated_dir)

    output_dir = Path(args.output_dir)
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

        height, width, channels = shape

        per_image_rows.append(
            {
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
        )

    per_image_csv = output_dir / "per_image_metrics.csv"
    skipped_csv = output_dir / "skipped.csv"
    summary_csv = output_dir / "summary.csv"

    write_per_image_metrics_csv(per_image_rows, per_image_csv)
    write_skipped_csv(skipped_rows, skipped_csv)

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
    else:
        write_summary_csv(
            summary_rows=[],
            output_path=summary_csv,
            num_targets_found=len(target_files),
            num_generated_found=len(generated_files),
            num_pairs_evaluated=0,
            num_skipped=len(skipped_rows),
        )

    print(f"Saved per-image metrics to: {per_image_csv}")
    print(f"Saved summary to:           {summary_csv}")
    print(f"Saved skipped samples to:   {skipped_csv}")
    print()
    print(f"Targets found:   {len(target_files)}")
    print(f"Generated found: {len(generated_files)}")
    print(f"Pairs evaluated: {len(per_image_rows)}")
    print(f"Skipped:         {len(skipped_rows)}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()