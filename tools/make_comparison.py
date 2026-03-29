from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


METRIC_SELECTION_ORDER = ["mae", "rmse", "psnr", "ssim"]
VALID_IMAGE_EXTENSIONS = {".tif", ".tiff", ".png"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python tools/make_comparison.py",
        description=(
            "Create side-by-side comparison panels for paired histology images, "
            "or generate representative panels from evaluation CSV files."
        ),
    )

    parser.add_argument("--input-image", type=Path, help="Path to the input/source image.")
    parser.add_argument("--output-image", type=Path, help="Path to the generated/output image.")
    parser.add_argument("--target-image", type=Path, help="Path to the real target image.")
    parser.add_argument(
        "--save-path",
        type=Path,
        default=None,
        help=(
            "Path where the comparison panel will be saved. If omitted, the script "
            "tries to infer .../results/NAME_RUN/comparisons from --output-image."
        ),
    )
    parser.add_argument(
        "--with-diagnostics",
        action="store_true",
        help="Also save single-case diagnostic plots alongside the comparison panel.",
    )
    parser.add_argument(
        "--from-metrics",
        action="store_true",
        help=(
            "Generate representative comparison panels from evaluation CSV files "
            "inside a run directory. Requires --run-path."
        ),
    )
    parser.add_argument(
        "--run-path",
        type=Path,
        default=None,
        help=(
            "Path to a run directory like local_workspace/results/NAME_RUN. "
            "Used together with --from-metrics."
        ),
    )

    return parser


def open_rgb(path: str | Path) -> Image.Image:
    image_path = Path(path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not image_path.is_file():
        raise FileNotFoundError(f"Not a file: {image_path}")
    with Image.open(image_path) as img:
        return img.convert("RGB")


def validate_same_size(*images: Image.Image) -> None:
    sizes = {image.size for image in images}
    if len(sizes) != 1:
        raise ValueError(
            f"All images must have the same size to build a comparison panel. Got: {sorted(sizes)}"
        )


def to_float01(image: Image.Image) -> np.ndarray:
    return np.asarray(image, dtype=np.float32) / 255.0


def compute_absolute_difference_map(output_img: Image.Image, target_img: Image.Image) -> np.ndarray:
    output_float = to_float01(output_img)
    target_float = to_float01(target_img)
    return np.mean(np.abs(target_float - output_float), axis=2)


def extract_generated_sample_id(path: str | Path) -> str:
    stem = Path(path).stem
    suffix = "_target_generated"
    if not stem.endswith(suffix):
        raise ValueError(f"Generated file does not end with '{suffix}': {path}")
    return stem[: -len(suffix)]


def infer_run_dir_from_output_path(output_path: str | Path) -> Path:
    path = Path(output_path).resolve()
    base = path.parent if path.is_file() else path
    parts = base.parts
    if "results" not in parts:
        raise ValueError(
            "Could not infer run directory from output path. Expected a path like "
            ".../results/NAME_RUN/output_test/..."
        )
    results_index = parts.index("results")
    if results_index + 1 >= len(parts):
        raise ValueError(
            "Could not infer NAME_RUN from output path. Expected a path like "
            ".../results/NAME_RUN/output_test/..."
        )
    run_dir = Path(*parts[: results_index + 2])
    if run_dir.parent.name != "results":
        raise ValueError(
            "Could not infer a valid run directory inside results/. "
            "Please provide --save-path explicitly."
        )
    return run_dir


def infer_default_save_path(output_image: str | Path) -> Path:
    output_path = Path(output_image)
    sample_id = extract_generated_sample_id(output_path)
    run_dir = infer_run_dir_from_output_path(output_path)
    return run_dir / "comparisons" / f"{sample_id}_comparison.png"


def infer_diagnostics_dir(save_path: str | Path) -> Path:
    save_path = Path(save_path)
    return save_path.parent / "diagnostics"


def save_comparison_panel(
    input_path: str | Path,
    output_path: str | Path,
    target_path: str | Path,
    save_path: str | Path,
    suptitle: str | None = None,
) -> Path:
    input_img = open_rgb(input_path)
    output_img = open_rgb(output_path)
    target_img = open_rgb(target_path)

    validate_same_size(input_img, output_img, target_img)

    images: list[Any] = [input_img, output_img, target_img]
    titles = ["INPUT", "OUTPUT", "TARGET"]

    diff_map = compute_absolute_difference_map(output_img, target_img)
    images.append(diff_map)
    titles.append("PER-PIXEL MAE MAP")

    fig_width = 4 * len(images)
    fig, axes = plt.subplots(1, len(images), figsize=(fig_width, 4))
    if len(images) == 1:
        axes = [axes]

    for ax, image, title in zip(axes, images, titles):
        if isinstance(image, np.ndarray):
            im = ax.imshow(image, cmap="inferno", vmin=0.0, vmax=1.0)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            ax.imshow(image)
        ax.set_title(title)
        ax.axis("off")

    if suptitle:
        fig.suptitle(suptitle)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return save_path


def save_diagnostic_plots(
    input_path: str | Path,
    output_path: str | Path,
    target_path: str | Path,
    save_dir: str | Path,
) -> list[Path]:
    _ = open_rgb(input_path)
    output_img = open_rgb(output_path)
    target_img = open_rgb(target_path)
    validate_same_size(output_img, target_img)

    target = to_float01(target_img)
    generated = to_float01(output_img)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    sample_id = extract_generated_sample_id(output_path)
    saved_paths: list[Path] = []

    absolute_error = np.mean(np.abs(target - generated), axis=2)
    histogram_path = save_dir / f"{sample_id}_error_histogram.png"
    plt.figure(figsize=(6, 4))
    plt.hist(absolute_error.ravel(), bins=50)
    plt.title("Absolute Error Histogram")
    plt.xlabel("Absolute error")
    plt.ylabel("Pixel count")
    plt.tight_layout()
    plt.savefig(histogram_path, dpi=200, bbox_inches="tight")
    plt.close()
    saved_paths.append(histogram_path)

    scatter_path = save_dir / f"{sample_id}_target_vs_generated_scatter_by_channel.png"
    rng = np.random.default_rng(42)
    channel_labels = ["R", "G", "B"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
    for channel_index, (ax, label) in enumerate(zip(axes, channel_labels)):
        target_channel = target[:, :, channel_index].ravel()
        generated_channel = generated[:, :, channel_index].ravel()
        n_points = min(20000, target_channel.size)
        sample_indices = rng.choice(target_channel.size, size=n_points, replace=False)
        ax.scatter(target_channel[sample_indices], generated_channel[sample_indices], s=4, alpha=0.25)
        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
        ax.set_title(f"{label} channel")
        ax.set_xlabel("Target intensity")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        if channel_index == 0:
            ax.set_ylabel("Generated intensity")
    fig.suptitle("Target vs Generated Intensity by Channel")
    fig.tight_layout()
    fig.savefig(scatter_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(scatter_path)

    overlay_histogram_path = save_dir / f"{sample_id}_intensity_overlay_histogram.png"
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for channel_index, (ax, label) in enumerate(zip(axes, channel_labels)):
        ax.hist(target[:, :, channel_index].ravel(), bins=50, alpha=0.5, label="Target")
        ax.hist(generated[:, :, channel_index].ravel(), bins=50, alpha=0.5, label="Generated")
        ax.set_title(f"{label} channel")
        ax.set_xlabel("Intensity")
        ax.set_xlim(0, 1)
        if channel_index == 0:
            ax.set_ylabel("Pixel count")
        ax.legend()
    fig.suptitle("Target vs Generated Intensity Histograms")
    fig.tight_layout()
    fig.savefig(overlay_histogram_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(overlay_histogram_path)

    return saved_paths


def read_summary_csv(path: str | Path) -> dict[str, dict[str, float]]:
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
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Per-image metrics CSV not found: {csv_path}")
    with csv_path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def find_existing_image(base_dir: str | Path, sample_id: str, suffix: str) -> Path:
    directory = Path(base_dir)
    for ext in sorted(VALID_IMAGE_EXTENSIONS):
        candidate = directory / f"{sample_id}{suffix}{ext}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not find image for sample '{sample_id}' with suffix '{suffix}' inside {directory}"
    )


def infer_input_path_from_row(row: dict[str, str]) -> Path:
    sample_id = row["sample_id"]
    if row.get("input_path"):
        candidate = Path(row["input_path"])
        if candidate.is_file():
            return candidate
    if row.get("target_path"):
        target_dir = Path(row["target_path"]).parent
        try:
            return find_existing_image(target_dir, sample_id, "_source")
        except FileNotFoundError:
            pass
    if row.get("generated_path"):
        generated_path = Path(row["generated_path"])
        dataset_test_dir = generated_path.parents[1] / "dataset_test"
        try:
            return find_existing_image(dataset_test_dir, sample_id, "_source")
        except FileNotFoundError:
            pass
    raise FileNotFoundError(f"Could not infer input/source path for sample '{sample_id}'.")


def select_representative_rows(
    metric_name: str,
    metric_summary: dict[str, float],
    per_image_rows: list[dict[str, str]],
) -> dict[str, dict[str, str]]:
    if not per_image_rows:
        raise ValueError("No per-image rows available for representative selection.")
    def metric_value(row: dict[str, str]) -> float:
        return float(row[metric_name])
    return {
        "min": min(per_image_rows, key=metric_value),
        "max": max(per_image_rows, key=metric_value),
        "mean": min(per_image_rows, key=lambda row: abs(metric_value(row) - metric_summary["mean"])),
    }


def write_metric_selection_summary(rows: list[dict[str, object]], save_path: str | Path) -> None:
    fieldnames = [
        "metric",
        "kind",
        "sample_id",
        "metric_value",
        "target_value",
        "abs_distance_from_target",
        "input_path",
        "target_path",
        "generated_path",
        "comparison_path",
    ]
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_single_mode(args: argparse.Namespace) -> None:
    if args.input_image is None or args.output_image is None or args.target_image is None:
        raise SystemExit("Single mode requires --input-image, --output-image and --target-image.")

    save_path = args.save_path if args.save_path is not None else infer_default_save_path(args.output_image)
    saved_path = save_comparison_panel(
        input_path=args.input_image,
        output_path=args.output_image,
        target_path=args.target_image,
        save_path=save_path,
    )
    print(f"Saved comparison image to: {saved_path}")

    if args.with_diagnostics:
        diagnostics_dir = infer_diagnostics_dir(saved_path)
        diagnostic_paths = save_diagnostic_plots(
            input_path=args.input_image,
            output_path=args.output_image,
            target_path=args.target_image,
            save_dir=diagnostics_dir,
        )
        for diagnostic_path in diagnostic_paths:
            print(f"Saved diagnostic plot to:  {diagnostic_path}")


def run_from_metrics_mode(args: argparse.Namespace) -> None:
    if args.run_path is None:
        raise SystemExit("--from-metrics requires --run-path.")
    run_path = args.run_path.resolve()
    evaluation_dir = run_path / "evaluation"
    summary_csv = evaluation_dir / "summary.csv"
    per_image_csv = evaluation_dir / "per_image_metrics.csv"
    summary_rows = read_summary_csv(summary_csv)
    per_image_rows = read_per_image_metrics_csv(per_image_csv)
    metrics_dir = run_path / "comparisons" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    selection_summary_rows: list[dict[str, object]] = []
    available_metrics = [metric for metric in METRIC_SELECTION_ORDER if metric in summary_rows]
    if not available_metrics:
        raise ValueError(
            f"No supported metrics found in {summary_csv}. Expected one of: {', '.join(METRIC_SELECTION_ORDER)}"
        )
    for metric_name in available_metrics:
        metric_summary = summary_rows[metric_name]
        metric_dir = metrics_dir / metric_name
        metric_dir.mkdir(parents=True, exist_ok=True)
        representative_rows = select_representative_rows(metric_name, metric_summary, per_image_rows)
        metric_selection_rows: list[dict[str, object]] = []
        for kind, row in representative_rows.items():
            sample_id = row["sample_id"]
            metric_value = float(row[metric_name])
            target_value = float(metric_summary[kind if kind in {"min", "max"} else "mean"])
            input_path = infer_input_path_from_row(row)
            output_path = Path(row["generated_path"])
            target_path = Path(row["target_path"])
            comparison_path = metric_dir / f"{kind}_{sample_id}_comparison.png"
            saved_path = save_comparison_panel(
                input_path=input_path,
                output_path=output_path,
                target_path=target_path,
                save_path=comparison_path,
                suptitle=(
                    f"{metric_name.upper()} | {kind.upper()} | "
                    f"sample={sample_id} | value={metric_value:.6f}"
                ),
            )
            selection_row = {
                "metric": metric_name,
                "kind": kind,
                "sample_id": sample_id,
                "metric_value": metric_value,
                "target_value": target_value,
                "abs_distance_from_target": abs(metric_value - target_value),
                "input_path": str(input_path),
                "target_path": str(target_path),
                "generated_path": str(output_path),
                "comparison_path": str(saved_path),
            }
            selection_summary_rows.append(selection_row)
            metric_selection_rows.append(selection_row)
        write_metric_selection_summary(metric_selection_rows, metric_dir / "selection_summary.csv")
    write_metric_selection_summary(selection_summary_rows, metrics_dir / "metrics_selection_summary.csv")
    print(f"Saved metric-based comparisons to: {metrics_dir}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.from_metrics:
        run_from_metrics_mode(args)
    else:
        run_single_mode(args)


if __name__ == "__main__":
    main()
