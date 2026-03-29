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

    parser.add_argument("--source-image", type=Path, help="Path to the real source image.")
    parser.add_argument("--generated-image", type=Path, help="Path to the generated/generated image.")
    parser.add_argument("--target-image", type=Path, help="Path to the real target image.")
    parser.add_argument(
        "--save-path",
        type=Path,
        default=None,
        help=(
            "Path where the comparison panel will be saved. If omitted, the script "
            "tries to infer .../results/NAME_RUN/comparisons from --generated-image."
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


def compute_absolute_difference_map(generated_img: Image.Image, target_img: Image.Image) -> np.ndarray:
    generated_float = to_float01(generated_img)
    target_float = to_float01(target_img)
    return np.mean(np.abs(target_float - generated_float), axis=2)


def extract_generated_sample_id(path: str | Path) -> str:
    stem = Path(path).stem
    suffix = "_target_generated"
    if not stem.endswith(suffix):
        raise ValueError(f"Generated file does not end with '{suffix}': {path}")
    return stem[: -len(suffix)]


def infer_run_dir_from_generated_path(generated_path: str | Path) -> Path:
    path = Path(generated_path).resolve()
    base = path.parent if path.is_file() else path
    parts = base.parts
    if "results" not in parts:
        raise ValueError(
            "Could not infer run directory from generated path. Expected a path like "
            ".../results/NAME_RUN/generated_test/..."
        )
    results_index = parts.index("results")
    if results_index + 1 >= len(parts):
        raise ValueError(
            "Could not infer NAME_RUN from generated path. Expected a path like "
            ".../results/NAME_RUN/generated_test/..."
        )
    run_dir = Path(*parts[: results_index + 2])
    if run_dir.parent.name != "results":
        raise ValueError(
            "Could not infer a valid run directory inside results/. "
            "Please provide --save-path explicitly."
        )
    return run_dir


def infer_default_save_path(generated_image: str | Path) -> Path:
    generated_path = Path(generated_image)
    sample_id = extract_generated_sample_id(generated_path)
    run_dir = infer_run_dir_from_generated_path(generated_path)
    return run_dir / "comparisons" / f"{sample_id}_comparison.png"


def infer_diagnostics_dir(save_path: str | Path) -> Path:
    save_path = Path(save_path)
    return save_path.parent / "diagnostics"

def infer_case_diagnostics_dir(save_path: str | Path, generated_image: str | Path) -> Path:
    diagnostics_dir = infer_diagnostics_dir(save_path)
    sample_id = extract_generated_sample_id(generated_image)
    return diagnostics_dir / sample_id

def save_comparison_panel(
    source_path: str | Path,
    generated_path: str | Path,
    target_path: str | Path,
    save_path: str | Path,
    suptitle: str | None = None,
) -> Path:
    source_img = open_rgb(source_path)
    generated_img = open_rgb(generated_path)
    target_img = open_rgb(target_path)

    validate_same_size(source_img, generated_img, target_img)

    images: list[Any] = [source_img, generated_img, target_img]
    titles = ["source", "generated", "target"]

    diff_map = compute_absolute_difference_map(generated_img, target_img)
    images.append(diff_map)
    titles.append("MAE map")

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
    source_path: str | Path,
    generated_path: str | Path,
    target_path: str | Path,
    save_dir: str | Path,
) -> list[Path]:
    _ = open_rgb(source_path)
    generated_img = open_rgb(generated_path)
    target_img = open_rgb(target_path)
    validate_same_size(generated_img, target_img)

    target = to_float01(target_img)
    generated = to_float01(generated_img)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    sample_id = extract_generated_sample_id(generated_path)
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


def infer_source_path_from_row(row: dict[str, str]) -> Path:
    sample_id = row["sample_id"]
    if row.get("source_path"):
        candidate = Path(row["source_path"])
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
    raise FileNotFoundError(f"Could not infer source/source path for sample '{sample_id}'.")


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
        "source_path",
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

def save_stacked_image_panel(
    image_paths: list[str | Path],
    save_path: str | Path,
    row_titles: list[str] | None = None,
    suptitle: str | None = None,
) -> Path:
    if not image_paths:
        raise ValueError("No image paths provided for stacked panel.")

    resolved_paths = [Path(path) for path in image_paths]
    for path in resolved_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Diagnostic image not found: {path}")

    images = [np.asarray(Image.open(path).convert("RGB")) for path in resolved_paths]

    max_width = max(image.shape[1] for image in images)
    total_height = sum(image.shape[0] for image in images)

    dpi = 200
    fig_width = max_width / dpi
    extra_title_space = 0.8 if suptitle else 0.2
    fig_height = total_height / dpi + extra_title_space + 0.4 * len(images)

    fig, axes = plt.subplots(len(images), 1, figsize=(fig_width, fig_height))
    if len(images) == 1:
        axes = [axes]

    for index, (ax, image, path) in enumerate(zip(axes, images, resolved_paths)):
        ax.imshow(image)
        if row_titles is not None:
            ax.set_title(row_titles[index])
        else:
            ax.set_title(path.stem)
        ax.axis("off")

    if suptitle:
        fig.suptitle(suptitle)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return save_path


def build_metric_kind_row_title(metric_name: str, kind: str, sample_id: str, metric_value: float) -> str:
    return f"{metric_name.upper()} | {kind.upper()} | sample={sample_id} | value={metric_value:.6f}"


def save_metric_diagnostics_summary(
    metric_name: str,
    metric_dir: str | Path,
    diagnostic_entries: list[dict[str, object]],
) -> list[Path]:
    metric_dir = Path(metric_dir)

    output_specs = [
        (
            "comparison_path",
            f"{metric_name}_comparisons_max_mean_min.png",
            f"{metric_name.upper()} - Comparison Panels (MAX / MEAN / MIN)",
        ),
        (
            "error_histogram_path",
            f"{metric_name}_error_histograms_max_mean_min.png",
            f"{metric_name.upper()} - Absolute Error Histograms (MAX / MEAN / MIN)",
        ),
        (
            "intensity_overlay_histogram_path",
            f"{metric_name}_intensity_overlay_histograms_max_mean_min.png",
            f"{metric_name.upper()} - Target vs Generated Intensity Histograms (MAX / MEAN / MIN)",
        ),
        (
            "target_vs_generated_scatter_by_channel_path",
            f"{metric_name}_target_vs_generated_scatters_by_channel_max_mean_min.png",
            f"{metric_name.upper()} - Target vs Generated Scatter by Channel (MAX / MEAN / MIN)",
        ),
    ]

    saved_paths: list[Path] = []

    for path_key, filename, suptitle in output_specs:
        image_paths = [entry[path_key] for entry in diagnostic_entries]
        row_titles = [
            build_metric_kind_row_title(
                metric_name=metric_name,
                kind=entry["kind"],
                sample_id=entry["sample_id"],
                metric_value=entry["metric_value"],
            )
            for entry in diagnostic_entries
        ]
        saved_path = save_stacked_image_panel(
            image_paths=image_paths,
            save_path=metric_dir / filename,
            row_titles=row_titles,
            suptitle=suptitle,
        )
        saved_paths.append(saved_path)

    return saved_paths

def run_single_mode(args: argparse.Namespace) -> None:
    if args.source_image is None or args.generated_image is None or args.target_image is None:
        raise SystemExit("Single mode requires --source-image, --generated-image and --target-image.")

    save_path = args.save_path if args.save_path is not None else infer_default_save_path(args.generated_image)
    saved_path = save_comparison_panel(
        source_path=args.source_image,
        generated_path=args.generated_image,
        target_path=args.target_image,
        save_path=save_path,
    )
    print(f"Saved comparison image to: {saved_path}")

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
        metric_diagnostic_entries: list[dict[str, object]] = []
        for kind, row in representative_rows.items():
            sample_id = row["sample_id"]
            metric_value = float(row[metric_name])
            target_value = float(metric_summary[kind if kind in {"min", "max"} else "mean"])
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
            metric_diagnostic_entries.append(
                {
                    "kind": kind,
                    "sample_id": sample_id,
                    "metric_value": metric_value,
                    "comparison_path": saved_path,
                    "error_histogram_path": diagnostic_paths_by_name[f"{sample_id}_error_histogram.png"],
                    "intensity_overlay_histogram_path": diagnostic_paths_by_name[f"{sample_id}_intensity_overlay_histogram.png"],
                    "target_vs_generated_scatter_by_channel_path": diagnostic_paths_by_name[
                        f"{sample_id}_target_vs_generated_scatter_by_channel.png"
                    ],
                }
            )
            selection_row = {
                "metric": metric_name,
                "kind": kind,
                "sample_id": sample_id,
                "metric_value": metric_value,
                "target_value": target_value,
                "abs_distance_from_target": abs(metric_value - target_value),
                "source_path": str(source_path),
                "target_path": str(target_path),
                "generated_path": str(generated_path),
                "comparison_path": str(saved_path),
            }
            selection_summary_rows.append(selection_row)
            metric_selection_rows.append(selection_row)
        write_metric_selection_summary(metric_selection_rows, metric_dir / "selection_summary.csv")
        kind_order = {"max": 0, "mean": 1, "min": 2}
        metric_diagnostic_entries.sort(key=lambda entry: kind_order[entry["kind"]])

        aggregated_paths = save_metric_diagnostics_summary(
            metric_name=metric_name,
            metric_dir=metric_dir,
            diagnostic_entries=metric_diagnostic_entries,
        )
        for aggregated_path in aggregated_paths:
            print(f"Saved aggregated diagnostic panel to: {aggregated_path}")
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
