from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Literal, NotRequired, TypedDict

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from virtual_staining.utils.image_io import VALID_IMAGE_EXTENSIONS, open_rgb, to_float01
from virtual_staining.utils.metrics import DEFAULT_METRICS, is_higher_better_metric

METRIC_SELECTION_ORDER = list(DEFAULT_METRICS)
SELECTION_SUMMARY_FIELDNAMES = [
    "metric",
    "kind",
    "rank",
    "sample_id",
    "metric_value",
    "target_value",
    "abs_distance_from_target",
    "source_path",
    "target_path",
    "generated_path",
    "comparison_path",
    "error_histogram_path",
    "intensity_overlay_histogram_path",
    "target_vs_generated_scatter_by_channel_path",
]
RESIDUAL_HEATMAP_FIELDNAMES = [
    "rank",
    "sample_id",
    "metric",
    "metric_value",
    "target_path",
    "generated_path",
    "heatmap_path",
]

DiagnosticPathKey = Literal[
    "comparison_path",
    "error_histogram_path",
    "intensity_overlay_histogram_path",
    "target_vs_generated_scatter_by_channel_path",
]


class DiagnosticEntry(TypedDict):
    kind: str
    rank: NotRequired[int]
    sample_id: str
    metric_value: float
    comparison_path: Path
    error_histogram_path: Path
    intensity_overlay_histogram_path: Path
    target_vs_generated_scatter_by_channel_path: Path


def validate_same_size(*images: Image.Image) -> None:
    """Verifies that all images have the same size."""
    sizes = {image.size for image in images}

    if len(sizes) != 1:
        raise ValueError(f"All images must have the same size. Got: {sorted(sizes)}")


def compute_absolute_difference_map(
    generated_img: Image.Image, target_img: Image.Image
) -> np.ndarray:
    """Computes the per-pixel MAE map between target and generated."""
    generated_float = to_float01(generated_img)
    target_float = to_float01(target_img)
    return np.mean(np.abs(target_float - generated_float), axis=2)


def save_residual_heatmap(
    target_path: str | Path,
    generated_path: str | Path,
    save_path: str | Path,
) -> Path:
    """Save a standalone absolute-error heatmap for one target/generated pair."""
    generated_img = open_rgb(generated_path)
    target_img = open_rgb(target_path)
    validate_same_size(generated_img, target_img)
    diff_map = compute_absolute_difference_map(generated_img, target_img)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(diff_map, cmap="inferno", vmin=0.0, vmax=1.0)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("MAE map")
    ax.axis("off")

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return save_path


def _safe_filename_part(value: object) -> str:
    safe = "".join(
        char if char.isalnum() or char in {"-", "_", "."} else "_" for char in str(value)
    )
    return safe or "sample"


def _row_metric_value(row: dict[str, object], metric_name: str) -> float:
    value = row[metric_name]
    if isinstance(value, str | int | float):
        return float(value)
    raise TypeError(f"Metric '{metric_name}' must be a scalar value, got {type(value).__name__}.")


def select_worst_metric_rows(
    rows: list[dict[str, object]],
    metric_name: str,
    top_k: int,
) -> list[dict[str, object]]:
    """Select the worst finite metric rows for residual heatmap export."""
    if top_k <= 0:
        raise ValueError("top_k must be a positive integer.")

    higher_is_better = is_higher_better_metric(metric_name)
    ranked_rows: list[tuple[float, dict[str, object]]] = []

    for row in rows:
        metric_value = _row_metric_value(row, metric_name)
        if math.isfinite(metric_value):
            ranked_rows.append((metric_value, row))

    ranked_rows.sort(key=lambda item: item[0], reverse=not higher_is_better)
    return [row for _, row in ranked_rows[:top_k]]


def write_residual_heatmap_artifacts(
    rows: list[dict[str, object]],
    output_dir: str | Path,
    *,
    metric_name: str,
    top_k: int,
    filename: str = "residual_heatmaps.csv",
) -> Path:
    """Write residual heatmaps for the worst-ranked rows and record them in a CSV."""
    output_dir = Path(output_dir)
    heatmap_dir = output_dir / "residual_heatmaps"
    csv_path = output_dir / filename
    selected_rows = select_worst_metric_rows(rows, metric_name, top_k)
    artifact_rows: list[dict[str, object]] = []

    for rank, row in enumerate(selected_rows, start=1):
        sample_id = str(row["sample_id"])
        target_path = Path(str(row["target_path"]))
        generated_path = Path(str(row["generated_path"]))
        heatmap_path = heatmap_dir / (
            f"{rank:03d}_{_safe_filename_part(sample_id)}_{metric_name}_residual_heatmap.png"
        )
        saved_path = save_residual_heatmap(
            target_path=target_path,
            generated_path=generated_path,
            save_path=heatmap_path,
        )
        artifact_rows.append(
            {
                "rank": rank,
                "sample_id": sample_id,
                "metric": metric_name,
                "metric_value": _row_metric_value(row, metric_name),
                "target_path": str(target_path),
                "generated_path": str(generated_path),
                "heatmap_path": str(saved_path),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=RESIDUAL_HEATMAP_FIELDNAMES)
        writer.writeheader()
        writer.writerows(artifact_rows)

    return csv_path


def extract_generated_sample_id(path: str | Path) -> str:
    """Extracts the sample id from the generated file name."""
    stem = Path(path).stem
    suffix = "_target_generated"

    if not stem.endswith(suffix):
        raise ValueError(f"Generated file does not end with '{suffix}': {path}")

    return stem[: -len(suffix)]


def find_existing_image(base_dir: str | Path, sample_id: str, suffix: str) -> Path:
    """Searches for an existing image file by trying all supported extensions."""
    directory = Path(base_dir)

    for ext in sorted(VALID_IMAGE_EXTENSIONS):
        candidate = directory / f"{sample_id}{suffix}{ext}"
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        f"Could not find image for sample '{sample_id}' with suffix '{suffix}' inside {directory}"
    )


def infer_source_path_from_row(row: dict[str, str]) -> Path:
    """Tries to reconstruct the source path from a CSV row."""
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
        test_split_dir = generated_path.parents[1] / "splits" / "test"
        try:
            return find_existing_image(test_split_dir, sample_id, "_source")
        except FileNotFoundError:
            pass

    raise FileNotFoundError(f"Could not infer source path for sample '{sample_id}'.")


def select_representative_rows(
    metric_name: str,
    metric_summary: dict[str, float],
    per_image_rows: list[dict[str, str]],
) -> dict[str, dict[str, str]]:
    """Selects the best, median and worst samples for a metric."""
    if not per_image_rows:
        raise ValueError("No per-image rows available for representative selection.")

    def metric_value(row: dict[str, str]) -> float:
        return float(row[metric_name])

    higher_is_better = is_higher_better_metric(metric_name)
    best_row = (
        max(per_image_rows, key=metric_value)
        if higher_is_better
        else min(per_image_rows, key=metric_value)
    )
    worst_row = (
        min(per_image_rows, key=metric_value)
        if higher_is_better
        else max(per_image_rows, key=metric_value)
    )

    return {
        "best": best_row,
        "median": min(
            per_image_rows,
            key=lambda row: abs(metric_value(row) - metric_summary["median"]),
        ),
        "worst": worst_row,
    }


def select_ranked_representative_rows(
    metric_name: str,
    metric_summary: dict[str, float],
    per_image_rows: list[dict[str, str]],
    *,
    top_k: int,
    kinds: tuple[str, ...] = ("best", "median", "worst"),
) -> dict[str, list[dict[str, str]]]:
    """Select ranked best, median-band, and worst samples for a metric."""
    if top_k <= 0:
        raise ValueError("top_k must be a positive integer.")
    if not per_image_rows:
        raise ValueError("No per-image rows available for representative selection.")
    invalid_kinds = sorted(set(kinds) - {"best", "median", "worst"})
    if invalid_kinds:
        raise ValueError(f"Unsupported representative kinds: {', '.join(invalid_kinds)}")

    if top_k == 1:
        single_rows = select_representative_rows(metric_name, metric_summary, per_image_rows)
        return {kind: [single_rows[kind]] for kind in kinds}

    ranked_rows: dict[str, list[dict[str, str]]] = {}

    for kind in kinds:
        if kind in {"best", "worst"}:
            ranked_rows[kind] = find_representative_samples(
                per_image_rows,
                metric=metric_name,
                kind=kind,
                n=top_k,
            )
            continue

        if kind == "median":
            median = metric_summary["median"]
            ranked_rows[kind] = sorted(
                per_image_rows,
                key=lambda row: (
                    abs(float(row[metric_name]) - median),
                    float(row[metric_name]),
                    row["sample_id"],
                ),
            )[:top_k]
            continue

        raise ValueError(f"Unsupported representative kind: {kind}")

    return ranked_rows


def build_selection_summary_row(
    metric_name: str,
    kind: str,
    rank: int | None,
    sample_id: str,
    metric_value: float,
    target_value: float,
    source_path: Path,
    target_path: Path,
    generated_path: Path,
    comparison_path: Path,
) -> dict[str, object]:
    """Builds a standard row for selection CSVs."""
    return {
        "metric": metric_name,
        "kind": kind,
        "rank": rank,
        "sample_id": sample_id,
        "metric_value": metric_value,
        "target_value": target_value,
        "abs_distance_from_target": abs(metric_value - target_value),
        "source_path": str(source_path),
        "target_path": str(target_path),
        "generated_path": str(generated_path),
        "comparison_path": str(comparison_path),
    }


def write_metric_selection_summary(rows: list[dict[str, object]], save_path: str | Path) -> None:
    """Writes the CSV with the selected samples for each metric."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with save_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=SELECTION_SUMMARY_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def build_metric_case_artifacts(
    metric_name: str,
    kind: str,
    row: dict[str, str],
    metric_summary: dict[str, float],
    metric_dir: Path,
    rank: int | None = None,
    include_rank_in_filename: bool = False,
) -> tuple[dict[str, object], DiagnosticEntry]:
    """Build and save the artefacts for a representative case."""
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
    if include_rank_in_filename and rank is not None:
        filename_prefix = f"{kind}_{rank:03d}_{sample_id}"
    else:
        filename_prefix = f"{kind}_{sample_id}"
    comparison_path = metric_dir / f"{filename_prefix}_comparison.png"
    saved_path = save_comparison_panel(
        source_path=source_path,
        generated_path=generated_path,
        target_path=target_path,
        save_path=comparison_path,
        suptitle=None,
    )

    diagnostics_case_dir = metric_dir / "diagnostics" / filename_prefix
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
    if rank is not None:
        diagnostic_entry["rank"] = rank

    selection_row = build_selection_summary_row(
        metric_name=metric_name,
        kind=kind,
        rank=rank,
        sample_id=sample_id,
        metric_value=metric_value,
        target_value=target_value,
        source_path=source_path,
        target_path=target_path,
        generated_path=generated_path,
        comparison_path=saved_path,
    )
    selection_row["error_histogram_path"] = str(diagnostic_entry["error_histogram_path"])
    selection_row["intensity_overlay_histogram_path"] = str(
        diagnostic_entry["intensity_overlay_histogram_path"]
    )
    selection_row["target_vs_generated_scatter_by_channel_path"] = str(
        diagnostic_entry["target_vs_generated_scatter_by_channel_path"]
    )
    return selection_row, diagnostic_entry


def save_comparison_panel(
    source_path: str | Path,
    generated_path: str | Path,
    target_path: str | Path,
    save_path: str | Path,
    suptitle: str | None = None,
) -> Path:
    """Saves a panel with source, generated, target and MAE map."""
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

    for ax, image, title in zip(axes, images, titles, strict=True):
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
    """Saves the diagnostic plots for the individual sample."""
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

    for channel_index, (ax, label) in enumerate(zip(axes, channel_labels, strict=True)):
        target_channel = target[:, :, channel_index].ravel()
        generated_channel = generated[:, :, channel_index].ravel()
        n_points = min(20000, target_channel.size)
        sample_indices = rng.choice(target_channel.size, size=n_points, replace=False)
        ax.scatter(
            target_channel[sample_indices],
            generated_channel[sample_indices],
            s=4,
            alpha=0.25,
        )
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

    for channel_index, (ax, label) in enumerate(zip(axes, channel_labels, strict=True)):
        ax.hist(target[:, :, channel_index].ravel(), bins=50, alpha=0.5, label="Target")
        ax.hist(
            generated[:, :, channel_index].ravel(),
            bins=50,
            alpha=0.5,
            label="Generated",
        )
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


def save_stacked_image_panel(
    image_paths: list[str | Path],
    save_path: str | Path,
    row_titles: list[str] | None = None,
    suptitle: str | None = None,
) -> Path:
    """Saves a vertical panel composed of already-generated images."""
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

    for index, (ax, image, path) in enumerate(zip(axes, images, resolved_paths, strict=True)):
        ax.imshow(image)
        ax.set_title(row_titles[index] if row_titles is not None else path.stem)
        ax.axis("off")

    if suptitle:
        fig.suptitle(suptitle)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return save_path


def _diagnostic_row_title(entry: DiagnosticEntry, metric_name: str) -> str:
    rank_suffix = f" #{entry['rank']}" if "rank" in entry else ""
    return (
        f"{entry['kind'].upper()}{rank_suffix} | sample={entry['sample_id']} | "
        f"{metric_name}={entry['metric_value']:.6f}"
    )


def save_metric_diagnostics_summary(
    metric_name: str,
    metric_dir: str | Path,
    diagnostic_entries: list[DiagnosticEntry],
) -> list[Path]:
    """Saves aggregated panels for a metric across best, median and worst cases."""
    metric_dir = Path(metric_dir)
    output_specs: list[tuple[DiagnosticPathKey, str, str]] = [
        (
            "comparison_path",
            f"{metric_name}_comparisons_best_median_worst.png",
            f"{metric_name.upper()} - Comparison Panels (BEST / MEDIAN / WORST)",
        ),
        (
            "error_histogram_path",
            f"{metric_name}_error_histograms_best_median_worst.png",
            f"{metric_name.upper()} - Absolute Error Histograms (BEST / MEDIAN / WORST)",
        ),
        (
            "intensity_overlay_histogram_path",
            f"{metric_name}_intensity_overlay_histograms_best_median_worst.png",
            f"{metric_name.upper()} - Target vs Generated Intensity Histograms"
            " (BEST / MEDIAN / WORST)",
        ),
        (
            "target_vs_generated_scatter_by_channel_path",
            f"{metric_name}_target_vs_generated_scatters_by_channel_best_median_worst.png",
            f"{metric_name.upper()} - Target vs Generated Scatter by Channel"
            " (BEST / MEDIAN / WORST)",
        ),
    ]
    saved_paths: list[Path] = []

    for path_key, filename, suptitle in output_specs:
        image_paths: list[str | Path] = [entry[path_key] for entry in diagnostic_entries]
        row_titles = [_diagnostic_row_title(entry, metric_name) for entry in diagnostic_entries]
        saved_path = save_stacked_image_panel(
            image_paths=image_paths,
            save_path=metric_dir / filename,
            row_titles=row_titles,
            suptitle=suptitle,
        )
        saved_paths.append(saved_path)

    return saved_paths


def make_comparison_panel(
    source: str | Path,
    target: str | Path,
    generated: str | Path,
    output_path: str | Path,
) -> Path:
    return save_comparison_panel(
        source_path=source,
        generated_path=generated,
        target_path=target,
        save_path=output_path,
    )


def make_error_histogram(
    target: str | Path,
    generated: str | Path,
    output_path: str | Path,
) -> Path:
    generated_img = open_rgb(generated)
    target_img = open_rgb(target)
    validate_same_size(generated_img, target_img)
    target_arr = to_float01(target_img)
    generated_arr = to_float01(generated_img)
    absolute_error = np.mean(np.abs(target_arr - generated_arr), axis=2)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 4))
    plt.hist(absolute_error.ravel(), bins=50)
    plt.title("Absolute Error Histogram")
    plt.xlabel("Absolute error")
    plt.ylabel("Pixel count")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    return output_path


def make_intensity_overlay_histogram(
    target: str | Path,
    generated: str | Path,
    output_path: str | Path,
) -> Path:
    generated_img = open_rgb(generated)
    target_img = open_rgb(target)
    validate_same_size(generated_img, target_img)
    target_arr = to_float01(target_img)
    generated_arr = to_float01(generated_img)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    channel_labels = ["R", "G", "B"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)

    for channel_index, (ax, label) in enumerate(zip(axes, channel_labels, strict=True)):
        ax.hist(target_arr[:, :, channel_index].ravel(), bins=50, alpha=0.5, label="Target")
        ax.hist(
            generated_arr[:, :, channel_index].ravel(),
            bins=50,
            alpha=0.5,
            label="Generated",
        )
        ax.set_title(f"{label} channel")
        ax.set_xlabel("Intensity")
        ax.set_xlim(0, 1)

        if channel_index == 0:
            ax.set_ylabel("Pixel count")

        ax.legend()

    fig.suptitle("Target vs Generated Intensity Histograms")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def make_scatter_by_channel(
    target: str | Path,
    generated: str | Path,
    output_path: str | Path,
) -> Path:
    generated_img = open_rgb(generated)
    target_img = open_rgb(target)
    validate_same_size(generated_img, target_img)
    target_arr = to_float01(target_img)
    generated_arr = to_float01(generated_img)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    channel_labels = ["R", "G", "B"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)

    for channel_index, (ax, label) in enumerate(zip(axes, channel_labels, strict=True)):
        target_channel = target_arr[:, :, channel_index].ravel()
        generated_channel = generated_arr[:, :, channel_index].ravel()
        n_points = min(20000, target_channel.size)
        sample_indices = rng.choice(target_channel.size, size=n_points, replace=False)
        ax.scatter(
            target_channel[sample_indices],
            generated_channel[sample_indices],
            s=4,
            alpha=0.25,
        )
        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
        ax.set_title(f"{label} channel")
        ax.set_xlabel("Target intensity")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        if channel_index == 0:
            ax.set_ylabel("Generated intensity")

    fig.suptitle("Target vs Generated Intensity by Channel")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def find_representative_samples(
    rows: list[dict[str, str]],
    metric: str,
    kind: str,
    n: int,
) -> list[dict[str, str]]:
    sorted_rows = sorted(
        rows,
        key=lambda row: float(row[metric]),
        reverse=is_higher_better_metric(metric),
    )
    if kind == "best":
        return sorted_rows[:n]
    if kind == "worst":
        return list(reversed(sorted_rows[-n:])) if n > 0 else []
    if kind == "median":
        if n <= 0:
            return []
        middle = len(sorted_rows) // 2
        start = max(0, middle - n // 2)
        end = min(len(sorted_rows), start + n)
        return sorted_rows[start:end]
    raise ValueError(f"Unsupported representative kind: {kind}")


def save_selection_summary_csv(entries: list[dict[str, object]], output_path: str | Path) -> None:
    write_metric_selection_summary(entries, output_path)
