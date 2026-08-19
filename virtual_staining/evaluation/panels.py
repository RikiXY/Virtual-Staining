from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, TypedDict

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from virtual_staining.evaluation.diagnostics import (
    compute_absolute_difference_map,
    save_diagnostic_plots,
    validate_same_size,
)
from virtual_staining.evaluation.selection import (
    build_selection_summary_row,
    infer_source_path_from_row,
)
from virtual_staining.metrics import is_higher_better_metric
from virtual_staining.utils.image_io import open_rgb

DiagnosticPathKey = Literal[
    "comparison_path",
    "error_histogram_path",
    "intensity_overlay_histogram_path",
    "target_vs_generated_scatter_by_channel_path",
]


class DiagnosticEntry(TypedDict):
    kind: str
    sample_id: str
    metric_value: float
    comparison_path: Path
    error_histogram_path: Path
    intensity_overlay_histogram_path: Path
    target_vs_generated_scatter_by_channel_path: Path


def build_metric_case_artifacts(
    metric_name: str,
    kind: str,
    row: dict[str, str],
    metric_summary: dict[str, float],
    metric_dir: Path,
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
        row_titles = [
            (
                f"{entry['kind'].upper()} | sample={entry['sample_id']} | "
                f"{metric_name}={entry['metric_value']:.6f}"
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
