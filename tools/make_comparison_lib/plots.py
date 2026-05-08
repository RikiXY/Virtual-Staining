from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from make_comparison_lib.core import extract_generated_sample_id


def open_rgb(path: str | Path) -> Image.Image:
    """Apre un file immagine e lo restituisce come immagine RGB."""
    image_path = Path(path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if not image_path.is_file():
        raise FileNotFoundError(f"Not a file: {image_path}")

    with Image.open(image_path) as img:
        return img.convert("RGB")


def validate_same_size(*images: Image.Image) -> None:
    """Verifica che tutte le immagini abbiano la stessa dimensione."""
    sizes = {image.size for image in images}

    if len(sizes) != 1:
        raise ValueError(
            "All images must have the same size to build a comparison panel. "
            f"Got: {sorted(sizes)}"
        )


def to_float01(image: Image.Image) -> np.ndarray:
    """Converte un'immagine PIL in array float32 normalizzato in [0,1]."""
    return np.asarray(image, dtype=np.float32) / 255.0


def compute_absolute_difference_map(generated_img: Image.Image, target_img: Image.Image) -> np.ndarray:
    """Calcola la mappa MAE per pixel tra target e generated."""
    generated_float = to_float01(generated_img)
    target_float = to_float01(target_img)
    return np.mean(np.abs(target_float - generated_float), axis=2)


def save_comparison_panel(
    source_path: str | Path,
    generated_path: str | Path,
    target_path: str | Path,
    save_path: str | Path,
    suptitle: str | None = None,
) -> Path:
    """Salva un pannello con source, generated, target e mappa MAE."""
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
    """Salva i plot diagnostici del singolo sample."""
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


def save_stacked_image_panel(
    image_paths: list[str | Path],
    save_path: str | Path,
    row_titles: list[str] | None = None,
    suptitle: str | None = None,
) -> Path:
    """Salva un pannello verticale composto da immagini gia generate."""
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
    diagnostic_entries: list[dict[str, object]],
) -> list[Path]:
    """Salva i pannelli aggregati per una metrica sui casi best, median e worst."""
    metric_dir = Path(metric_dir)

    row_titles = [
        (
            f"{metric_name.upper()} | {str(entry['kind']).upper()} | "
            f"sample={entry['sample_id']} | "
            f"value={float(entry['metric_value']):.6f}"
        )
        for entry in diagnostic_entries
    ]

    output_specs = [
        (
            "comparison_path",
            f"{metric_name}_comparisons_best_median_worst.png",
        ),
        (
            "error_histogram_path",
            f"{metric_name}_error_histograms_best_median_worst.png",
        ),
        (
            "intensity_overlay_histogram_path",
            f"{metric_name}_intensity_overlay_histograms_best_median_worst.png",
        ),
        (
            "target_vs_generated_scatter_by_channel_path",
            f"{metric_name}_target_vs_generated_scatters_by_channel_best_median_worst.png",
        ),
    ]

    saved_paths: list[Path] = []

    for path_key, filename in output_specs:
        image_paths = [entry[path_key] for entry in diagnostic_entries]

        saved_path = save_stacked_image_panel(
            image_paths=image_paths,
            save_path=metric_dir / filename,
            row_titles=row_titles,
            suptitle=None,
        )

        saved_paths.append(saved_path)

    return saved_paths