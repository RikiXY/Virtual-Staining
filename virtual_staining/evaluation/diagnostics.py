from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from virtual_staining.evaluation.selection import extract_generated_sample_id
from virtual_staining.utils.image_io import open_rgb, to_float01


def validate_same_size(*images: Image.Image) -> None:
    sizes = {image.size for image in images}
    if len(sizes) != 1:
        raise ValueError(
            f"All images must have the same size to build a comparison panel. Got: {sorted(sizes)}"
        )


def compute_absolute_difference_map(
    generated_img: Image.Image, target_img: Image.Image
) -> np.ndarray:
    generated = to_float01(generated_img)
    target = to_float01(target_img)
    return np.mean(np.abs(target - generated), axis=2)


def make_error_histogram(
    target: np.ndarray,
    generated: np.ndarray,
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    absolute_error = np.mean(np.abs(target - generated), axis=2)

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
    target: np.ndarray,
    generated: np.ndarray,
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    channel_labels = ["R", "G", "B"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)

    for channel_index, (ax, label) in enumerate(zip(axes, channel_labels, strict=True)):
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
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def make_scatter_by_channel(
    target: np.ndarray,
    generated: np.ndarray,
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
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
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_diagnostic_plots(
    generated_path: str | Path,
    target_path: str | Path,
    save_dir: str | Path,
) -> list[Path]:
    generated_img = open_rgb(generated_path)
    target_img = open_rgb(target_path)
    validate_same_size(generated_img, target_img)

    target = to_float01(target_img)
    generated = to_float01(generated_img)
    sample_id = extract_generated_sample_id(generated_path)
    save_dir = Path(save_dir)

    return [
        make_error_histogram(
            target,
            generated,
            save_dir / f"{sample_id}_error_histogram.png",
        ),
        make_scatter_by_channel(
            target,
            generated,
            save_dir / f"{sample_id}_target_vs_generated_scatter_by_channel.png",
        ),
        make_intensity_overlay_histogram(
            target,
            generated,
            save_dir / f"{sample_id}_intensity_overlay_histogram.png",
        ),
    ]
