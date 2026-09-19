from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from nicegui import ui
from PIL import Image

from virtual_staining.applications.api import RepresentativeSample, RunDescriptor, metric_quality
from virtual_staining.ui.theme import empty_state, metric_quality_style

logger = logging.getLogger(__name__)


@dataclass
class AsyncRequestGuard:
    """Reject results from superseded work or a disconnected browser page."""

    generation: int = 0

    def start(self) -> int:
        self.generation += 1
        return self.generation

    def invalidate(self) -> None:
        self.generation += 1

    def is_current(self, generation: int) -> bool:
        return generation == self.generation


def result_image(label: str, image: Image.Image | Path, note: str) -> None:
    with ui.column().classes("gap-2 min-w-0"):
        with ui.column().classes("vs-result-image-header w-full gap-0"):
            ui.label(label).classes("text-base font-semibold")
            ui.label(note).classes("text-sm text-slate-400")
        with ui.element("div").classes("vs-image-frame w-full aspect-square rounded-lg"):
            ui.image(image).props("fit=contain").classes("vs-image")


def result_placeholder(label: str, note: str) -> None:
    with ui.column().classes("gap-2 min-w-0"):
        ui.label(label).classes("text-base font-semibold")
        with (
            ui.element("div").classes(
                "vs-image-frame w-full aspect-square rounded-lg flex items-center justify-center"
            ),
            ui.column().classes("items-center text-center gap-2 px-4"),
        ):
            ui.icon("add_photo_alternate", size="md").classes("text-slate-300")
            ui.label(note).classes("text-sm text-slate-400")


def metric_card(metric: str, value: float | int | str, detail: str | None = None) -> None:
    numeric_value = float(value) if isinstance(value, int | float) else float("nan")
    quality = metric_quality(metric, numeric_value)
    quality_label, quality_class = metric_quality_style(quality)
    with ui.column().classes(f"vs-metric {quality_class} rounded-lg px-3 py-3 gap-0 min-w-0"):
        ui.label(metric.replace("_", " ").upper()).classes(
            "text-sm font-semibold tracking-wide text-slate-500"
        )
        ui.label(format_value(value)).classes("text-xl font-semibold text-slate-800 truncate")
        if detail or quality != "unknown":
            ui.label(detail or quality_label).classes("text-sm text-slate-500")


def browser_image_source(path: Path) -> Path | Image.Image | None:
    """Return a browser-compatible source, converting TIFF previews in memory."""
    if path.suffix.lower() not in {".tif", ".tiff"}:
        return path
    try:
        with Image.open(path) as image:
            return image.convert("RGB").copy()
    except (OSError, ValueError):
        logger.warning("Could not create browser preview for %s", path, exc_info=True)
        return None


def absolute_difference_preview(
    generated_path: Path | None,
    target_path: Path | None,
) -> Image.Image | None:
    if generated_path is None or target_path is None:
        return None
    try:
        with Image.open(generated_path) as generated_image, Image.open(target_path) as target_image:
            generated = np.asarray(generated_image.convert("RGB"), dtype=np.int16)
            target = np.asarray(target_image.convert("RGB"), dtype=np.int16)
        if generated.shape != target.shape:
            return None
        difference = np.mean(np.abs(target - generated), axis=2).astype(np.uint8)
        return Image.fromarray(difference, mode="L")
    except (OSError, ValueError):
        logger.warning("Could not create difference preview", exc_info=True)
        return None


def comparison_case_row(sample: RepresentativeSample) -> None:
    with ui.column().classes(
        "vs-case-row vs-subtle w-full self-center rounded-xl p-3 gap-3 min-w-0"
    ):
        with ui.row().classes("w-full items-center gap-3 flex-wrap"):
            ui.badge(sample.kind.upper()).props("outline color=teal-8")
            ui.label(sample.sample_id).classes("font-medium text-slate-700")
            ui.space()
            ui.label(f"{sample.metric.upper()}: {format_value(sample.value)}").classes(
                "font-mono text-sm text-slate-500"
            )
        with ui.row().classes(
            "vs-case-grid w-full grid grid-cols-2 lg:grid-cols-4 gap-3 items-stretch"
        ):
            previews: tuple[tuple[str, Path | Image.Image | None, str], ...] = (
                (
                    "Source",
                    browser_image_source(sample.source_path) if sample.source_path else None,
                    "Model input",
                ),
                (
                    "Generated",
                    browser_image_source(sample.generated_path) if sample.generated_path else None,
                    "Virtual stain",
                ),
                (
                    "Target",
                    browser_image_source(sample.target_path) if sample.target_path else None,
                    "Ground truth",
                ),
                (
                    "Absolute difference",
                    absolute_difference_preview(sample.generated_path, sample.target_path),
                    "Mean RGB error per pixel",
                ),
            )
            for label, image, detail in previews:
                if image is None:
                    result_placeholder(label, "Image unavailable")
                else:
                    result_image(label, image, detail)


def representative_card(sample: RepresentativeSample) -> None:
    with ui.column().classes("vs-subtle rounded-xl p-3 gap-2 min-w-0"):
        with ui.row().classes("w-full justify-between items-baseline gap-2"):
            ui.badge(sample.kind.upper()).props("outline color=teal-8")
            ui.label(format_value(sample.value)).classes("font-mono text-base")
        ui.label(sample.sample_id).classes("text-sm text-slate-500 truncate")
        if sample.comparison_path is not None:
            ui.image(sample.comparison_path).props("fit=contain").classes(
                "w-full rounded-lg border bg-white"
            )
            ui.label("Source · generated · target · MAE map").classes("text-sm text-slate-500")
        elif sample.generated_path is not None:
            preview = browser_image_source(sample.generated_path)
            if preview is None:
                empty_state(
                    "broken_image",
                    "Preview unavailable",
                    "The generated image could not be opened.",
                )
                return
            ui.image(preview).props("fit=contain").classes("w-full rounded-lg border bg-white")
            ui.label("Generated preview only").classes("text-sm font-medium text-amber-700")
            ui.label("Restore the source and target dataset to build the full panel.").classes(
                "text-sm text-slate-500"
            )
        else:
            empty_state("broken_image", "Image unavailable", "The metrics row has no usable path.")


def run_label(run_descriptor: RunDescriptor) -> str:
    count = f" · {run_descriptor.sample_count} samples" if run_descriptor.sample_count else ""
    return f"{run_descriptor.display_name}{count}"


def format_value(value: float | int | str) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "n/a"
        if math.isinf(value):
            return "∞"
        if abs(value) >= 1000 or (0 < abs(value) < 0.001):
            return f"{value:.3g}"
        return f"{value:.4f}"
    return str(value).replace("_", " ")
