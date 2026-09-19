from __future__ import annotations

from collections.abc import Callable

from nicegui import ui
from nicegui.element import Element


def build_tutorial() -> Callable[[], None]:
    """Build the reusable, user-opened onboarding dialog and return its opener."""
    slide_index = 0
    slides: list[Element] = []
    dots: list[Element] = []

    with (
        ui.dialog() as dialog,
        ui.card().classes(
            "vs-tutorial w-full h-full sm:h-auto sm:min-h-[620px] "
            "rounded-none sm:rounded-2xl p-0 gap-0"
        ),
    ):
        with ui.row().classes("w-full items-center justify-between px-5 sm:px-8 py-4 border-b"):
            with ui.row().classes("items-center gap-2"):
                ui.icon("school", size="sm").classes("text-teal-700")
                ui.label("Virtual Staining tutorial").classes("font-semibold text-xl")
            ui.button(icon="close", on_click=dialog.close).props(
                "flat round color=blue-grey-8 aria-label=Close"
            )

        slide_host = ui.column().classes("w-full flex-1 px-5 sm:px-10 py-7 gap-0 overflow-y-auto")
        with slide_host:
            slides.append(
                _slide(
                    "biotech",
                    "One application, two workflows",
                    "Inference is the fast path for generating a virtual stain. Experiments adds "
                    "ground truth, metrics, run-level evaluation, and statistical comparison.",
                    lambda: _flow(
                        ("Image", "image"),
                        ("Inference", "auto_awesome"),
                        ("Virtual stain", "texture"),
                    ),
                )
            )
            slides.append(
                _slide(
                    "auto_awesome",
                    "Generate a virtual stain",
                    "Choose a validated checkpoint, upload an exact-size RGB source patch, then "
                    "generate. Inspect source and output side by side and save the PNG with its "
                    "portable provenance record.",
                    lambda: _numbered_steps(
                        "Choose model",
                        "Upload & validate",
                        "Generate",
                        "Inspect & save",
                    ),
                )
            )
            slides.append(
                _slide(
                    "science",
                    "Experiments uses a real target",
                    "A target is the measured stained image used as ground truth. It is never fed "
                    "to the model; it is compared with the generated image after inference.",
                    lambda: _flow(
                        ("Source", "image"),
                        ("Model", "memory"),
                        ("Generated ↔ Target", "compare"),
                    ),
                )
            )
            slides.append(
                _slide(
                    "calculate",
                    "Test one paired sample",
                    "Generate with only a source, then optionally add its target before or after "
                    "generation. Evaluation reuses the generated result to compute SSIM, PSNR, "
                    "errors, and correlations.",
                    lambda: _pipeline_diagram(),
                )
            )
            slides.append(
                _slide(
                    "analytics",
                    "Evaluate a complete run",
                    "Use this after inference has produced predictions for a complete test set. "
                    "Load a run that already has evaluation CSVs, or provide its YAML config and "
                    "let the app execute evaluation for you.",
                    lambda: _evaluation_guide(),
                )
            )
            slides.append(
                _slide(
                    "compare_arrows",
                    "Compare two runs",
                    "Choose a metric and use paired mode when both runs contain the same sample "
                    "IDs. Use unpaired mode for independent test sets. Positive signed improvement "
                    "favors run B after accounting for metric direction.",
                    lambda: _comparison_diagram(),
                )
            )
            slides.append(
                _slide(
                    "fingerprint",
                    "Keep results reproducible",
                    "Saved inference includes checkpoint and input metadata. Experiment outputs "
                    "stay inside the configured output/results locations, while run metadata and "
                    "config "
                    "snapshots document how results were produced.",
                    lambda: _provenance_diagram(),
                )
            )

        with ui.element("div").classes("vs-tutorial-footer w-full px-5 sm:px-8 py-4 border-t"):
            with ui.row().classes("justify-self-start min-w-24"):
                previous = ui.button("Previous", icon="arrow_back").props("flat color=blue-grey-8")
            with ui.row().classes("justify-self-center items-center gap-2"):
                for _ in slides:
                    dots.append(ui.icon("circle", size="8px"))
            with ui.row().classes("justify-self-end min-w-24 justify-end"):
                next_button = ui.button("Next", icon="arrow_forward").props(
                    "unelevated color=primary icon-right"
                )

    def show_slide(index: int) -> None:
        nonlocal slide_index
        slide_index = max(0, min(index, len(slides) - 1))
        for position, slide in enumerate(slides):
            slide.set_visibility(position == slide_index)
        for position, dot in enumerate(dots):
            dot.classes(replace="text-teal-600" if position == slide_index else "text-slate-300")
        previous.set_visibility(slide_index > 0)
        next_button.set_text("Finish" if slide_index == len(slides) - 1 else "Next")
        next_button.set_icon("check" if slide_index == len(slides) - 1 else "arrow_forward")

    def go_next() -> None:
        if slide_index == len(slides) - 1:
            dialog.close()
        else:
            show_slide(slide_index + 1)

    previous.on_click(lambda: show_slide(slide_index - 1))
    next_button.on_click(go_next)
    show_slide(0)

    def open_tutorial() -> None:
        show_slide(0)
        dialog.open()

    return open_tutorial


def _slide(
    icon: str,
    title: str,
    body: str,
    visual: Callable[[], None],
) -> Element:
    with ui.column().classes("w-full h-full items-center text-center gap-5") as slide:
        with ui.element("div").classes(
            "w-16 h-16 rounded-2xl bg-teal-50 flex items-center justify-center"
        ):
            ui.icon(icon, size="lg").classes("text-teal-700")
        ui.label(title).classes("text-3xl sm:text-4xl font-semibold tracking-tight")
        ui.label(body).classes("text-lg text-slate-600 leading-relaxed max-w-2xl")
        with ui.element("div").classes("vs-subtle w-full max-w-3xl rounded-xl p-5 sm:p-7 mt-2"):
            visual()
    return slide


def _flow(*nodes: tuple[str, str]) -> None:
    with ui.row().classes("w-full items-center justify-center gap-2 sm:gap-5 flex-wrap"):
        for index, (label, icon) in enumerate(nodes):
            if index:
                ui.icon("arrow_forward", size="sm").classes("text-slate-300")
            with ui.column().classes("items-center gap-2 min-w-24"):
                ui.icon(icon, size="md").classes("text-sky-700")
                ui.label(label).classes("text-base font-medium")


def _numbered_steps(*labels: str) -> None:
    with ui.row().classes("w-full justify-center gap-3 flex-wrap"):
        for index, label in enumerate(labels, 1):
            with ui.row().classes("items-center gap-2 bg-white border rounded-lg px-3 py-2"):
                ui.badge(str(index)).props("rounded color=teal-7").classes("vs-step-number")
                ui.label(label).classes("text-base font-medium")


def _pipeline_diagram() -> None:
    _flow(
        ("Source", "image"),
        ("Generated", "texture"),
        ("Target", "verified"),
        ("Metrics + error map", "calculate"),
    )


def _evaluation_guide() -> None:
    with ui.column().classes("vs-evaluation-guide w-full gap-4 text-left"):
        ui.label("Choose one way to begin").classes(
            "text-sm font-semibold uppercase tracking-wide text-slate-500"
        )
        with ui.row().classes("w-full grid grid-cols-1 sm:grid-cols-2 gap-3"):
            _evaluation_option(
                "folder_open",
                "Existing run",
                "Select a run discovered under --results-dir whose evaluation folder contains "
                "summary.csv and per_image_metrics.csv, then choose Load evaluation.",
            )
            _evaluation_option(
                "description",
                "Run from config",
                "Switch the mode, enter the run YAML path, then choose Run evaluation. The app "
                "executes the evaluation stage before loading it.",
            )
        with ui.row().classes("w-full items-center justify-center gap-3 text-teal-700"):
            ui.separator().classes("flex-1")
            ui.icon("arrow_downward", size="xs")
            ui.label("Then inspect").classes("text-sm font-semibold")
            ui.separator().classes("flex-1")
        with ui.row().classes("w-full grid grid-cols-1 sm:grid-cols-3 gap-2"):
            for title, detail in (
                ("Aggregate metrics", "Mean, median, spread, and range"),
                ("Plots", "Metric distributions across the test set"),
                ("Representative cases", "Best, median, and worst; keep panels enabled"),
            ):
                with ui.column().classes("bg-white border rounded-lg px-3 py-2 gap-0"):
                    ui.label(title).classes("text-sm font-semibold text-slate-700")
                    ui.label(detail).classes("text-sm leading-snug text-slate-500")


def _evaluation_option(icon: str, title: str, detail: str) -> None:
    with ui.row().classes("bg-white border rounded-lg p-3 gap-3 items-start flex-nowrap"):
        ui.icon(icon, size="sm").classes("text-sky-700 shrink-0")
        with ui.column().classes("gap-1 min-w-0"):
            ui.label(title).classes("text-base font-semibold text-slate-800")
            ui.label(detail).classes("text-sm leading-relaxed text-slate-500")


def _comparison_diagram() -> None:
    _flow(
        ("Run A", "folder"),
        ("Metric + mode", "tune"),
        ("Run B", "folder"),
        ("Statistics + plots", "insights"),
    )


def _provenance_diagram() -> None:
    _flow(
        ("Input", "image"),
        ("Checkpoint", "inventory_2"),
        ("Output", "texture"),
        ("Metadata", "description"),
    )
