from __future__ import annotations

from pathlib import Path

from nicegui import ui

from virtual_staining.applications.api import ApplicationService
from virtual_staining.ui.experiments_page import build_experiments_page
from virtual_staining.ui.inference_page import build_inference_page
from virtual_staining.ui.theme import configure_theme
from virtual_staining.ui.tutorial import build_tutorial

APP_ICON_PATH = Path(__file__).with_name("assets") / "icon.png"


def _build_page(service: ApplicationService, output_directory: Path) -> None:
    catalog = service.discover_models()
    runs = service.discover_runs()
    configure_theme()
    open_tutorial = build_tutorial()

    with (
        ui.left_drawer(value=False, bordered=True)
        .props("overlay width=320")
        .classes("vs-menu-drawer p-0") as menu_drawer,
        ui.column().classes("w-full h-full gap-0"),
    ):
        with ui.row().classes("w-full items-center gap-3 px-5 py-5"):
            with ui.element("div").classes(
                "w-10 h-10 shrink-0 rounded-lg bg-teal-50 flex items-center justify-center"
            ):
                ui.icon("biotech", size="sm").classes("text-teal-700")
            with ui.column().classes("gap-0 min-w-0"):
                ui.label("Virtual Staining").classes("text-lg font-semibold leading-tight")
                ui.label("Research workspace").classes("text-sm text-slate-500")
        ui.separator()
        ui.label("WORKSPACE").classes(
            "px-5 pt-5 pb-2 text-sm font-semibold tracking-wider text-slate-400"
        )
        inference_menu_button = (
            ui.button("Inference", icon="auto_awesome")
            .props('flat no-caps align=left aria-label="Open Inference"')
            .classes("vs-menu-item mx-3")
        )
        experiments_menu_button = (
            ui.button("Experiments", icon="science")
            .props('flat no-caps align=left aria-label="Open Experiments"')
            .classes("vs-menu-item mx-3")
        )
        ui.space()
        ui.separator()
        tutorial_menu_button = (
            ui.button("Tutorial", icon="school", on_click=open_tutorial)
            .props('flat no-caps align=left aria-label="Open tutorial"')
            .classes("vs-menu-item vs-tutorial-button mx-3 my-3")
        )
        tutorial_menu_button.tooltip("Open tutorial")

    with (
        ui.header().classes("vs-header h-auto px-0 text-slate-800"),
        ui.element("div").classes("vs-header-bar w-full max-w-7xl mx-auto px-4 sm:px-6 py-4 gap-4"),
    ):
        menu_button = (
            ui.button(icon="menu", on_click=menu_drawer.toggle)
            .props('round flat color=primary aria-label="Open navigation menu"')
            .classes("vs-menu-trigger")
        )
        menu_button.tooltip("Open navigation")
        with ui.row().classes("items-center gap-4 min-w-0 justify-self-start"):
            with ui.element("div").classes(
                "w-14 h-14 shrink-0 rounded-xl bg-teal-50 flex items-center justify-center"
            ):
                ui.icon("biotech", size="md").classes("vs-brand-symbol text-teal-700")
            with ui.column().classes("vs-brand-copy gap-0 min-w-0"):
                ui.label("Virtual Staining").classes(
                    "text-2xl font-semibold leading-tight whitespace-nowrap"
                )
                ui.label("Research workspace").classes("text-base text-slate-500 whitespace-nowrap")
        current_page_label = ui.label("Inference").classes(
            "vs-current-page justify-self-end text-base font-semibold text-teal-700"
        )

    with ui.column().classes("w-full max-w-7xl mx-auto px-4 sm:px-6 py-6 md:py-8 gap-5"):
        content = ui.column().classes("w-full")
        with content:
            inference_page = ui.column().classes("vs-inference-page w-full")
            with inference_page:
                build_inference_page(service, catalog, output_directory)
            experiments_page = ui.column().classes("vs-experiments-page w-full")
            with experiments_page:
                build_experiments_page(service, catalog, runs)
            experiments_page.set_visibility(False)

        ui.separator().classes("mt-4")
        with ui.row().classes("w-full justify-between gap-3 text-sm text-slate-500 pb-4"):
            ui.label("Research software · outputs retain reproducibility metadata")
            ui.label("Not a clinically validated diagnostic system")

    def select_page(name: str) -> None:
        inference_selected = name == "inference"
        inference_page.set_visibility(inference_selected)
        experiments_page.set_visibility(not inference_selected)
        selected_name = "Inference" if inference_selected else "Experiments"
        current_page_label.set_text(selected_name)
        for button, selected in (
            (inference_menu_button, inference_selected),
            (experiments_menu_button, not inference_selected),
        ):
            if selected:
                button.props(remove="flat")
                button.props("unelevated color=primary")
            else:
                button.props(remove="unelevated color")
                button.props("flat")
        menu_drawer.hide()

    inference_menu_button.on_click(lambda: select_page("inference"))
    experiments_menu_button.on_click(lambda: select_page("experiments"))
    select_page("inference")


def run_ui(
    checkpoint_directory: Path,
    output_directory: Path,
    results_directory: Path = Path("results"),
    *,
    host: str = "0.0.0.0",
    port: int = 8080,
) -> None:
    """Configure and launch the single-entry-point NiceGUI application."""
    service = ApplicationService(
        checkpoint_directory,
        output_directory,
        results_directory,
    )

    @ui.page("/")
    def index() -> None:
        _build_page(service, output_directory)

    ui.run(
        title="Virtual Staining",
        favicon=APP_ICON_PATH,
        host=host,
        port=port,
        reload=False,
        show=False,
    )
