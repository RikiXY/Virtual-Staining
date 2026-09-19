"""Compose the single-sample, evaluation, and comparison experiment workflows."""

from __future__ import annotations

from nicegui import ui

from virtual_staining.applications.api import ApplicationService, ModelCatalog, RunDescriptor
from virtual_staining.ui.experiment_comparison import build_run_comparison
from virtual_staining.ui.experiment_evaluation import build_run_evaluation
from virtual_staining.ui.experiment_single_sample import build_single_sample


def build_experiments_page(
    service: ApplicationService,
    catalog: ModelCatalog,
    runs: tuple[RunDescriptor, ...],
) -> None:
    with ui.column().classes("w-full gap-5"):
        with ui.column().classes("gap-1"):
            ui.label("Experiments").classes("text-3xl md:text-4xl font-semibold tracking-tight")
            ui.label(
                "Evaluate predictions against ground truth and compare reproducible runs."
            ).classes("text-slate-500")
        with (
            ui.tabs()
            .props("align=center no-caps")
            .classes("vs-section-tabs text-slate-600")
            .style("align-self: center; width: auto; max-width: 100%")
        ) as tabs:
            test_tab = ui.tab("Test / Single sample", icon="science")
            evaluate_tab = ui.tab("Evaluate", icon="analytics")
            compare_tab = ui.tab("Compare", icon="compare_arrows")
        with ui.tab_panels(tabs, value=test_tab, animated=False).classes(
            "w-full bg-transparent p-0"
        ):
            with ui.tab_panel(test_tab).classes("p-0"):
                build_single_sample(service, catalog)
            with ui.tab_panel(evaluate_tab).classes("p-0"):
                build_run_evaluation(service, runs)
            with ui.tab_panel(compare_tab).classes("p-0"):
                build_run_comparison(service, runs)
