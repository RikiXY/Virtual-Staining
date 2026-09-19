from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

from nicegui import context, run, ui

from virtual_staining.applications.api import (
    ApplicationError,
    ApplicationService,
    ComparisonRequest,
    ComparisonResult,
    RunDescriptor,
)
from virtual_staining.ui.experiment_components import (
    AsyncRequestGuard,
    comparison_case_row,
    metric_card,
    run_label,
)
from virtual_staining.ui.theme import section_heading

logger = logging.getLogger(__name__)


def build_run_comparison(service: ApplicationService, runs: tuple[RunDescriptor, ...]) -> None:
    evaluated = tuple(item for item in runs if item.has_evaluation)
    run_options = {str(item.path): run_label(item) for item in evaluated}
    result_cache: dict[tuple[str, str, str, str], ComparisonResult] = {}
    request_guard = AsyncRequestGuard()
    # NiceGUI pages may be deleted while background work is awaiting completion.
    # Invalidating the guard prevents stale callbacks from updating detached elements.
    client = context.client
    client.on_disconnect(request_guard.invalidate)
    client.on_delete(request_guard.invalidate)

    with ui.column().classes("w-full gap-5"):
        with ui.card().classes("vs-card w-full rounded-xl p-5 md:p-6 gap-4"):
            section_heading(
                "compare_arrows",
                "Compare runs",
                "Use paired mode for shared samples and unpaired mode for independent sets.",
            )
            with ui.row().classes("w-full flex-col md:flex-row gap-4"):
                run_a = ui.select(run_options, label="Run A").classes("flex-1")
                run_b = ui.select(run_options, label="Run B").classes("flex-1")
            with ui.row().classes("w-full flex-col sm:flex-row gap-4"):
                metric = ui.select(
                    list(service.supported_metrics),
                    value="ssim",
                    label="Metric",
                ).classes("flex-1")
                comparison_mode = ui.select(
                    {"paired": "Paired (same samples)", "unpaired": "Unpaired"},
                    value="paired",
                    label="Comparison mode",
                ).classes("flex-1")
            compare_button = ui.button("Compare runs", icon="compare_arrows").props(
                "unelevated color=primary"
            )
            if len(evaluated) < 2:
                ui.label("At least two runs with evaluation results are required.").classes(
                    "text-base text-amber-700"
                )

        result_card = ui.card().classes("vs-card w-full rounded-xl p-5 md:p-6 gap-5")
        result_card.set_visibility(False)
        with result_card:
            comparison_heading = ui.label().classes("text-2xl font-semibold")
            comparison_note = ui.label().classes("text-base text-slate-500")
            comparison_summary = ui.row().classes(
                "w-full grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3"
            )
            comparison_plots = ui.column().classes("w-full gap-3")
            qualitative = ui.column().classes("w-full gap-4")

    async def execute() -> None:
        if not run_a.value or not run_b.value:
            ui.notify("Choose two evaluated runs.", type="warning")
            return
        run_a_value = str(run_a.value)
        run_b_value = str(run_b.value)
        metric_value = str(metric.value)
        try:
            mode_value = normalise_comparison_mode(comparison_mode.value)
        except ValueError as exc:
            ui.notify(str(exc), type="warning")
            return
        cache_key = (run_a_value, run_b_value, metric_value, mode_value)
        request_generation = request_guard.start()
        controls = (run_a, run_b, metric, comparison_mode)
        for control in controls:
            control.disable()
        compare_button.disable()
        compare_button.props("loading")
        try:
            result = result_cache.get(cache_key)
            if result is None:
                completed = await run.io_bound(
                    service.compare_runs,
                    ComparisonRequest(
                        run_a=Path(run_a_value),
                        run_b=Path(run_b_value),
                        metric=metric_value,
                        mode=mode_value,
                    ),
                )
                if completed is not None:
                    result = completed
                    result_cache[cache_key] = completed
        except ApplicationError as exc:
            if request_guard.is_current(request_generation) and not client.is_deleted:
                ui.notify(str(exc), type="negative", multi_line=True, close_button=True)
        except Exception:
            logger.exception("Unexpected run-comparison UI failure")
            if request_guard.is_current(request_generation) and not client.is_deleted:
                ui.notify("An unexpected comparison error occurred.", type="negative")
        else:
            if (
                result is None
                or not request_guard.is_current(request_generation)
                or client.is_deleted
            ):
                return
            render_comparison(
                result,
                comparison_heading,
                comparison_note,
                comparison_summary,
                comparison_plots,
                qualitative,
            )
            result_card.set_visibility(True)
            ui.notify("Comparison complete.", type="positive")
        finally:
            if request_guard.is_current(request_generation) and not client.is_deleted:
                compare_button.props(remove="loading")
                compare_button.enable()
                for control in controls:
                    control.enable()

    def reset_result() -> None:
        request_guard.invalidate()
        result_card.set_visibility(False)
        compare_button.props(remove="loading")
        compare_button.enable()
        for control in (run_a, run_b, metric, comparison_mode):
            control.enable()

    run_a.on_value_change(lambda _event: reset_result())
    run_b.on_value_change(lambda _event: reset_result())
    metric.on_value_change(lambda _event: reset_result())
    comparison_mode.on_value_change(lambda _event: reset_result())
    compare_button.on_click(execute)
    client.on_connect(reset_result)


def normalise_comparison_mode(value: object) -> Literal["paired", "unpaired"]:
    """Normalise select values across browser updates and NiceGUI label/key forms."""
    normalised = str(value).strip().lower()
    if normalised in {"paired", "paired (same samples)"}:
        return "paired"
    if normalised == "unpaired":
        return "unpaired"
    raise ValueError("Choose paired or unpaired comparison mode.")


def render_comparison(
    result: ComparisonResult,
    heading,
    note,
    summary_container,
    plots_container,
    qualitative_container,
) -> None:
    heading.set_text(f"{result.label_a} vs {result.label_b}")
    direction = "higher" if result.higher_is_better else "lower"
    note.set_text(
        f"{result.mode.title()} · {result.metric.upper()} ({direction} is better) · "
        f"outputs: {result.output_directory}"
    )
    summary_container.clear()
    with summary_container:
        for key, value in result.summary.items():
            metric_card(key, value)
    plots_container.clear()
    with plots_container:
        ui.label("Statistical plots").classes("text-xl font-semibold")
        with ui.row().classes(
            "vs-statistical-plots w-full grid grid-cols-1 lg:grid-cols-2 gap-4 items-stretch"
        ):
            for path in result.plot_paths:
                with ui.column().classes(
                    "vs-statistical-plot-card vs-subtle h-full rounded-lg p-3 gap-2 min-w-0"
                ):
                    ui.label(path.stem.replace("_", " ").title()).classes("text-base font-medium")
                    with ui.element("div").classes(
                        "vs-statistical-plot-frame w-full flex items-center justify-center rounded"
                    ):
                        ui.image(path).props("fit=contain").classes(
                            "vs-statistical-plot-image w-full h-full rounded"
                        )
    qualitative_container.clear()
    with qualitative_container:
        ui.label("Representative case quadruplets").classes("text-xl font-semibold")
        ui.label(
            "Each row shows source, generated, target, and the absolute difference map. "
            "Cases are selected independently within each run and grouped by rank for comparison."
        ).classes("text-sm text-slate-500 -mt-3")
        samples_a = {sample.kind: sample for sample in result.representatives_a}
        samples_b = {sample.kind: sample for sample in result.representatives_b}
        for kind in ("best", "median", "worst"):
            ranked_samples = tuple(
                (label, samples[kind])
                for label, samples in (
                    (result.label_a, samples_a),
                    (result.label_b, samples_b),
                )
                if kind in samples
            )
            if not ranked_samples:
                continue
            with ui.column().classes(
                "vs-ranked-comparison vs-subtle w-full gap-3 rounded-xl p-3 min-w-0"
            ):
                ui.label(f"{kind.title()} cases").classes("text-lg font-semibold text-teal-800")
                for label, sample in ranked_samples:
                    ui.label(label).classes("text-base font-semibold text-slate-700")
                    comparison_case_row(sample)
