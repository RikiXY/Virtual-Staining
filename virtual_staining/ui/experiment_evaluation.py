from __future__ import annotations

import logging
import math
from pathlib import Path

from nicegui import context, run, ui

from virtual_staining.applications.api import (
    ApplicationError,
    ApplicationService,
    RunDescriptor,
    RunEvaluationRequest,
    RunEvaluationResult,
)
from virtual_staining.ui.experiment_components import (
    AsyncRequestGuard,
    comparison_case_row,
    format_value,
    run_label,
)
from virtual_staining.ui.theme import empty_state, section_heading

logger = logging.getLogger(__name__)

EVALUATION_PLOT_PRIORITY = (
    "ssim_histogram.png",
    "psnr_histogram.png",
    "mae_histogram.png",
    "pcc_rgb_mean_histogram.png",
)

EVALUATION_METRIC_RANGES = {
    "ssim": (0.0, 1.0),
    "psnr": (0.0, 60.0),
    "mae": (0.0, 1.0),
    "pcc_rgb_mean": (-1.0, 1.0),
}


def build_run_evaluation(service: ApplicationService, runs: tuple[RunDescriptor, ...]) -> None:
    run_options = {str(item.path): run_label(item) for item in runs}
    latest_result: RunEvaluationResult | None = None
    request_guard = AsyncRequestGuard()
    # NiceGUI pages may be deleted while background work is awaiting completion.
    # Invalidating the guard prevents stale callbacks from updating detached elements.
    client = context.client
    client.on_disconnect(request_guard.invalidate)
    client.on_delete(request_guard.invalidate)

    with ui.column().classes("w-full gap-5"):
        with ui.card().classes("vs-card w-full rounded-xl p-5 md:p-6 gap-4"):
            section_heading(
                "analytics",
                "Run evaluation",
                "Load existing metrics or execute the configured evaluation stage.",
            )
            mode = ui.toggle(
                {"run": "Existing run", "config": "Run from config"}, value="run"
            ).props("no-caps")
            run_select = ui.select(run_options, label="Evaluated run").classes("w-full")
            config_path = ui.input(
                label="Run config path", placeholder="configs/experiment.yaml"
            ).classes("w-full")
            config_path.set_visibility(False)
            with ui.row().classes("items-center gap-4 flex-wrap"):
                panels_toggle = ui.checkbox("Build representative panels", value=True)
                evaluate_button = ui.button("Load evaluation", icon="analytics").props(
                    "unelevated color=primary"
                )
                loading_indicator = ui.label("Loading…").classes(
                    "vs-evaluation-status text-sm text-slate-500 whitespace-nowrap"
                )
            loading_indicator.set_visibility(False)
            if not runs:
                ui.label(
                    "No evaluated runs were discovered. Use a config path or check --results-dir."
                ).classes("text-base text-amber-700")

        result_card = ui.card().classes("vs-card w-full rounded-xl p-5 md:p-6 gap-5")
        result_card.set_visibility(False)
        with result_card:
            result_heading = ui.label().classes("text-2xl font-semibold")
            result_meta = ui.label().classes("text-base text-slate-500")
            warnings_container = ui.column().classes("w-full gap-1")
            summary_container = ui.column().classes("w-full gap-3")
            plots_container = ui.column().classes("w-full gap-3")
            with ui.row().classes("w-full items-center justify-between gap-3 flex-wrap"):
                ui.label("Representative samples").classes("text-xl font-semibold")
                representative_metric = ui.select(
                    list(service.supported_metrics),
                    value=service.supported_metrics[0],
                    label="Metric",
                ).classes("w-48")
            representatives_container = ui.column().classes("vs-representative-grid w-full gap-4")

    def change_mode() -> None:
        nonlocal latest_result
        request_guard.invalidate()
        latest_result = None
        result_card.set_visibility(False)
        loading_indicator.set_visibility(False)
        evaluate_button.enable()
        is_config = mode.value == "config"
        run_select.set_visibility(not is_config)
        config_path.set_visibility(is_config)
        evaluate_button.set_text("Run evaluation" if is_config else "Load evaluation")

    def show_representatives() -> None:
        representatives_container.clear()
        if latest_result is None:
            return
        samples = latest_result.representatives.get(str(representative_metric.value), ())
        with representatives_container:
            if not samples:
                empty_state(
                    "image_not_supported",
                    "No representative images available",
                    "The metric exists, but source/target/generated paths could not be resolved.",
                )
            order = {"best": 0, "median": 1, "worst": 2}
            for sample in sorted(samples, key=lambda item: order.get(item.kind, 3)):
                comparison_case_row(sample)

    async def execute() -> None:
        nonlocal latest_result
        if mode.value == "config":
            if not str(config_path.value or "").strip():
                ui.notify("Enter a run config path.", type="warning")
                return
            request = RunEvaluationRequest(
                config_path=Path(str(config_path.value)),
                build_representative_panels=bool(panels_toggle.value),
            )
        else:
            if not run_select.value:
                ui.notify("Choose an evaluated run.", type="warning")
                return
            request = RunEvaluationRequest(
                run_path=Path(str(run_select.value)),
                build_representative_panels=bool(panels_toggle.value),
            )
        request_generation = request_guard.start()
        evaluate_button.disable()
        loading_indicator.set_text(
            "Running evaluation…"
            if mode.value == "config"
            else "Loading evaluation data… This may take a few minutes."
        )
        loading_indicator.set_visibility(True)
        try:
            completed = await run.io_bound(service.evaluate_run, request)
        except ApplicationError as exc:
            if request_guard.is_current(request_generation) and not client.is_deleted:
                ui.notify(str(exc), type="negative", multi_line=True, close_button=True)
        except Exception:
            logger.exception("Unexpected run-evaluation UI failure")
            if request_guard.is_current(request_generation) and not client.is_deleted:
                ui.notify("An unexpected run evaluation error occurred.", type="negative")
        else:
            if (
                completed is None
                or not request_guard.is_current(request_generation)
                or client.is_deleted
            ):
                return
            latest_result = completed
            result_heading.set_text(completed.run.display_name)
            result_meta.set_text(f"{len(completed.rows)} evaluated samples · {completed.run.path}")
            render_warnings(completed.warnings, warnings_container)
            render_run_summary(completed, summary_container)
            render_plots(completed.plot_paths, plots_container, completed.rows)
            available = list(completed.summary)
            representative_metric.set_options(available, value=available[0] if available else None)
            show_representatives()
            result_card.set_visibility(True)
            ui.notify("Evaluation results ready.", type="positive")
        finally:
            if request_guard.is_current(request_generation) and not client.is_deleted:
                loading_indicator.set_visibility(False)
                evaluate_button.enable()

    mode.on_value_change(lambda _event: change_mode())
    representative_metric.on_value_change(lambda _event: show_representatives())
    evaluate_button.on_click(execute)
    client.on_connect(change_mode)


def render_run_summary(result: RunEvaluationResult, container) -> None:
    container.clear()
    with container:
        ui.label("Aggregate metrics").classes("text-xl font-semibold")
        rows = [
            {
                "metric": metric.upper().replace("_", " "),
                "mean": format_value(values["mean"]),
                "median": format_value(values["median"]),
                "std": format_value(values["std"]),
                "min": format_value(values["min"]),
                "max": format_value(values["max"]),
                "n": int(values["finite_count"]),
            }
            for metric, values in result.summary.items()
        ]
        ui.table(
            columns=[
                {"name": key, "label": label, "field": key, "align": align}
                for key, label, align in (
                    ("metric", "Metric", "left"),
                    ("mean", "Mean", "right"),
                    ("median", "Median", "right"),
                    ("std", "Std", "right"),
                    ("min", "Min", "right"),
                    ("max", "Max", "right"),
                    ("n", "N", "right"),
                )
            ],
            rows=rows,
            row_key="metric",
        ).props("flat bordered dense").classes("w-full")
        with ui.expansion(f"Per-sample results ({len(result.rows)})", icon="table_rows").classes(
            "w-full"
        ):
            available_metrics = [
                metric for metric in result.summary if result.rows and metric in result.rows[0]
            ]
            sample_columns = [
                {"name": "sample_id", "label": "Sample", "field": "sample_id", "align": "left"},
                *[
                    {
                        "name": metric,
                        "label": metric.upper().replace("_", " "),
                        "field": metric,
                        "align": "right",
                    }
                    for metric in available_metrics
                ],
            ]
            sample_rows = [
                {
                    "sample_id": row.get("sample_id", ""),
                    **{metric: format_value(float(row[metric])) for metric in available_metrics},
                }
                for row in result.rows
            ]
            ui.table(
                columns=sample_columns,
                rows=sample_rows,
                row_key="sample_id",
                pagination=10,
            ).props("flat bordered dense").classes("w-full")


def render_warnings(warnings: tuple[str, ...], container) -> None:
    container.clear()
    with container:
        for warning in warnings:
            with ui.row().classes("w-full items-start gap-2 text-amber-800"):
                ui.icon("warning_amber", size="xs")
                ui.label(warning).classes("text-base")


def render_plots(
    paths: tuple[Path, ...],
    container,
    rows: tuple[dict[str, str], ...] = (),
) -> None:
    container.clear()
    with container:
        ui.label("Interactive metric distributions").classes("text-xl font-semibold")
        ui.label(
            "SSIM, PSNR, MAE, and RGB correlation provide complementary views; "
            "bars show exact bin shares and the translucent curve shows the smoothed trend."
        ).classes("text-sm text-slate-500")
        with ui.row().classes("items-center gap-1 text-teal-700"):
            ui.icon("zoom_in", size="xs")
            ui.label(
                "Use Shift + Scroll to zoom, then hold and drag to move left or right; "
                "normal scrolling moves the page."
            ).classes("text-sm font-medium")
        selected_paths = select_evaluation_plot_paths(paths)
        if not selected_paths:
            empty_state("insert_chart", "No plots available", "The numeric summary is still valid.")
            return
        with ui.row().classes("w-full grid grid-cols-1 lg:grid-cols-2 gap-4"):
            for path in selected_paths:
                with ui.column().classes("vs-subtle rounded-lg p-3 gap-2 min-w-0"):
                    metric = path.stem.removesuffix("_histogram")
                    values = _finite_row_values(rows, metric)
                    ui.label(metric.replace("_", " ").upper()).classes("text-base font-medium")
                    if values and metric in EVALUATION_METRIC_RANGES:
                        ui.echart(_histogram_options(metric, values)).classes(
                            "vs-evaluation-chart w-full h-80"
                        )
                    else:
                        ui.image(path).props("fit=contain").classes("w-full rounded")


def _finite_row_values(rows: tuple[dict[str, str], ...], metric: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        try:
            value = float(row[metric])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return values


def _histogram_options(metric: str, values: list[float], bin_count: int = 24) -> dict:
    lower, upper = EVALUATION_METRIC_RANGES[metric]
    counts = [0] * bin_count
    for value in values:
        if not lower <= value <= upper:
            continue
        index = min(bin_count - 1, int((value - lower) / (upper - lower) * bin_count))
        counts[index] += 1
    denominator = max(1, sum(counts))
    width = (upper - lower) / bin_count
    labels = [
        f"{lower + index * width:.3g}–{lower + (index + 1) * width:.3g}"
        for index in range(bin_count)
    ]
    centers = [lower + (index + 0.5) * width for index in range(bin_count)]
    shares = [count / denominator for count in counts]
    occupied = [index for index, count in enumerate(counts) if count]
    curve_start = max(0, occupied[0] - 1) if occupied else 0
    curve_end = min(bin_count - 1, occupied[-1] + 1) if occupied else -1
    curve_shares: list[float | None] = [
        share if curve_start <= index <= curve_end else None for index, share in enumerate(shares)
    ]
    # Omitting empty bars avoids a misleading one-pixel row along the baseline.
    bar_data = [
        {"value": [centers[index], shares[index]], "binLabel": labels[index]}
        for index, count in enumerate(counts)
        if count
    ]
    return {
        "animation": True,
        "animationDuration": 700,
        "animationEasing": "cubicOut",
        "color": ["#14b8a6"],
        "grid": {"left": 54, "right": 24, "top": 58, "bottom": 48},
        "tooltip": {
            "trigger": "item",
            "renderMode": "html",
            "className": "vs-evaluation-tooltip",
            "showDelay": 0,
            "hideDelay": 0,
            "transitionDuration": 0,
            "confine": True,
            "backgroundColor": "#020617",
            "borderColor": "#5eead4",
            "borderWidth": 2,
            "padding": [10, 12],
            "textStyle": {"color": "#ffffff", "fontSize": 15, "fontWeight": 600},
            "extraCssText": (
                "background-color:#020617 !important; color:#ffffff !important; "
                "border:2px solid #5eead4 !important; opacity:1 !important; "
                "transition:none !important; "
                "box-shadow:0 8px 24px rgba(0,0,0,.55);"
            ),
            ":formatter": (
                "item => { if (item.seriesType !== 'bar') return ''; "
                "const value = item.value[1]; "
                "return '<strong>' + item.data.binLabel + '</strong><br/>' + "
                "'<span style=\"color:#2dd4bf\">●</span> Histogram share &nbsp; ' + "
                "(value * 100).toFixed(1) + '%'; }"
            ),
        },
        "toolbox": {
            "top": 0,
            "right": 8,
            "feature": {
                "restore": {},
                "saveAsImage": {"name": f"{metric}_histogram"},
            },
        },
        # ECharts handles navigation natively: page scrolling remains untouched,
        # Shift+Scroll zooms, and dragging pans the visible range.
        "dataZoom": [
            {
                "type": "inside",
                "xAxisIndex": 0,
                "filterMode": "none",
                "zoomOnMouseWheel": "shift",
                "moveOnMouseWheel": False,
                "moveOnMouseMove": True,
                "preventDefaultMouseMove": True,
                "minValueSpan": (upper - lower) / 200,
            }
        ],
        "xAxis": {
            "type": "value",
            "min": lower,
            "max": upper,
            "splitNumber": 6,
            "axisLabel": {"color": "#64748b"},
            "axisLine": {"lineStyle": {"color": "#cbd5e1"}},
        },
        "yAxis": {
            "type": "value",
            "name": "Share",
            "axisLabel": {":formatter": "value => Math.round(value * 100) + '%'"},
            "splitLine": {"lineStyle": {"color": "#e2e8f0"}},
        },
        "series": [
            {
                "name": "Distribution",
                "type": "line",
                "data": [
                    [center, share] for center, share in zip(centers, curve_shares, strict=True)
                ],
                "smooth": 0.35,
                "symbol": "none",
                "silent": True,
                "z": 1,
                "lineStyle": {"width": 2, "color": "#7dd3fc", "opacity": 0.42},
                "areaStyle": {"color": "#7dd3fc", "opacity": 0.08},
            },
            {
                "name": "Samples",
                "type": "bar",
                "data": bar_data,
                "barWidth": max(4, round(220 / bin_count)),
                "barMaxWidth": 24,
                "z": 3,
                "itemStyle": {
                    "color": "#0d9488",
                    "borderColor": "#5eead4",
                    "borderWidth": 1,
                    "borderRadius": [4, 4, 0, 0],
                    "opacity": 1,
                },
                "emphasis": {
                    "focus": "self",
                    "itemStyle": {
                        "color": "#f59e0b",
                        "borderColor": "#fef3c7",
                        "borderWidth": 2,
                    },
                },
            },
        ],
    }


def select_evaluation_plot_paths(paths: tuple[Path, ...]) -> tuple[Path, ...]:
    """Keep the dashboard concise while preserving all saved artifacts on disk."""
    by_name = {path.name.lower(): path for path in paths}
    selected = tuple(by_name[name] for name in EVALUATION_PLOT_PRIORITY if name in by_name)
    if selected:
        return selected
    return tuple(
        path
        for path in paths
        if path.name.lower() not in {"metrics_boxplot.png", "pcc_gray_histogram.png"}
    )[:4]
