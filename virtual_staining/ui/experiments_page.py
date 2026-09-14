from __future__ import annotations

import logging
import math
from collections.abc import Callable
from io import BytesIO
from pathlib import Path
from time import monotonic
from typing import Literal

from nicegui import events, run, ui
from PIL import Image

from virtual_staining.applications.api import (
    ApplicationError,
    ApplicationService,
    ComparisonRequest,
    ComparisonResult,
    GeneratedSampleEvaluationRequest,
    InferenceRequest,
    InferenceResult,
    ModelCatalog,
    RepresentativeSample,
    RunDescriptor,
    RunEvaluationRequest,
    RunEvaluationResult,
    SingleSampleResult,
    metric_quality,
)
from virtual_staining.ui.theme import empty_state, metric_quality_style, section_heading

logger = logging.getLogger(__name__)

EVALUATION_PLOT_PRIORITY = (
    "ssim_histogram.png",
    "psnr_histogram.png",
    "mae_histogram.png",
    "pcc_rgb_mean_histogram.png",
)


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
        with ui.tab_panels(tabs, value=test_tab).classes("w-full bg-transparent p-0"):
            with ui.tab_panel(test_tab).classes("p-0"):
                _build_single_sample(service, catalog)
            with ui.tab_panel(evaluate_tab).classes("p-0"):
                _build_run_evaluation(service, runs)
            with ui.tab_panel(compare_tab).classes("p-0"):
                _build_run_comparison(service, runs)


def _build_single_sample(service: ApplicationService, catalog: ModelCatalog) -> None:
    source: Image.Image | None = None
    source_filename: str | None = None
    target: Image.Image | None = None
    target_filename: str | None = None
    generated: InferenceResult | None = None
    models = {model.identifier: model for model in catalog.models}

    with ui.column().classes("w-full gap-5"):
        with ui.card().classes("vs-card w-full rounded-xl p-5 md:p-6 gap-4"):
            section_heading(
                "science",
                "Generate, then optionally evaluate",
                "A source is enough to generate. Add a target before or after generation "
                "when you want canonical comparison metrics.",
            )
            model_select = ui.select(
                {model.identifier: model.display_name for model in catalog.models},
                value=catalog.models[0].identifier if catalog.models else None,
                label="Checkpoint / model",
            ).classes("w-full")
            if not catalog.models:
                model_select.disable()
                empty_state(
                    "inventory_2",
                    "A compatible checkpoint is required",
                    "Configure the checkpoint directory, then restart the application.",
                )
            with ui.row().classes("w-full flex-col md:flex-row gap-4"):
                source_uploader, source_preview, source_empty, source_status = _upload_card(
                    "Source", "Input to the model"
                )
                target_uploader, target_preview, target_empty, target_status = _upload_card(
                    "Target", "Ground-truth stained image"
                )
            with ui.row().classes("w-full gap-3 flex-wrap"):
                generate_button = ui.button("Generate", icon="auto_awesome").props(
                    "unelevated color=primary"
                )
                evaluate_button = ui.button("Evaluate against target", icon="fact_check").props(
                    "outline color=primary"
                )
                generate_button.disable()
                evaluate_button.disable()
            status = ui.label("Upload a source image to generate.").classes(
                "text-base text-slate-500"
            )

        result_card = ui.card().classes("vs-card w-full rounded-xl p-5 md:p-6 gap-5")
        result_card.set_visibility(False)
        with result_card:
            section_heading(
                "fact_check",
                "Sample result",
                "Qualitative alignment and quantitative image-quality metrics.",
            )
            result_note = ui.label().classes("text-base text-slate-500")
            images_container = ui.row().classes(
                "vs-sample-images w-full grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-4"
            )
            metrics_container = ui.row().classes(
                "vs-sample-metrics w-full grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3"
            )
            overlap_container = ui.column().classes("vs-sample-comparison w-full gap-3")

    def can_generate() -> bool:
        return source is not None and str(model_select.value or "") in models

    def can_evaluate() -> bool:
        return generated is not None and target is not None and target_filename is not None

    def update_ready() -> None:
        if can_generate():
            generate_button.enable()
        else:
            generate_button.disable()
        if can_evaluate():
            evaluate_button.enable()
            status.set_text("Generated image and target are ready for comparison.")
        else:
            evaluate_button.disable()
            if generated is not None:
                status.set_text(
                    "Generation complete. Upload a target whenever you want to evaluate it."
                )
            elif can_generate():
                status.set_text("Source ready. A target is optional for generation.")

    def clear_generated() -> None:
        nonlocal generated
        generated = None
        result_card.set_visibility(False)

    async def upload_source(event: events.UploadEventArguments) -> None:
        nonlocal source, source_filename
        clear_generated()
        try:
            candidate = await _read_upload(event)
            model_id = str(model_select.value or "")
            if not model_id:
                raise ApplicationError("Choose a model before uploading the source.")
            service.validate_inference_input(model_id, candidate)
        except (ApplicationError, OSError) as exc:
            source = None
            source_filename = None
            _upload_error(exc)
            source_uploader.reset()
            source_preview.set_visibility(False)
            source_empty.set_visibility(True)
            source_status.set_text("No valid source selected")
        else:
            source = candidate
            source_filename = event.file.name
            _show_upload(candidate, source_preview, source_empty, source_status)
        update_ready()

    async def upload_target(event: events.UploadEventArguments) -> None:
        nonlocal target, target_filename
        try:
            candidate = await _read_upload(event)
            if candidate.mode != "RGB":
                raise ApplicationError(f"The target must be RGB; received {candidate.mode}.")
        except (ApplicationError, OSError) as exc:
            target = None
            target_filename = None
            _upload_error(exc)
            target_uploader.reset()
            target_preview.set_visibility(False)
            target_empty.set_visibility(True)
            target_status.set_text("No valid target selected")
            if generated is not None:
                _render_pending_generation(
                    generated,
                    None,
                    images_container,
                    overlap_container,
                    metrics_container,
                )
        else:
            target = candidate
            target_filename = event.file.name
            _show_upload(candidate, target_preview, target_empty, target_status)
            if generated is not None:
                _render_pending_generation(
                    generated,
                    target,
                    images_container,
                    overlap_container,
                    metrics_container,
                )
                result_card.set_visibility(True)
        update_ready()

    async def generate() -> None:
        nonlocal generated
        if not can_generate() or source_filename is None:
            ui.notify("Upload a valid source image first.", type="warning")
            return
        assert source is not None
        generate_button.disable()
        evaluate_button.disable()
        generate_button.props("loading")
        status.set_text("Generating the virtual stain…")
        try:
            completed = await run.io_bound(
                service.run_inference,
                InferenceRequest(str(model_select.value), source, source_filename),
            )
        except ApplicationError as exc:
            status.set_text("Generation failed. Review the validation message.")
            ui.notify(str(exc), type="negative", multi_line=True, close_button=True)
        except Exception:
            logger.exception("Unexpected experimental inference UI failure")
            status.set_text("Generation failed.")
            ui.notify("An unexpected inference error occurred.", type="negative")
        else:
            assert completed is not None
            generated = completed
            _render_pending_generation(
                generated,
                target,
                images_container,
                overlap_container,
                metrics_container,
            )
            result_note.set_text(
                "Generated in memory. Add a target and evaluate to save scientific artifacts."
            )
            result_card.set_visibility(True)
            ui.notify("Virtual stain generated.", type="positive")
        finally:
            generate_button.props(remove="loading")
            update_ready()

    async def evaluate() -> None:
        if not can_evaluate():
            ui.notify("Generate an image and upload its target first.", type="warning")
            return
        assert generated is not None and target is not None and target_filename is not None
        evaluate_button.disable()
        evaluate_button.props("loading")
        status.set_text("Evaluating generated image against target…")
        try:
            result = await run.io_bound(
                service.evaluate_generated_sample,
                GeneratedSampleEvaluationRequest(
                    inference=generated,
                    target_image=target,
                    target_filename=target_filename,
                ),
            )
        except ApplicationError as exc:
            status.set_text("Evaluation failed. Review the validation message.")
            ui.notify(str(exc), type="negative", multi_line=True, close_button=True)
        except Exception:
            logger.exception("Unexpected generated-sample evaluation UI failure")
            status.set_text("Evaluation failed.")
            ui.notify("An unexpected evaluation error occurred.", type="negative")
        else:
            assert result is not None
            _render_single_result(
                result,
                images_container,
                overlap_container,
                metrics_container,
            )
            result_note.set_text(f"Evaluation artifacts: {result.output_directory}")
            status.set_text("Evaluation complete.")
            ui.notify("Sample evaluated.", type="positive")
        finally:
            evaluate_button.props(remove="loading")
            update_ready()

    def change_model() -> None:
        clear_generated()
        update_ready()

    source_uploader.on_upload(upload_source)
    target_uploader.on_upload(upload_target)
    model_select.on_value_change(lambda _event: change_model())
    generate_button.on_click(generate)
    evaluate_button.on_click(evaluate)


def _build_run_evaluation(service: ApplicationService, runs: tuple[RunDescriptor, ...]) -> None:
    run_options = {str(item.path): _run_label(item) for item in runs}
    latest_result: RunEvaluationResult | None = None
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
            representatives_container = ui.row().classes(
                "w-full grid grid-cols-1 md:grid-cols-3 gap-4"
            )

    def change_mode() -> None:
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
            for sample in samples:
                _representative_card(sample)

    async def execute() -> None:
        nonlocal latest_result
        request: RunEvaluationRequest
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
        evaluate_button.disable()
        evaluate_button.props("loading")
        try:
            completed = await run.io_bound(service.evaluate_run, request)
        except ApplicationError as exc:
            ui.notify(str(exc), type="negative", multi_line=True, close_button=True)
        except Exception:
            logger.exception("Unexpected run-evaluation UI failure")
            ui.notify("An unexpected run evaluation error occurred.", type="negative")
        else:
            assert completed is not None
            latest_result = completed
            result_heading.set_text(completed.run.display_name)
            result_meta.set_text(f"{len(completed.rows)} evaluated samples · {completed.run.path}")
            _render_warnings(completed.warnings, warnings_container)
            _render_run_summary(completed, summary_container)
            _render_plots(completed.plot_paths, plots_container)
            available = list(completed.summary)
            representative_metric.set_options(available, value=available[0] if available else None)
            show_representatives()
            result_card.set_visibility(True)
            ui.notify("Evaluation results ready.", type="positive")
        finally:
            evaluate_button.props(remove="loading")
            evaluate_button.enable()

    mode.on_value_change(lambda _event: change_mode())
    representative_metric.on_value_change(lambda _event: show_representatives())
    evaluate_button.on_click(execute)


def _build_run_comparison(service: ApplicationService, runs: tuple[RunDescriptor, ...]) -> None:
    evaluated = tuple(item for item in runs if item.has_evaluation)
    run_options = {str(item.path): _run_label(item) for item in evaluated}
    result_cache: dict[tuple[str, str, str, str], ComparisonResult] = {}
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
            mode_value = _normalise_comparison_mode(comparison_mode.value)
        except ValueError as exc:
            ui.notify(str(exc), type="warning")
            return
        cache_key = (run_a_value, run_b_value, metric_value, mode_value)
        controls = (run_a, run_b, metric, comparison_mode)
        for control in controls:
            control.disable()
        compare_button.disable()
        compare_button.props("loading")
        try:
            result = result_cache.get(cache_key)
            if result is None:
                result = await run.io_bound(
                    service.compare_runs,
                    ComparisonRequest(
                        run_a=Path(run_a_value),
                        run_b=Path(run_b_value),
                        metric=metric_value,
                        mode=mode_value,
                    ),
                )
                result_cache[cache_key] = result
        except ApplicationError as exc:
            ui.notify(str(exc), type="negative", multi_line=True, close_button=True)
        except Exception:
            logger.exception("Unexpected run-comparison UI failure")
            ui.notify("An unexpected comparison error occurred.", type="negative")
        else:
            assert result is not None
            _render_comparison(
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
            compare_button.props(remove="loading")
            compare_button.enable()
            for control in controls:
                control.enable()

    compare_button.on_click(execute)


def _normalise_comparison_mode(value: object) -> Literal["paired", "unpaired"]:
    """Normalise select values across browser updates and NiceGUI label/key forms."""
    normalised = str(value).strip().lower()
    if normalised in {"paired", "paired (same samples)"}:
        return "paired"
    if normalised == "unpaired":
        return "unpaired"
    raise ValueError("Choose paired or unpaired comparison mode.")


def _upload_card(title: str, subtitle: str):
    with ui.column().classes("flex-1 gap-3 vs-subtle rounded-xl p-4"):
        with ui.column().classes("gap-0"):
            ui.label(title).classes("font-semibold")
            ui.label(subtitle).classes("text-sm text-slate-500")
        with ui.element("div").classes("vs-image-frame w-full aspect-square rounded-lg"):
            empty = ui.label(f"No {title.lower()} selected").classes(
                "absolute inset-0 flex items-center justify-center text-base text-slate-400"
            )
            preview = ui.image().props("fit=contain").classes("vs-image")
            preview.set_visibility(False)
        uploader = (
            ui.upload(label=f"Upload {title.lower()}", auto_upload=True, max_files=1)
            .props('accept=".bmp,.jpeg,.jpg,.png,.tif,.tiff" flat bordered no-thumbnails')
            .classes("w-full")
        )
        status = ui.label("Waiting for image").classes("text-sm text-slate-500")
    return uploader, preview, empty, status


async def _read_upload(event: events.UploadEventArguments) -> Image.Image:
    with Image.open(BytesIO(await event.file.read())) as uploaded:
        uploaded.load()
        return uploaded.copy()


def _show_upload(image: Image.Image, preview, empty, status) -> None:
    preview.set_source(image)
    preview.set_visibility(True)
    empty.set_visibility(False)
    status.set_text(f"{image.mode} · {image.size[0]} × {image.size[1]} px")


def _upload_error(exc: Exception) -> None:
    message = str(exc) if str(exc) else "The upload is not a readable image."
    ui.notify(message, type="negative", multi_line=True, close_button=True)


def _render_pending_generation(
    inference: InferenceResult,
    target: Image.Image | None,
    images,
    overlap,
    metrics,
) -> None:
    images.clear()
    overlap.clear()
    metrics.clear()
    with images:
        for label, image, note in (
            ("Source", inference.source_image, "Model input"),
            ("Generated", inference.generated_image, "Virtual stain"),
        ):
            _result_image(label, image, note)
        if target is not None:
            _result_image("Target", target, "Ready for evaluation")
        else:
            _result_placeholder("Target", "Optional · upload before or after generation")
        _result_placeholder("Absolute difference", "Available after evaluation")
    if target is not None:
        _render_overlap_comparison(inference.generated_image, target, overlap)
    with metrics:
        empty_state(
            "calculate",
            "Generation complete",
            "Upload a target and choose Evaluate against target to calculate metrics.",
        )


def _render_single_result(result: SingleSampleResult, images, overlap, metrics) -> None:
    images.clear()
    overlap.clear()
    metrics.clear()
    with images:
        for label, image, note in (
            ("Source", result.inference.source_image, "Model input"),
            ("Generated", result.inference.generated_image, "Virtual stain"),
            ("Target", result.target_image, "Ground truth"),
            ("Absolute difference", result.difference_map, "Mean RGB error per pixel"),
        ):
            _result_image(label, image, note)
    _render_overlap_comparison(result.inference.generated_image, result.target_image, overlap)
    with metrics:
        for metric in ("ssim", "psnr", "mae", "rmse", "mse", "pcc_rgb_mean", "pcc_gray"):
            _metric_card(metric, result.metrics[metric])


def _render_overlap_comparison(
    generated: Image.Image,
    target: Image.Image,
    container,
) -> None:
    """Render pixel-aligned opacity and horizontal-reveal comparisons."""
    with container:
        section_heading(
            "compare",
            "Generated / target comparison",
            "Inspect alignment with an opacity blend or a horizontal reveal.",
        )
        if generated.size != target.size:
            empty_state(
                "aspect_ratio",
                "Comparison unavailable",
                "The generated and target images must have matching dimensions.",
            )
            return

        with ui.row().classes(
            "vs-comparison-grid w-full grid grid-cols-1 lg:grid-cols-2 gap-5 items-stretch"
        ):
            with ui.column().classes(
                "vs-comparison-panel vs-subtle rounded-xl p-4 gap-3 min-w-0 h-full"
            ):
                _render_opacity_comparison(generated, target)
            with ui.column().classes(
                "vs-comparison-panel vs-subtle rounded-xl p-4 gap-3 min-w-0 h-full"
            ):
                _render_horizontal_reveal(generated, target)


def _render_opacity_comparison(generated: Image.Image, target: Image.Image) -> None:
    with ui.row().classes("vs-comparison-header w-full items-start justify-between gap-3"):
        with ui.column().classes("gap-0 min-w-0"):
            ui.label("Opacity overlap").classes("text-base font-semibold text-slate-800")
            ui.label("Blend the generated image over the target.").classes("text-sm text-slate-500")
        play_button, speed_input, is_looping = _playback_controls("opacity")
    with ui.element("div").classes(
        "vs-image-frame vs-overlap-frame w-full aspect-square rounded-lg"
    ):
        ui.image(target).props("fit=contain").classes("vs-image")
        generated_layer = (
            ui.image(generated)
            .props("fit=contain")
            .classes("vs-image vs-overlap-generated")
            .style("opacity: 0")
        )

    with ui.row().classes("w-full items-center justify-between gap-3"):
        ui.label("Target").classes("text-sm font-medium text-slate-500")
        opacity_label = ui.label("0% generated").classes("text-sm font-semibold text-slate-700")
        ui.label("Generated").classes("text-sm font-medium text-slate-500")

    opacity_slider = (
        ui.slider(min=0, max=100, value=0, step=1)
        .props('aria-label="Generated image opacity" color=primary')
        .classes("w-full -mt-2")
    )

    def update_opacity(event) -> None:
        value = min(100.0, max(0.0, float(event.value)))
        generated_layer.style(f"opacity: {value / 100:.2f}")
        opacity_label.set_text(f"{value:.0f}% generated")

    opacity_slider.on_value_change(update_opacity)
    _attach_progress_animation(
        lambda: float(opacity_slider.value or 0),
        opacity_slider.set_value,
        play_button,
        speed_input,
        is_looping,
        "opacity",
    )


def _render_horizontal_reveal(generated: Image.Image, target: Image.Image) -> None:
    reveal_value = 50.0
    with (
        ui.row().classes("vs-comparison-header w-full items-start gap-3"),
        ui.column().classes("gap-0 min-w-0"),
    ):
        ui.label("Horizontal reveal").classes("text-base font-semibold text-slate-800")
        ui.label("Reveal the generated image over the target from left to right.").classes(
            "text-sm text-slate-500"
        )
    with ui.element("div").classes(
        "vs-image-frame vs-overlap-frame w-full aspect-square rounded-lg"
    ) as reveal_frame:
        ui.image(target).props("fit=contain").classes("vs-image")
        generated_layer = (
            ui.image(generated)
            .props("fit=contain")
            .classes("vs-image vs-reveal-generated")
            .style("clip-path: inset(0 50% 0 0)")
        )
        divider = ui.element("div").classes("vs-reveal-divider").style("left: 50%")

    reveal_label = ui.label("50% revealed").classes(
        "w-full text-center text-sm font-semibold text-slate-700"
    )

    def set_reveal(value: float) -> None:
        nonlocal reveal_value
        value = min(100.0, max(0.0, float(value)))
        reveal_value = value
        generated_layer.style(f"clip-path: inset(0 {100 - value:.0f}% 0 0)")
        divider.style(f"left: {value:.0f}%")
        reveal_label.set_text(f"{value:.0f}% revealed")

    def update_reveal_from_drag(event: events.GenericEventArguments) -> None:
        set_reveal(float(event.args))

    divider.on(
        "pointerdown",
        update_reveal_from_drag,
        throttle=0.03,
        trailing_events=True,
        js_handler=f"""(event) => {{
            event.preventDefault();
            const handle = event.currentTarget;
            const frame = document.getElementById('{reveal_frame.html_id}');
            const layer = document.getElementById('{generated_layer.html_id}');
            if (!frame || !layer) return;
            handle.setPointerCapture(event.pointerId);
            const update = (clientX) => {{
                const bounds = frame.getBoundingClientRect();
                const value = Math.max(0, Math.min(100,
                    ((clientX - bounds.left) / bounds.width) * 100));
                layer.style.clipPath = 'inset(0 ' + (100 - value) + '% 0 0)';
                handle.style.left = value + '%';
                emit(value);
            }};
            const move = (moveEvent) => update(moveEvent.clientX);
            const finish = (finishEvent) => {{
                update(finishEvent.clientX);
                handle.removeEventListener('pointermove', move);
                handle.removeEventListener('pointerup', finish);
                handle.removeEventListener('pointercancel', finish);
                if (handle.hasPointerCapture(event.pointerId))
                    handle.releasePointerCapture(event.pointerId);
            }};
            handle.addEventListener('pointermove', move);
            handle.addEventListener('pointerup', finish, {{once: true}});
            handle.addEventListener('pointercancel', finish, {{once: true}});
            update(event.clientX);
        }}""",
    )


def _playback_controls(name: str):
    looping = False
    with ui.row().classes("items-center justify-end gap-1 flex-wrap"):
        speed_input = (
            ui.number(
                "Speed",
                value=1.0,
                min=0.25,
                max=4.0,
                precision=2,
                step=0.25,
                suffix="×",
            )
            .props(f'dense outlined hide-bottom-space aria-label="{name.title()} animation speed"')
            .classes("vs-playback-speed w-24")
        )
        repeat_button = ui.button(icon="repeat").props(
            f'round flat dense color=blue-grey-7 aria-label="Loop {name} animation" '
            'aria-pressed="false"'
        )
        repeat_button.tooltip("Toggle continuous replay")
        play_button = ui.button(icon="play_arrow").props(
            f'round flat dense color=primary aria-label="Play {name} animation"'
        )
        play_button.tooltip(f"Play {name} animation")

    def toggle_loop() -> None:
        nonlocal looping
        looping = not looping
        if looping:
            repeat_button.props(remove="flat")
            repeat_button.props('unelevated color=primary aria-pressed="true"')
        else:
            repeat_button.props(remove="unelevated")
            repeat_button.props('flat color=blue-grey-7 aria-pressed="false"')

    repeat_button.on_click(toggle_loop)
    return play_button, speed_input, lambda: looping


def _attach_progress_animation(
    get_value: Callable[[], float],
    set_value: Callable[[float], None],
    play_button,
    speed_input,
    is_looping: Callable[[], bool],
    name: str,
) -> None:
    """Animate one comparison from zero with configurable speed and replay."""
    can_resume = False
    zero_hold_until = 0.0

    def stop(*, resumable: bool = False) -> None:
        nonlocal can_resume
        can_resume = resumable
        timer.deactivate()
        play_button.set_icon("play_arrow")
        play_button.props(f'aria-label="Play {name} animation"')

    def advance() -> None:
        nonlocal zero_hold_until
        if monotonic() < zero_hold_until:
            return
        value = get_value()
        if value >= 100:
            if is_looping():
                set_value(0)
                # Keep zero rendered long enough to make the reset visible before replaying.
                zero_hold_until = monotonic() + 0.6
                return
            stop(resumable=False)
            return
        try:
            speed = float(speed_input.value or 1.0)
        except (TypeError, ValueError):
            speed = 1.0
        set_value(min(100.0, value + 2.0 * min(4.0, max(0.25, speed))))

    timer = ui.timer(0.04, advance, active=False, immediate=False)

    def toggle() -> None:
        nonlocal can_resume
        if timer.active:
            stop(resumable=True)
            return
        if not can_resume:
            set_value(0)
        can_resume = False
        play_button.set_icon("pause")
        play_button.props(f'aria-label="Pause {name} animation"')
        timer.activate()

    play_button.on_click(toggle)


def _result_image(label: str, image: Image.Image, note: str) -> None:
    with ui.column().classes("gap-2 min-w-0"):
        with ui.row().classes("w-full items-baseline justify-between gap-2"):
            ui.label(label).classes("text-base font-semibold")
            ui.label(note).classes("text-sm text-slate-400")
        with ui.element("div").classes("vs-image-frame w-full aspect-square rounded-lg"):
            ui.image(image).props("fit=contain").classes("vs-image")


def _result_placeholder(label: str, note: str) -> None:
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


def _metric_card(metric: str, value: float | int | str, detail: str | None = None) -> None:
    numeric_value = float(value) if isinstance(value, int | float) else float("nan")
    quality = metric_quality(metric, numeric_value)
    quality_label, quality_class = metric_quality_style(quality)
    with ui.column().classes(f"vs-metric {quality_class} rounded-lg px-3 py-3 gap-0 min-w-0"):
        ui.label(metric.replace("_", " ").upper()).classes(
            "text-sm font-semibold tracking-wide text-slate-500"
        )
        ui.label(_format_value(value)).classes("text-xl font-semibold text-slate-800 truncate")
        if detail or quality != "unknown":
            ui.label(detail or quality_label).classes("text-sm text-slate-500")


def _render_run_summary(result: RunEvaluationResult, container) -> None:
    container.clear()
    with container:
        ui.label("Aggregate metrics").classes("text-xl font-semibold")
        rows = [
            {
                "metric": metric.upper().replace("_", " "),
                "mean": _format_value(values["mean"]),
                "median": _format_value(values["median"]),
                "std": _format_value(values["std"]),
                "min": _format_value(values["min"]),
                "max": _format_value(values["max"]),
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
                    **{metric: _format_value(float(row[metric])) for metric in available_metrics},
                }
                for row in result.rows
            ]
            ui.table(
                columns=sample_columns,
                rows=sample_rows,
                row_key="sample_id",
                pagination=10,
            ).props("flat bordered dense").classes("w-full")


def _render_warnings(warnings: tuple[str, ...], container) -> None:
    container.clear()
    with container:
        for warning in warnings:
            with ui.row().classes("w-full items-start gap-2 text-amber-800"):
                ui.icon("warning_amber", size="xs")
                ui.label(warning).classes("text-base")


def _render_plots(paths: tuple[Path, ...], container) -> None:
    container.clear()
    with container:
        ui.label("Key metric distributions").classes("text-xl font-semibold")
        ui.label(
            "SSIM, PSNR, MAE, and RGB correlation provide complementary views; "
            "all metrics remain available in the table and CSV."
        ).classes("text-sm text-slate-500")
        selected_paths = _select_evaluation_plot_paths(paths)
        if not selected_paths:
            empty_state("insert_chart", "No plots available", "The numeric summary is still valid.")
            return
        with ui.row().classes("w-full grid grid-cols-1 lg:grid-cols-2 gap-4"):
            for path in selected_paths:
                with ui.column().classes("vs-subtle rounded-lg p-3 gap-2 min-w-0"):
                    ui.label(path.stem.replace("_", " ").title()).classes("text-base font-medium")
                    ui.image(path).props("fit=contain").classes("w-full rounded")


def _select_evaluation_plot_paths(paths: tuple[Path, ...]) -> tuple[Path, ...]:
    """Keep the evaluation dashboard concise while preserving all saved artefacts."""
    by_name = {path.name.lower(): path for path in paths}
    selected = tuple(by_name[name] for name in EVALUATION_PLOT_PRIORITY if name in by_name)
    if selected:
        return selected
    return tuple(
        path
        for path in paths
        if path.name.lower() not in {"metrics_boxplot.png", "pcc_gray_histogram.png"}
    )[:4]


def _representative_card(sample: RepresentativeSample) -> None:
    with ui.column().classes("vs-subtle rounded-xl p-3 gap-2 min-w-0"):
        with ui.row().classes("w-full justify-between items-baseline gap-2"):
            ui.badge(sample.kind.upper()).props("outline color=teal-8")
            ui.label(_format_value(sample.value)).classes("font-mono text-base")
        ui.label(sample.sample_id).classes("text-sm text-slate-500 truncate")
        if sample.comparison_path is not None:
            ui.image(sample.comparison_path).props("fit=contain").classes(
                "w-full rounded-lg border bg-white"
            )
            ui.label("Source · generated · target · MAE map").classes("text-sm text-slate-500")
        elif sample.generated_path is not None:
            preview = _browser_image_source(sample.generated_path)
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


def _browser_image_source(path: Path) -> Path | Image.Image | None:
    """Return a browser-compatible source, converting TIFF previews in memory."""
    if path.suffix.lower() not in {".tif", ".tiff"}:
        return path
    try:
        with Image.open(path) as image:
            return image.convert("RGB").copy()
    except (OSError, ValueError):
        logger.warning("Could not create browser preview for %s", path, exc_info=True)
        return None


def _render_comparison(
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
            _metric_card(key, value)
    plots_container.clear()
    with plots_container:
        ui.label("Statistical plots").classes("text-xl font-semibold")
        with ui.row().classes("w-full grid grid-cols-1 lg:grid-cols-2 gap-4"):
            for path in result.plot_paths:
                with ui.column().classes("vs-subtle rounded-lg p-3 gap-2 min-w-0"):
                    ui.label(path.stem.replace("_", " ").title()).classes("text-base font-medium")
                    ui.image(path).props("fit=contain").classes("w-full rounded")
    qualitative_container.clear()
    with qualitative_container:
        ui.label("Representative generated samples").classes("text-xl font-semibold")
        ui.label(
            "Best, median, and worst are selected independently within each run for this metric."
        ).classes("text-sm text-slate-500 -mt-3")
        with ui.row().classes("w-full grid grid-cols-1 xl:grid-cols-2 gap-5"):
            for label, samples in (
                (result.label_a, result.representatives_a),
                (result.label_b, result.representatives_b),
            ):
                with ui.column().classes("gap-3 min-w-0"):
                    ui.label(label).classes("font-semibold")
                    with ui.row().classes("w-full grid grid-cols-1 sm:grid-cols-3 gap-3"):
                        for sample in samples:
                            _representative_card(sample)


def _run_label(run_descriptor: RunDescriptor) -> str:
    count = f" · {run_descriptor.sample_count} samples" if run_descriptor.sample_count else ""
    return f"{run_descriptor.display_name}{count}"


def _format_value(value: float | int | str) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "n/a"
        if math.isinf(value):
            return "∞"
        if abs(value) >= 1000 or (0 < abs(value) < 0.001):
            return f"{value:.3g}"
        return f"{value:.4f}"
    return str(value).replace("_", " ")
