from __future__ import annotations

import logging
from io import BytesIO

from nicegui import events, run, ui
from PIL import Image

from virtual_staining.applications.api import (
    ApplicationError,
    ApplicationService,
    GeneratedSampleEvaluationRequest,
    InferenceRequest,
    InferenceResult,
    ModelCatalog,
    SingleSampleResult,
)
from virtual_staining.ui.experiment_components import (
    metric_card,
    result_image,
    result_placeholder,
)
from virtual_staining.ui.image_comparison import render_image_comparison
from virtual_staining.ui.theme import empty_state, section_heading

logger = logging.getLogger(__name__)


def build_single_sample(service: ApplicationService, catalog: ModelCatalog) -> None:
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
                render_pending_generation(
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
                render_pending_generation(
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
            render_pending_generation(
                completed,
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
            render_single_result(result, images_container, overlap_container, metrics_container)
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


def render_pending_generation(
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
            result_image(label, image, note)
        if target is None:
            result_placeholder("Target", "Upload a target to compare")
        else:
            result_image("Target", target, "Ground truth")
        result_placeholder("Absolute difference", "Available after evaluation")
    if target is not None:
        render_image_comparison(inference.generated_image, target, overlap)
    with metrics:
        empty_state(
            "calculate",
            "Generation complete",
            "Upload a target and choose Evaluate against target to calculate metrics.",
        )


def render_single_result(result: SingleSampleResult, images, overlap, metrics) -> None:
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
            result_image(label, image, note)
    render_image_comparison(result.inference.generated_image, result.target_image, overlap)
    with metrics:
        for metric in ("ssim", "psnr", "mae", "rmse", "mse", "pcc_rgb_mean", "pcc_gray"):
            metric_card(metric, result.metrics[metric])
