from __future__ import annotations

import logging
from dataclasses import replace
from io import BytesIO
from pathlib import Path

from nicegui import events, run, ui
from PIL import Image

from virtual_staining.applications.ui_inference import (
    ModelCatalog,
    ModelDescriptor,
    ResultProvenance,
    UIInferenceError,
    UIInferenceResult,
    UIInferenceService,
)

logger = logging.getLogger(__name__)

_PAGE_CSS = """
body { background: #f8fafc; color: #0f172a; }
.research-card { border: 1px solid #e2e8f0; box-shadow: 0 1px 3px rgb(15 23 42 / 0.05); }
.preview-image { position: absolute !important; inset: 0; width: 100%; height: 100% !important; }
.preview-image .q-img__image { object-fit: contain !important; }
.q-uploader { box-shadow: none !important; border: 1px dashed #94a3b8; border-radius: 0.75rem; }
"""


def _descriptor_details(descriptor: ModelDescriptor) -> str:
    width, height = descriptor.image_size
    return (
        f"**Transformation:** {descriptor.transformation}  \n"
        f"**Checkpoint:** `{descriptor.checkpoint_filename}`  \n"
        f"**Identifier:** `{descriptor.identifier}`  \n"
        f"**Checkpoint schema:** v{descriptor.checkpoint_schema_version}  \n"
        f"**Architecture:** {descriptor.architecture_id} ({descriptor.model_class})  \n"
        f"**Required input:** {width} × {height} px, "
        f"{descriptor.channels_per_input} channels"
    )


def _provenance_details(provenance: ResultProvenance) -> str:
    width, height = provenance.required_image_size
    return (
        f"**Transformation:** {provenance.transformation}  \n"
        f"**Model identifier:** `{provenance.model_identifier}`  \n"
        f"**Checkpoint:** `{provenance.checkpoint_filename}` "
        f"(schema v{provenance.checkpoint_schema_version})  \n"
        f"**Source / target:** {' + '.join(provenance.source_domains)} → "
        f"{provenance.target_domain}  \n"
        f"**Architecture:** {provenance.architecture_id} ({provenance.model_class})  \n"
        f"**Required input:** {width} × {height} px, "
        f"{provenance.required_input_channels} channels  \n"
        f"**Uploaded input:** `{provenance.source_filename}` "
        f"({provenance.input_image_mode}, "
        f"{provenance.input_image_size[0]} × {provenance.input_image_size[1]} px)  \n"
        f"**Generated output:** `{provenance.generated_filename}`  \n"
        f"**Runtime device:** {provenance.runtime_device}"
    )


def _build_page(
    service: UIInferenceService,
    catalog: ModelCatalog,
    output_directory: Path,
) -> None:
    selected_image: Image.Image | None = None
    selected_filename: str | None = None
    inference_result: UIInferenceResult | None = None
    models = {descriptor.identifier: descriptor for descriptor in catalog.models}
    initial_model_id = catalog.models[0].identifier if catalog.models else None

    ui.page_title("Virtual Staining")
    ui.add_css(_PAGE_CSS)

    with ui.column().classes("w-full min-h-screen items-center"):
        content = ui.column().classes("w-full max-w-6xl px-4 sm:px-6 py-6 md:py-10 gap-6")
        with content:
            with ui.row().classes("w-full items-center justify-between gap-3"):
                with ui.row().classes("items-center gap-3"):
                    ui.icon("biotech", size="md").classes("text-slate-700")
                    ui.label("Virtual Staining").classes(
                        "text-2xl md:text-3xl font-semibold tracking-tight"
                    )
                ui.badge("Research prototype").props("outline color=blue-grey-7").classes(
                    "text-xs px-2 py-1"
                )
            ui.label(
                "Single-patch inference using trained Virtual-Staining models and "
                "validated checkpoint metadata."
            ).classes("text-sm md:text-base text-slate-600 -mt-3")

            with ui.card().classes("research-card w-full rounded-xl p-5 md:p-6 gap-4"):
                with ui.row().classes("w-full items-center gap-3"):
                    ui.badge("1").props("rounded color=blue-grey-8")
                    with ui.column().classes("gap-0"):
                        ui.label("Model").classes("text-lg font-semibold")
                        ui.label("Choose a validated checkpoint for this transformation.").classes(
                            "text-sm text-slate-500"
                        )

                if catalog.models:
                    model_select = ui.select(
                        {item.identifier: item.display_name for item in catalog.models},
                        value=initial_model_id,
                        label="Available model",
                    ).classes("w-full")
                    with ui.row().classes("w-full gap-2 flex-wrap"):
                        transformation_chip = ui.badge().props("color=blue-grey-7")
                        size_chip = ui.badge().props("outline color=blue-grey-7")
                        channel_chip = ui.badge().props("outline color=blue-grey-7")
                        schema_chip = ui.badge().props("outline color=blue-grey-7")
                    model_details = ui.markdown().classes("text-sm text-slate-600")
                else:
                    model_select = ui.select({}, label="Available model").classes("w-full")
                    model_select.disable()
                    transformation_chip = ui.badge()
                    size_chip = ui.badge()
                    channel_chip = ui.badge()
                    schema_chip = ui.badge()
                    model_details = ui.markdown()
                    for element in (
                        transformation_chip,
                        size_chip,
                        channel_chip,
                        schema_chip,
                        model_details,
                    ):
                        element.set_visibility(False)
                    with ui.row().classes(
                        "w-full rounded-lg border border-dashed border-slate-300 "
                        "bg-slate-50 p-5 items-start gap-3"
                    ):
                        ui.icon("inventory_2", size="sm").classes("text-slate-400")
                        with ui.column().classes("gap-1"):
                            ui.label("No compatible checkpoints found").classes("font-medium")
                            directory_state = (
                                "The configured checkpoint directory does not exist."
                                if not catalog.checkpoint_directory_exists
                                else "The directory contains no current v3 RGB single-input models."
                            )
                            ui.label(
                                f"{directory_state} Start the UI with --checkpoint-dir PATH "
                                "or set VIRTUAL_STAINING_CHECKPOINT_DIR."
                            ).classes("text-sm text-slate-500")

                if catalog.issues:
                    with ui.expansion(
                        f"{len(catalog.issues)} checkpoint(s) skipped",
                        icon="warning_amber",
                    ).classes("w-full text-sm text-amber-800"):
                        for issue in catalog.issues:
                            ui.label(f"{issue.checkpoint}: {issue.reason}").classes(
                                "text-xs text-slate-600"
                            )

            with ui.card().classes("research-card w-full rounded-xl p-5 md:p-6 gap-4"):
                with ui.row().classes("w-full items-center gap-3"):
                    ui.badge("2").props("rounded color=blue-grey-8")
                    with ui.column().classes("gap-0"):
                        ui.label("Input").classes("text-lg font-semibold")
                        ui.label(
                            "Upload one exact-size RGB patch. No resizing is applied."
                        ).classes("text-sm text-slate-500")
                uploader = (
                    ui.upload(
                        label="Drop an image here or browse",
                        auto_upload=True,
                        max_files=1,
                        max_file_size=50 * 1024 * 1024,
                    )
                    .props('accept=".bmp,.jpeg,.jpg,.png,.tif,.tiff" flat bordered no-thumbnails')
                    .classes("w-full")
                )
                if not catalog.models:
                    uploader.disable()
                input_status = ui.label(
                    "Select a model, then upload a compatible image patch."
                ).classes("text-sm text-slate-500")

            with ui.card().classes("research-card w-full rounded-xl p-5 md:p-6 gap-5"):
                with ui.row().classes("w-full items-center gap-3"):
                    ui.badge("3").props("rounded color=blue-grey-8")
                    with ui.column().classes("gap-0"):
                        ui.label("Result").classes("text-lg font-semibold")
                        result_status = ui.label(
                            "A generated comparison will appear after inference."
                        ).classes("text-sm text-slate-500")

                with ui.row().classes("w-full flex-col md:flex-row gap-5 items-stretch"):
                    with ui.column().classes("flex-1 min-w-0 gap-2"):
                        ui.label("Source").classes("text-sm font-medium text-slate-700")
                        with ui.element("div").classes(
                            "relative w-full aspect-square rounded-lg border border-slate-200 "
                            "bg-slate-50 overflow-hidden"
                        ):
                            input_empty = ui.label("No input selected").classes(
                                "absolute inset-0 flex items-center justify-center "
                                "text-sm text-slate-400"
                            )
                            input_preview = (
                                ui.image()
                                .props("fit=contain")
                                .classes("preview-image absolute inset-0 w-full h-full")
                            )
                            input_preview.set_visibility(False)
                    with ui.column().classes("flex-1 min-w-0 gap-2"):
                        ui.label("Generated").classes("text-sm font-medium text-slate-700")
                        with ui.element("div").classes(
                            "relative w-full aspect-square rounded-lg border border-slate-200 "
                            "bg-slate-50 overflow-hidden"
                        ):
                            output_empty = ui.label("No result generated").classes(
                                "absolute inset-0 flex items-center justify-center "
                                "text-sm text-slate-400"
                            )
                            output_preview = (
                                ui.image()
                                .props("fit=contain")
                                .classes("preview-image absolute inset-0 w-full h-full")
                            )
                            output_preview.set_visibility(False)

                output_folder = ui.input(
                    label="Output folder",
                    value=str(output_directory),
                ).classes("w-full")
                ui.label("Relative output paths are resolved from the launch directory.").classes(
                    "text-xs text-slate-500 -mt-3"
                )
                with ui.row().classes("w-full gap-3 flex-wrap"):
                    generate_button = ui.button("Generate", icon="auto_awesome").props(
                        "color=primary unelevated"
                    )
                    save_button = ui.button("Save result", icon="save").props(
                        "outline color=blue-grey-8"
                    )
                    reset_button = ui.button("New image", icon="refresh").props(
                        "flat color=blue-grey-8"
                    )
                    generate_button.disable()
                    save_button.disable()
                    reset_button.disable()

                with ui.expansion(
                    "Model & provenance details",
                    icon="description",
                ).classes("w-full border-t border-slate-100 pt-2 text-sm"):
                    provenance_details = ui.markdown(
                        "Generate an image to record result provenance."
                    ).classes("text-sm text-slate-600")

            ui.separator().classes("mt-2")
            ui.label(
                "Research prototype only — not a clinically validated diagnostic system."
            ).classes("w-full text-center text-xs text-slate-500 pb-4")

    def selected_descriptor() -> ModelDescriptor | None:
        value = model_select.value
        return models.get(str(value)) if value is not None else None

    def show_descriptor(descriptor: ModelDescriptor | None) -> None:
        if descriptor is None:
            return
        width, height = descriptor.image_size
        transformation_chip.set_text(descriptor.transformation)
        size_chip.set_text(f"{width} × {height} px")
        channel_chip.set_text(f"{descriptor.channels_per_input} channels")
        schema_chip.set_text(f"schema v{descriptor.checkpoint_schema_version}")
        model_details.set_content(_descriptor_details(descriptor))

    def clear_input() -> None:
        nonlocal selected_image, selected_filename, inference_result
        selected_image = None
        selected_filename = None
        inference_result = None
        uploader.reset()
        input_preview.set_source("")
        input_preview.set_visibility(False)
        input_empty.set_visibility(True)
        output_preview.set_source("")
        output_preview.set_visibility(False)
        output_empty.set_visibility(True)
        input_status.set_text("Upload a compatible exact-size RGB image patch.")
        result_status.set_text("A generated comparison will appear after inference.")
        provenance_details.set_content("Generate an image to record result provenance.")
        generate_button.disable()
        save_button.disable()
        reset_button.disable()

    def handle_model_change() -> None:
        clear_input()
        show_descriptor(selected_descriptor())

    async def handle_upload(event: events.UploadEventArguments) -> None:
        nonlocal selected_image, selected_filename, inference_result
        descriptor = selected_descriptor()
        if descriptor is None:
            ui.notify("Select an available model before uploading an image.", type="warning")
            return
        inference_result = None
        generate_button.disable()
        save_button.disable()
        output_preview.set_visibility(False)
        output_empty.set_visibility(True)
        try:
            with Image.open(BytesIO(await event.file.read())) as uploaded:
                uploaded.load()
                candidate = uploaded.copy()
            service.validate_input(descriptor.identifier, candidate)
        except UIInferenceError as exc:
            ui.notify(str(exc), type="negative", multi_line=True, close_button=True)
            clear_input()
            return
        except OSError:
            ui.notify(
                "The uploaded file could not be read as a supported image.",
                type="negative",
                close_button=True,
            )
            clear_input()
            return
        except Exception:
            logger.exception("Unexpected uploaded-image processing failure")
            ui.notify("The image could not be processed.", type="negative", close_button=True)
            clear_input()
            return

        selected_image = candidate
        selected_filename = event.file.name
        input_preview.set_source(candidate)
        input_preview.set_visibility(True)
        input_empty.set_visibility(False)
        input_status.set_text(
            f"Ready · {candidate.mode} · {candidate.size[0]} × {candidate.size[1]} px"
        )
        result_status.set_text("Input validated. Ready to generate.")
        generate_button.enable()
        reset_button.enable()

    async def handle_generate() -> None:
        nonlocal inference_result
        descriptor = selected_descriptor()
        if selected_image is None or selected_filename is None or descriptor is None:
            ui.notify("Select a valid input image first.", type="warning")
            return
        generate_button.disable()
        save_button.disable()
        generate_button.props("loading")
        result_status.set_text("Generating the virtual stain…")
        try:
            inference_result = await run.io_bound(
                service.run_inference,
                descriptor.identifier,
                selected_image,
                selected_filename,
            )
        except UIInferenceError as exc:
            inference_result = None
            result_status.set_text("Inference failed. Review the message and try again.")
            ui.notify(str(exc), type="negative", multi_line=True, close_button=True)
        except Exception:
            logger.exception("Unexpected UI inference failure")
            inference_result = None
            result_status.set_text("Inference failed.")
            ui.notify("An unexpected inference error occurred.", type="negative", close_button=True)
        else:
            assert inference_result is not None
            output_preview.set_source(inference_result.generated_image)
            output_preview.set_visibility(True)
            output_empty.set_visibility(False)
            provenance_details.set_content(_provenance_details(inference_result.provenance))
            result_status.set_text("Inference complete. Review the comparison before saving.")
            save_button.enable()
            ui.notify("Virtual stain generated.", type="positive")
        finally:
            generate_button.props(remove="loading")
            generate_button.enable()

    def handle_save() -> None:
        nonlocal inference_result
        if inference_result is None:
            ui.notify("Generate an image before saving.", type="warning")
            return
        try:
            saved = service.save_result(inference_result, str(output_folder.value or ""))
        except UIInferenceError as exc:
            ui.notify(str(exc), type="negative", multi_line=True, close_button=True)
            return
        inference_result = replace(inference_result, provenance=saved.provenance)
        provenance_details.set_content(_provenance_details(saved.provenance))
        ui.notify(
            f"Saved {saved.image_path.name} with {saved.sidecar_path.name}.",
            type="positive",
        )

    if catalog.models:
        show_descriptor(catalog.models[0])
    model_select.on_value_change(lambda _event: handle_model_change())
    uploader.on_upload(handle_upload)
    generate_button.on_click(handle_generate)
    save_button.on_click(handle_save)
    reset_button.on_click(clear_input)


def run_ui(
    checkpoint_directory: Path,
    output_directory: Path,
    *,
    host: str = "0.0.0.0",
    port: int = 8080,
) -> None:
    """Configure and launch the NiceGUI application."""
    service = UIInferenceService(checkpoint_directory, output_directory)
    catalog = service.discover_models()

    @ui.page("/")
    def index() -> None:
        _build_page(service, catalog, output_directory)

    ui.run(title="Virtual Staining", host=host, port=port, reload=False, show=False)
