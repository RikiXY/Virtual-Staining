from __future__ import annotations

from nicegui import ui

from virtual_staining.applications.api import (
    ApplicationError,
    ApplicationService,
    TrainingConfigDocument,
    TrainingConfigDraft,
)
from virtual_staining.ui.theme import section_heading


def build_training_config_page(service: ApplicationService) -> None:
    """Build the focused editor for a minimal, training-only run YAML."""
    current_document: TrainingConfigDocument | None = None

    with ui.column().classes("w-full gap-5"):
        with (
            ui.element("section").classes(
                "vs-config-hero w-full rounded-2xl px-5 py-6 md:px-8 md:py-8"
            ),
            ui.row().classes("w-full items-start justify-between gap-4 flex-wrap"),
        ):
            with ui.column().classes("gap-2 max-w-3xl"):
                ui.badge("TRAINING YAML · PROTOTYPE").props("outline color=teal-8")
                ui.label("Configure a training run").classes(
                    "text-3xl md:text-4xl font-semibold tracking-tight"
                )
                ui.label(
                    "Fill in the essentials and get a validated YAML file ready for the "
                    "training command. Optional settings keep their tested defaults."
                ).classes("text-slate-600 text-lg leading-relaxed")
            with ui.element("div").classes("vs-config-hero-icon"):
                ui.icon("tune", size="lg").classes("text-teal-700")

        with ui.element("div").classes("vs-config-layout w-full gap-5 items-start"):
            with ui.column().classes("w-full gap-5"):
                with ui.card().classes("vs-card vs-config-step w-full rounded-xl p-5 md:p-6 gap-5"):
                    _step_heading(
                        "1",
                        "Experiment",
                        "Name the run and point it at an already prepared dataset.",
                    )
                    run_name = ui.input(
                        label="Run name *",
                        value="my_training_run",
                        placeholder="e.g. lf_to_he_baseline",
                    ).classes("w-full vs-config-field")
                    with ui.row().classes("w-full flex-col md:flex-row gap-4 items-start"):
                        dataset_root = ui.input(
                            label="Prepared dataset root *",
                            value="local_workspace/datasets/your_sample",
                            placeholder="Path containing manifests/ and splits/",
                        ).classes("w-full md:flex-1 vs-config-field")
                        results_path = ui.input(
                            label="Results directory *",
                            value="local_workspace/results",
                            placeholder="Where run artifacts will be written",
                        ).classes("w-full md:flex-1 vs-config-field")
                    ui.label(
                        "This prototype references prepared data; it does not create or "
                        "modify a dataset."
                    ).classes("text-sm text-slate-500")

                with ui.card().classes("vs-card vs-config-step w-full rounded-xl p-5 md:p-6 gap-5"):
                    _step_heading(
                        "2",
                        "Modalities",
                        "Describe the model inputs and the stain it should learn to generate.",
                    )
                    with ui.row().classes("w-full flex-col md:flex-row gap-4 items-start"):
                        input_modalities = ui.input(
                            label="Input modalities *",
                            value="label_free",
                            placeholder="Comma-separated, e.g. autofluorescence, label_free",
                        ).classes("w-full md:flex-1 vs-config-field")
                        target_modality = ui.input(
                            label="Target modality *",
                            value="HE",
                            placeholder="e.g. HE",
                        ).classes("w-full md:flex-1 vs-config-field")
                    ui.label(
                        "Multiple inputs are accepted as comma-separated names and kept in "
                        "this order."
                    ).classes("text-sm text-slate-500")

                with ui.card().classes("vs-card vs-config-step w-full rounded-xl p-5 md:p-6 gap-5"):
                    _step_heading(
                        "3",
                        "Training objective",
                        "Set the duration and the three weights required by the standard "
                        "objective.",
                    )
                    epochs = (
                        ui.number(label="Epochs *", value=100, min=1, step=1, precision=0)
                        .props("outlined")
                        .classes("w-full vs-config-field")
                    )
                    with ui.row().classes("w-full flex-col md:flex-row gap-4 items-start"):
                        generator_adversarial_weight = ui.number(
                            label="Generator adversarial weight *", value=1.0, min=0, step=0.1
                        ).classes("w-full md:flex-1 vs-config-field")
                        reconstruction_weight = ui.number(
                            label="L1 reconstruction weight *", value=25.0, min=0, step=1
                        ).classes("w-full md:flex-1 vs-config-field")
                        discriminator_adversarial_weight = ui.number(
                            label="Discriminator weight *", value=1.0, min=0, step=0.1
                        ).classes("w-full md:flex-1 vs-config-field")
                    with ui.row().classes(
                        "w-full items-start gap-2 vs-default-note rounded-lg p-3"
                    ):
                        ui.icon("info", size="xs").classes("text-sky-700 mt-1")
                        ui.label(
                            "Batch size, learning rates, optimizer values, checkpoint cadence, and "
                            "augmentation are intentionally omitted and use application defaults."
                        ).classes("text-sm text-slate-600 flex-1")

            with ui.column().classes("vs-config-preview-column w-full gap-4"):
                with ui.card().classes("vs-card w-full rounded-xl p-0 gap-0 overflow-hidden"):
                    with ui.row().classes(
                        "w-full items-center justify-between gap-3 px-5 py-4 border-b "
                        "border-slate-200"
                    ):
                        with ui.row().classes("items-center gap-2"):
                            ui.icon("description", size="sm").classes("text-teal-700")
                            ui.label("YAML preview").classes("font-semibold text-slate-800")
                        validation_badge = ui.badge("Valid").props("color=positive")
                    preview = ui.code("", language="yaml").classes(
                        "vs-yaml-preview w-full rounded-none"
                    )
                    with ui.column().classes("w-full gap-1 px-5 py-4 border-t border-slate-200"):
                        filename_label = ui.label().classes("font-medium text-slate-700")
                        validation_message = ui.label().classes("text-sm text-slate-500")

                with ui.card().classes("vs-card w-full rounded-xl p-5 gap-4"):
                    section_heading(
                        "rocket_launch",
                        "Ready to use",
                        "Download locally or save into the configured YAML directory.",
                    )
                    ui.label(f"Server directory: {service.training_config_directory}").classes(
                        "text-sm text-slate-500 break-all"
                    )
                    with ui.row().classes("w-full gap-3 flex-wrap"):
                        download_button = ui.button("Download YAML", icon="download").props(
                            "color=primary unelevated no-caps"
                        )
                        save_button = ui.button("Save on server", icon="save").props(
                            "outline color=primary no-caps"
                        )
                    saved_status = ui.label().classes("text-sm text-teal-700")
                    with ui.element("div").classes("vs-command-box w-full rounded-lg p-3"):
                        command_label = ui.label().classes(
                            "font-mono text-sm text-slate-700 break-all"
                        )

    controls = (
        run_name,
        dataset_root,
        results_path,
        input_modalities,
        target_modality,
        epochs,
        generator_adversarial_weight,
        reconstruction_weight,
        discriminator_adversarial_weight,
    )

    def make_draft() -> TrainingConfigDraft:
        modalities = tuple(
            item.strip() for item in str(input_modalities.value or "").split(",") if item.strip()
        )
        return TrainingConfigDraft(
            run_name=str(run_name.value or ""),
            dataset_root=str(dataset_root.value or ""),
            results_path=str(results_path.value or ""),
            input_modalities=modalities,
            target_modality=str(target_modality.value or ""),
            epochs=int(epochs.value or 0),
            generator_adversarial_weight=float(generator_adversarial_weight.value or 0),
            reconstruction_weight=float(reconstruction_weight.value or 0),
            discriminator_adversarial_weight=float(discriminator_adversarial_weight.value or 0),
        )

    def refresh_preview() -> None:
        nonlocal current_document
        saved_status.set_text("")
        try:
            current_document = service.preview_training_config(make_draft())
        except ApplicationError as exc:
            current_document = None
            validation_badge.set_text("Needs attention")
            validation_badge.props("color=negative")
            validation_message.set_text(str(exc))
            filename_label.set_text("YAML is not ready")
            preview.set_content("# Fix the values to generate a valid training YAML.")
            command_label.set_text("uv run vs train --config <training-config.yaml>")
            download_button.disable()
            save_button.disable()
            return
        validation_badge.set_text("Valid")
        validation_badge.props("color=positive")
        validation_message.set_text("Validated with the same schema used by the training CLI.")
        filename_label.set_text(current_document.filename)
        preview.set_content(current_document.yaml_text)
        command_label.set_text(f"uv run vs train --config {current_document.filename}")
        download_button.enable()
        save_button.enable()

    def download_yaml() -> None:
        refresh_preview()
        if current_document is not None:
            ui.download(
                current_document.yaml_text.encode("utf-8"),
                filename=current_document.filename,
                media_type="application/yaml",
            )

    def save_yaml() -> None:
        refresh_preview()
        if current_document is None:
            return
        try:
            saved = service.save_training_config(make_draft())
        except ApplicationError as exc:
            ui.notify(str(exc), type="negative")
            return
        saved_status.set_text(f"Saved without overwriting: {saved.path}")
        command_label.set_text(f"uv run vs train --config {saved.path}")
        ui.notify(f"Saved {saved.path.name}", type="positive")

    for control in controls:
        control.on("update:model-value", lambda _event: refresh_preview())
    download_button.on_click(download_yaml)
    save_button.on_click(save_yaml)
    refresh_preview()


def _step_heading(number: str, title: str, description: str) -> None:
    with ui.row().classes("w-full items-start gap-3"):
        ui.badge(number).props("rounded color=teal-8").classes("vs-step-number")
        with ui.column().classes("gap-0 min-w-0"):
            ui.label(title).classes("text-xl font-semibold text-slate-800")
            ui.label(description).classes("text-base text-slate-500")
