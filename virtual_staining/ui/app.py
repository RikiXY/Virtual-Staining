from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import torch
from nicegui import events, ui
from PIL import Image

from virtual_staining.inference.runner import (
    LoadedCheckpointGenerator,
    load_checkpoint_generator,
    resolve_inference_device,
)
from virtual_staining.inference.single import predict_single_patch, validate_patch_image

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CHECKPOINT_DIR = _PROJECT_ROOT / "local_workspace" / "ui" / "checkpoints"


@dataclass(frozen=True)
class Transformation:
    checkpoint_path: Path
    input_modality: str
    target_modality: str


TRANSFORMATIONS = {
    "Label-Free → H&E": Transformation(
        checkpoint_path=_CHECKPOINT_DIR / "lf-to-he-v1.pth",
        input_modality="label_free",
        target_modality="H&E",
    ),
    "H&E → Label-Free-like": Transformation(
        checkpoint_path=_CHECKPOINT_DIR / "he-to-lf-v1.pth",
        input_modality="H&E",
        target_modality="label_free",
    ),
}


def _save_generated_image(
    image: Image.Image,
    input_filename: str,
    output_folder: str,
) -> Path:
    if not output_folder.strip():
        raise ValueError("Output folder must not be empty.")

    output_dir = Path(output_folder).expanduser()
    if not output_dir.is_absolute():
        output_dir = _PROJECT_ROOT / output_dir
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(f"Output folder is not a directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_filename = Path(input_filename.replace("\\", "/")).name
    input_stem = Path(safe_filename).stem or "image"
    output_path = output_dir / f"{input_stem}_generated.png"
    index = 2
    while output_path.exists():
        output_path = output_dir / f"{input_stem}_generated_{index}.png"
        index += 1

    image.save(output_path, format="PNG")
    return output_path


def _load_transformation(
    transformation: Transformation,
    device: torch.device,
) -> LoadedCheckpointGenerator:
    checkpoint_path = transformation.checkpoint_path
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            "No local pretrained checkpoint was found for this transformation. "
            f"Expected: {checkpoint_path}"
        )

    runtime = load_checkpoint_generator(checkpoint_path, device)
    if runtime.input_names != (transformation.input_modality,):
        raise ValueError(
            "Checkpoint input metadata does not match the selected transformation. "
            f"Expected {(transformation.input_modality,)}, got {runtime.input_names}."
        )
    if runtime.target_modality != transformation.target_modality:
        raise ValueError(
            "Checkpoint target metadata does not match the selected transformation. "
            f"Expected {transformation.target_modality!r}, got {runtime.target_modality!r}."
        )
    if runtime.channels_per_input != 3:
        raise ValueError(
            "This GUI version supports RGB checkpoints only. "
            f"The selected checkpoint expects {runtime.channels_per_input} channels."
        )
    return runtime


def _build_page() -> None:
    selected_image: Image.Image | None = None
    selected_filename: str | None = None
    generated_image: Image.Image | None = None
    runtime: LoadedCheckpointGenerator | None = None

    ui.page_title("Virtual Staining")
    with ui.column().classes("w-full max-w-5xl mx-auto p-6 gap-5"):
        ui.label("Virtual Staining").classes("text-3xl font-semibold")
        ui.label(
            "Pretrained mode reconstructs the generator from validated checkpoint metadata. "
            "It does not use or cross-check a YAML configuration."
        ).classes("text-sm text-gray-600")

        transformation_select = ui.select(
            list(TRANSFORMATIONS),
            value=next(iter(TRANSFORMATIONS)),
            label="Transformation",
        ).classes("w-full max-w-md")
        checkpoint_status = ui.label(
            "The local checkpoint will be validated when an image is selected."
        ).classes("text-sm text-gray-600")

        ui.label("Input").classes("text-xl font-medium mt-2")
        uploader = (
            ui.upload(
                label="Select one patch-sized image",
                auto_upload=True,
                max_files=1,
                max_file_size=50 * 1024 * 1024,
            )
            .props('accept=".bmp,.jpeg,.jpg,.png,.tif,.tiff"')
            .classes("w-full max-w-md")
        )

        with ui.row().classes("w-full gap-8 items-start"):
            with ui.column().classes("flex-1 min-w-64"):
                ui.label("Input").classes("font-medium")
                input_preview = ui.image().classes("w-full border rounded")
                input_preview.set_visibility(False)
            with ui.column().classes("flex-1 min-w-64"):
                ui.label("Generated").classes("font-medium")
                output_preview = ui.image().classes("w-full border rounded")
                output_preview.set_visibility(False)

        output_folder = ui.input(
            label="Output folder",
            value="local_workspace/ui/outputs",
        ).classes("w-full max-w-xl")
        ui.label("Relative paths are resolved from the repository root.").classes(
            "text-sm text-gray-600"
        )
        with ui.row().classes("gap-3"):
            generate_button = ui.button("Generate").props("color=primary")
            save_button = ui.button("Save")
            generate_button.disable()
            save_button.disable()

    def reset_input() -> None:
        nonlocal selected_image, selected_filename, generated_image, runtime
        selected_image = None
        selected_filename = None
        generated_image = None
        runtime = None
        uploader.reset()
        input_preview.set_source("")
        input_preview.set_visibility(False)
        output_preview.set_source("")
        output_preview.set_visibility(False)
        generate_button.disable()
        save_button.disable()
        checkpoint_status.set_text(
            "The local checkpoint will be validated when an image is selected."
        )

    async def handle_upload(event: events.UploadEventArguments) -> None:
        nonlocal selected_image, selected_filename, generated_image, runtime
        generated_image = None
        generate_button.disable()
        save_button.disable()
        output_preview.set_visibility(False)
        try:
            transformation = TRANSFORMATIONS[str(transformation_select.value)]
            loaded_runtime = _load_transformation(transformation, resolve_inference_device())
            with Image.open(BytesIO(await event.file.read())) as uploaded:
                uploaded.load()
                candidate = uploaded.copy()
            validate_patch_image(candidate, loaded_runtime.image_size)
        except (OSError, RuntimeError, ValueError, KeyError) as exc:
            selected_image = None
            selected_filename = None
            runtime = None
            uploader.reset()
            input_preview.set_source("")
            input_preview.set_visibility(False)
            ui.notify(str(exc), type="negative", multi_line=True, close_button=True)
            return

        selected_image = candidate
        selected_filename = event.file.name
        runtime = loaded_runtime
        width, height = loaded_runtime.image_size
        checkpoint_status.set_text(
            "Checkpoint validated · "
            f"expected input {width} × {height} px · device {loaded_runtime.device}"
        )
        input_preview.set_source(selected_image)
        input_preview.set_visibility(True)
        generate_button.enable()

    def handle_generate() -> None:
        nonlocal generated_image
        if selected_image is None or runtime is None:
            ui.notify("Select a valid input image first.", type="warning")
            return
        generate_button.disable()
        save_button.disable()
        try:
            generated = predict_single_patch(runtime, selected_image)
        except (RuntimeError, ValueError) as exc:
            generated_image = None
            ui.notify(str(exc), type="negative", multi_line=True, close_button=True)
        else:
            generated_image = generated
            output_preview.set_source(generated)
            output_preview.set_visibility(True)
            save_button.enable()
        finally:
            generate_button.enable()

    def handle_save() -> None:
        if generated_image is None or selected_filename is None:
            ui.notify("Generate an image before saving.", type="warning")
            return
        try:
            output_path = _save_generated_image(
                generated_image,
                selected_filename,
                str(output_folder.value or ""),
            )
        except (OSError, ValueError) as exc:
            ui.notify(str(exc), type="negative", multi_line=True, close_button=True)
        else:
            ui.notify(f"Saved to {output_path}", type="positive", multi_line=True)

    transformation_select.on_value_change(lambda _event: reset_input())
    uploader.on_upload(handle_upload)
    generate_button.on_click(handle_generate)
    save_button.on_click(handle_save)


def main() -> None:
    @ui.page("/")
    def index() -> None:
        _build_page()

    ui.run(title="Virtual Staining", reload=False, show=False)


if __name__ in {"__main__", "__mp_main__"}:
    main()
