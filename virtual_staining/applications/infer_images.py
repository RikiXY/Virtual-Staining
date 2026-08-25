from __future__ import annotations

from pathlib import Path

from virtual_staining.config.run import RunConfig
from virtual_staining.experiment.run_paths import RunPaths
from virtual_staining.inference.runner import load_inference_generator, resolve_inference_device
from virtual_staining.inference.single import (
    DEFAULT_TILE_OVERLAP,
    SUPPORTED_OUTPUT_FORMATS,
    DirectoryInferenceResult,
    InferenceRuntime,
    SingleInferenceMode,
    SingleInferenceResult,
    run_image_path_inference,
)

__all__ = [
    "DEFAULT_TILE_OVERLAP",
    "SUPPORTED_OUTPUT_FORMATS",
    "DirectoryInferenceResult",
    "SingleInferenceResult",
    "infer_images",
]


def _resolve_input_specs(
    input_specs: tuple[str, ...], input_names: tuple[str, ...]
) -> dict[str, Path]:
    if not input_specs:
        raise ValueError("At least one input is required.")
    if not input_names:
        raise ValueError("Configured model inputs must not be empty.")

    named: dict[str, Path] = {}
    bare: list[str] = []
    for spec in input_specs:
        if not spec:
            raise ValueError("Input specifications must not be empty.")
        name, separator, raw_path = spec.partition("=")
        if not separator:
            bare.append(spec)
            continue
        if not name.strip() or not raw_path.strip():
            raise ValueError(f"Input specification must include a name and path: {spec!r}")
        if name in named:
            raise ValueError(f"Duplicate input modality: {name}")
        if name not in input_names:
            raise ValueError(f"Unknown input modality: {name}")
        named[name] = Path(raw_path)

    if bare:
        if len(input_specs) != 1 or len(input_names) != 1:
            raise ValueError(
                "Bare --input PATH is only supported for models with one configured input."
            )
        if not bare[0].strip():
            raise ValueError("Input path must not be blank.")
        named[input_names[0]] = Path(bare[0])

    missing = [name for name in input_names if name not in named]
    if missing:
        raise ValueError(f"Missing input modalities: {missing}")
    return {name: named[name] for name in input_names}


def _create_runtime(config: RunConfig) -> InferenceRuntime:
    if config.inference is None:
        raise ValueError("RunConfig.inference is required to run image inference.")
    paths = RunPaths(config.project.run_root)
    paths.create_directories()
    device = resolve_inference_device()
    generator, checkpoint_path = load_inference_generator(config, paths, device)
    output_dir = config.inference.output_dir
    return InferenceRuntime(
        generator=generator,
        checkpoint_path=checkpoint_path,
        image_size=config.project.image_size,
        device=device,
        default_single_output_dir=output_dir or paths.artifacts_dir / "output_single",
        default_directory_output_dir=output_dir or paths.artifacts_dir / "output_images",
    )


def infer_images(
    config_path: Path,
    input_specs: tuple[str, ...],
    output_path: Path | None = None,
    *,
    recursive: bool = False,
    mode: SingleInferenceMode = "auto",
    tile_overlap: int = DEFAULT_TILE_OVERLAP,
    output_format: str = "same",
) -> SingleInferenceResult | DirectoryInferenceResult:
    """Application-level image inference entry point for files or directories."""
    config = RunConfig.from_yaml(config_path.resolve())
    input_paths = _resolve_input_specs(input_specs, tuple(config.model.inputs))
    return run_image_path_inference(
        lambda: _create_runtime(config),
        input_paths,
        output_path,
        recursive=recursive,
        mode=mode,
        tile_overlap=tile_overlap,
        output_format=output_format,
    )
