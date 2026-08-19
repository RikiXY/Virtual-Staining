from __future__ import annotations

from pathlib import Path

from virtual_staining.config.run import RunConfig
from virtual_staining.inference.single import (
    DEFAULT_TILE_OVERLAP,
    SUPPORTED_OUTPUT_FORMATS,
    DirectoryInferenceResult,
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


def infer_images(
    config_path: Path,
    input_path: Path,
    output_path: Path | None = None,
    *,
    recursive: bool = False,
    mode: SingleInferenceMode = "auto",
    tile_overlap: int = DEFAULT_TILE_OVERLAP,
    output_format: str = "same",
) -> SingleInferenceResult | DirectoryInferenceResult:
    """Application-level image inference entry point for files or directories."""
    config = RunConfig.from_yaml(config_path.resolve())
    return run_image_path_inference(
        config,
        input_path,
        output_path,
        recursive=recursive,
        mode=mode,
        tile_overlap=tile_overlap,
        output_format=output_format,
    )
