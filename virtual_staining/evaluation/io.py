from __future__ import annotations

from pathlib import Path

from virtual_staining.utils.artifacts import (
    GENERATED_SUFFIX,
    TARGET_SUFFIX,
    sample_id_for_suffix,
)
from virtual_staining.utils.image_io import VALID_IMAGE_EXTENSIONS


def extract_sample_id(path: str | Path, suffix: str, label: str = "File") -> str:
    return sample_id_for_suffix(path, suffix, label)


def extract_single_sample_id(target_path: str | Path, generated_path: str | Path) -> str:
    target_id = extract_sample_id(target_path, TARGET_SUFFIX, "Target")
    generated_id = extract_sample_id(generated_path, GENERATED_SUFFIX, "Generated")
    if target_id != generated_id:
        raise ValueError(
            "Target and generated files refer to different sample ids. "
            f"Got '{target_id}' and '{generated_id}'."
        )
    return target_id


def collect_image_files(directory_path: str | Path, suffix: str, label: str) -> dict[str, Path]:
    directory = Path(directory_path)
    if not directory.is_dir():
        raise NotADirectoryError(f"{label} directory not found: {directory}")
    files: dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if (
            path.is_file()
            and path.suffix.lower() in VALID_IMAGE_EXTENSIONS
            and path.stem.endswith(suffix)
        ):
            files[extract_sample_id(path, suffix, label)] = path
    return files
