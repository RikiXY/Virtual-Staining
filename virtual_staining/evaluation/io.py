from __future__ import annotations

from pathlib import Path

from virtual_staining.utils.image_io import VALID_IMAGE_EXTENSIONS


def extract_sample_id(path: str | Path, suffix: str, label: str = "File") -> str:
    """Extract the sample id by removing the expected suffix from the filename."""
    name = Path(path).stem

    if not name.endswith(suffix):
        raise ValueError(f"{label} file does not end with '{suffix}': {path}")

    return name[: -len(suffix)]


def extract_single_sample_id(target_path: str | Path, generated_path: str | Path) -> str:
    """Check that target and generated belong to the same sample."""
    target_id = extract_sample_id(target_path, "_target", "Target")
    generated_id = extract_sample_id(generated_path, "_target_generated", "Generated")

    if target_id != generated_id:
        raise ValueError(
            "Target and generated files refer to different sample ids. "
            f"Got '{target_id}' and '{generated_id}'."
        )

    return target_id


def collect_image_files(directory_path: str | Path, suffix: str, label: str) -> dict[str, Path]:
    """Collect valid files from a directory, indexed by sample id."""
    directory = Path(directory_path)

    if not directory.is_dir():
        raise NotADirectoryError(f"{label} directory not found: {directory}")

    files: dict[str, Path] = {}

    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in VALID_IMAGE_EXTENSIONS:
            continue
        if not path.stem.endswith(suffix):
            continue

        sample_id = extract_sample_id(path, suffix, label)
        files[sample_id] = path

    return files


def build_evaluation_pairs(
    target_dir: Path,
    generated_dir: Path,
) -> tuple[list[tuple[Path, Path, str]], list[str]]:
    """Pair target images with generated images by sample ID.

    Returns (matched_pairs, skipped_sample_ids) where matched_pairs contains
    (target_path, generated_path, sample_id) tuples and skipped_sample_ids
    lists IDs that could not be paired due to a missing file on either side.
    """
    target_files = collect_image_files(target_dir, "_target", "Target")
    generated_files = collect_image_files(generated_dir, "_target_generated", "Generated")
    all_sample_ids = sorted(set(target_files) | set(generated_files))
    pairs: list[tuple[Path, Path, str]] = []
    skipped: list[str] = []

    for sample_id in all_sample_ids:
        if sample_id in target_files and sample_id in generated_files:
            pairs.append((target_files[sample_id], generated_files[sample_id], sample_id))
        else:
            skipped.append(sample_id)

    return pairs, skipped
