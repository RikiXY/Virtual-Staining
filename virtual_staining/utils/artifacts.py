from __future__ import annotations

from pathlib import Path

from virtual_staining.utils.image_io import VALID_IMAGE_EXTENSIONS

TARGET_SUFFIX = "_target"
GENERATED_SUFFIX = "_target_generated"


def generated_suffix(direction: str | None = None) -> str:
    """Return the stem suffix of a generated artifact; CycleGAN directions never collide."""
    return GENERATED_SUFFIX if direction is None else f"_{direction}_generated"


def generated_filename(sample_id: str, suffix: str, direction: str | None = None) -> str:
    return f"{sample_id}{generated_suffix(direction)}{suffix.lower()}"


def collect_generated_artifacts(root: Path, direction: str | None = None) -> tuple[Path, ...]:
    """Recursively list the sorted generated images under ``root`` for one direction."""
    label = generated_suffix(direction)
    return tuple(
        path
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and path.suffix.lower() in VALID_IMAGE_EXTENSIONS
        and path.stem.endswith(label)
    )


def generated_sample_id(path: str | Path) -> str:
    return sample_id_for_suffix(path, GENERATED_SUFFIX, "Generated")


def sample_id_for_suffix(path: str | Path, suffix: str, label: str = "File") -> str:
    name = Path(path).stem
    if not name.endswith(suffix):
        raise ValueError(f"{label} file does not end with '{suffix}': {path}")
    return name[: -len(suffix)]
