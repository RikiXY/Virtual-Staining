from __future__ import annotations

from pathlib import Path

TARGET_SUFFIX = "_target"
GENERATED_SUFFIX = "_target_generated"


def generated_filename(sample_id: str, suffix: str, direction: str | None = None) -> str:
    """Name a generated artifact; CycleGAN passes its direction so directions never collide."""
    label = GENERATED_SUFFIX if direction is None else f"_{direction}_generated"
    return f"{sample_id}{label}{suffix.lower()}"


def generated_sample_id(path: str | Path) -> str:
    return sample_id_for_suffix(path, GENERATED_SUFFIX, "Generated")


def sample_id_for_suffix(path: str | Path, suffix: str, label: str = "File") -> str:
    name = Path(path).stem
    if not name.endswith(suffix):
        raise ValueError(f"{label} file does not end with '{suffix}': {path}")
    return name[: -len(suffix)]
