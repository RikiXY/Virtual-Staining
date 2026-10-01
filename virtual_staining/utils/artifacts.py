"""The one identity of a generated artifact: ``(sample_id, output_name)``.

Layout: ``<output_dir>/<output_name>/<sample_id>_generated<ext>`` (recursive directory
inference inserts the input's relative parent before ``<output_name>``). ``output_name``
must be a persisted modality identifier (``[A-Za-z][A-Za-z0-9_-]*``), so it is always
exactly one ordinary directory below ``output_dir``: never ``.``, ``..`` or a path. One
directory per output keeps every pair collision-free, and the path inverts back to its
pair (``generated_identity``). Every producer and consumer of generated images uses these
helpers.
"""

from __future__ import annotations

from pathlib import Path

from virtual_staining.utils.identifiers import is_modality_name
from virtual_staining.utils.image_io import VALID_IMAGE_EXTENSIONS

TARGET_SUFFIX = "_target"
GENERATED_SUFFIX = "_generated"


def require_output_name(output_name: object) -> str:
    """Return ``output_name`` if it is a safe persisted identifier, else raise ``ValueError``."""
    if not is_modality_name(output_name):
        raise ValueError(
            f"output_name must be an identifier matching [A-Za-z][A-Za-z0-9_-]*, "
            f"got {output_name!r}"
        )
    return str(output_name)


def generated_path(output_dir: Path, sample_id: str, output_name: str, suffix: str) -> Path:
    """Where the generated ``output_name`` image of ``sample_id`` lives under ``output_dir``."""
    if not sample_id or "/" in sample_id or "\\" in sample_id:
        raise ValueError(f"sample_id must be a non-empty file-name component: {sample_id!r}")
    directory = require_output_name(output_name)
    return output_dir / directory / f"{sample_id}{GENERATED_SUFFIX}{suffix.lower()}"


def generated_identity(path: str | Path) -> tuple[str, str]:
    """Invert ``generated_path``: the ``(sample_id, output_name)`` of a generated file."""
    file = Path(path)
    sample_id = sample_id_for_suffix(file, GENERATED_SUFFIX, "Generated")
    return sample_id, require_output_name(file.parent.name)


def collect_generated_artifacts(root: Path, output_name: str) -> tuple[Path, ...]:
    """Recursively list the sorted generated images of one output under ``root``.

    These are the files in any ``<output_name>`` directory, which is where
    ``generated_path`` places them (below any recursive relative parent).
    """
    require_output_name(output_name)
    return tuple(
        path
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and path.parent.name == output_name
        and path.suffix.lower() in VALID_IMAGE_EXTENSIONS
        and path.stem.endswith(GENERATED_SUFFIX)
    )


def sample_id_for_suffix(path: str | Path, suffix: str, label: str = "File") -> str:
    name = Path(path).stem
    if not name.endswith(suffix) or name == suffix:
        raise ValueError(f"{label} file does not end with '{suffix}': {path}")
    return name[: -len(suffix)]
