from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from virtual_staining.utils.artifacts import generated_path

if TYPE_CHECKING:
    from virtual_staining.data.manifest import ManifestRecord


def generated_path_for_record(record: ManifestRecord, output_dir: Path, output_name: str) -> Path:
    """The generated ``output_name`` image of ``record``, with that domain's file suffix."""
    suffix = record.domain_path(output_name).suffix
    return generated_path(output_dir, record.sample_id, output_name, suffix)
