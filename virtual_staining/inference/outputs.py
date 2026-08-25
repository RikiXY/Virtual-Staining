from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from virtual_staining.utils.artifacts import generated_filename

if TYPE_CHECKING:
    from virtual_staining.data.manifest import ManifestRecord


def generated_path_for_record(record: ManifestRecord, output_dir: Path) -> Path:
    """Return the expected generated path for a manifest record."""
    return output_dir / generated_filename(record.sample_id, record.target_path.suffix)
