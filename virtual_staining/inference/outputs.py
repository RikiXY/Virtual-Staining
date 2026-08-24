from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from virtual_staining.data.manifest import ManifestRecord


def generated_filename_for_sample(sample_id: str, suffix: str) -> str:
    """Return the canonical generated filename for a manifest sample ID."""
    return f"{sample_id}_target_generated{suffix.lower()}"


def generated_path_for_record(record: ManifestRecord, output_dir: Path) -> Path:
    """Return the expected generated path for a manifest record."""
    return output_dir / generated_filename_for_sample(record.sample_id, record.target_path.suffix)
