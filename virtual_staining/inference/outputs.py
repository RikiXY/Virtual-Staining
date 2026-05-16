from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torchvision.utils import save_image

if TYPE_CHECKING:
    from virtual_staining.data.manifest import ManifestRecord


def generated_filename_for_sample(sample_id: str, suffix: str) -> str:
    """Return the canonical generated filename for a manifest sample ID."""
    return f"{sample_id}_target_generated{suffix.lower()}"


def generated_path_for_record(record: ManifestRecord, output_dir: Path) -> Path:
    """Return the expected generated path for a manifest record."""
    return output_dir / generated_filename_for_sample(record.sample_id, record.input_path.suffix)


class InferenceOutputWriter:
    """Writes generated images to a directory using the canonical naming policy."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write(self, sample_id: str, suffix: str, tensor: torch.Tensor) -> Path:
        """
        Save a generated tensor as <sample_id>_target_generated<suffix>.

        Parameters
        ----------
        sample_id:
            The manifest sample identifier (e.g. "00000_00000").
        suffix:
            The file extension (e.g. ".tif").
        tensor:
            A [C, H, W] tensor already in [0, 1] range.
        """
        filename = generated_filename_for_sample(sample_id, suffix)
        path = self.output_dir / filename
        save_image(tensor, path)
        return path
