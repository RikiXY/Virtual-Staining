from __future__ import annotations

from pathlib import Path

import torch
from torchvision.utils import save_image


class InferenceOutputWriter:
    """Writes generated images to a directory using the canonical naming policy."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write(self, source_stem: str, suffix: str, tensor: torch.Tensor) -> Path:
        """
        Save a generated tensor as <source_stem>_target_generated<suffix>.

        Parameters
        ----------
        source_stem:
            The stem of the original source file (e.g. "00000_00000_source").
            The "_source" suffix will be stripped automatically.
        suffix:
            The file extension (e.g. ".tif").
        tensor:
            A [C, H, W] tensor already in [0, 1] range.
        """
        prefix = source_stem[: -len("_source")] if source_stem.endswith("_source") else source_stem
        filename = f"{prefix}_target_generated{suffix.lower()}"
        path = self.output_dir / filename
        save_image(tensor, path)
        return path
