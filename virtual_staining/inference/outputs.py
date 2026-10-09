from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from PIL import Image
from torchvision.utils import save_image

from virtual_staining.utils.artifacts import generated_path

if TYPE_CHECKING:
    from virtual_staining.data.manifest import ManifestRecord


def generated_path_for_record(record: ManifestRecord, output_dir: Path, output_name: str) -> Path:
    """The generated ``output_name`` image of ``record``, with that domain's file suffix."""
    suffix = record.domain_path(output_name).suffix
    return generated_path(output_dir, record.sample_id, output_name, suffix)


def save_rgb(output: torch.Tensor, output_path: Path) -> None:
    """Encode and verify one image before atomically replacing its destination."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(
        prefix=f".{output_path.stem}.", suffix=output_path.suffix, dir=output_path.parent
    )
    partial = Path(name)
    try:
        os.close(fd)
        save_image(output, partial)
        with Image.open(partial) as image:
            image.verify()
        # verify() checks structure; decoding also catches truncated pixel data.
        with Image.open(partial) as image:
            image.load()
        os.replace(partial, output_path)
    finally:
        partial.unlink(missing_ok=True)
