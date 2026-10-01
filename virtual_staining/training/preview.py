"""Validation preview boundary between method-owned validation and preview persistence."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import torch
from torchvision.utils import save_image

from virtual_staining.models.io_contract import denormalize_model_output

if TYPE_CHECKING:
    from virtual_staining.training.benchmarking import TrainingBenchmarkRecorder

_CYCLE_GRID_ROLES = ("real_A", "fake_B", "real_B", "fake_A")


@dataclass(frozen=True)
class ValidationPreview:
    """Detached, device-resident NCHW batches keyed by semantic role, in display order.

    Sinks are called synchronously and must copy any tensor they keep after ``write``.
    """

    epoch: int
    batch_index: int
    images: dict[str, torch.Tensor]


class ValidationPreviewSink(Protocol):
    def wants(self, epoch: int, batch_index: int) -> bool: ...
    def write(self, preview: ValidationPreview) -> None: ...


class ValidationPreviewWriter:
    """Write the first sample of the first ``batches`` validation batches as TIFFs.

    Roles become ``epoch{e}_batch{b}_{role}.tif``; CycleGAN's four roles become one
    ``_preview.tif`` grid.
    """

    def __init__(
        self,
        output_dir: Path,
        *,
        batches: int = 5,
        benchmark_recorder: TrainingBenchmarkRecorder | None = None,
    ) -> None:
        self.output_dir = output_dir
        self.batches = batches
        self._benchmark_recorder = benchmark_recorder

    def wants(self, epoch: int, batch_index: int) -> bool:
        return batch_index < self.batches

    def write(self, preview: ValidationPreview) -> None:
        recorder = self._benchmark_recorder
        stem = f"epoch{preview.epoch}_batch{preview.batch_index}"
        with nullcontext() if recorder is None else recorder.phase("preview_io"):
            self.output_dir.mkdir(parents=True, exist_ok=True)
            if tuple(preview.images) == _CYCLE_GRID_ROLES:
                grid = torch.stack([image[0].float() for image in preview.images.values()])
                save_image(
                    denormalize_model_output(grid),
                    self.output_dir / f"{stem}_preview.tif",
                    nrow=4,
                )
                return
            for role, image in preview.images.items():
                save_image(
                    denormalize_model_output(image[0]), self.output_dir / f"{stem}_{role}.tif"
                )
