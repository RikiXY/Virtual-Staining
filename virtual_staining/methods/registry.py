from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from virtual_staining.config.run import RunConfig
from virtual_staining.experiment.run_layout import RunLayout
from virtual_staining.methods.pix2pix import Pix2PixMethod
from virtual_staining.training.runtime import TrainingMethodRuntime

if TYPE_CHECKING:
    from virtual_staining.training.benchmarking import TrainingBenchmarkRecorder


def resolve_training_method(
    config: RunConfig,
    run_paths: RunLayout,
    device: torch.device,
    *,
    benchmark_recorder: TrainingBenchmarkRecorder | None = None,
) -> TrainingMethodRuntime:
    """Construct the selected built-in training method."""
    if config.method.name == "pix2pix":
        return Pix2PixMethod(
            config,
            run_paths,
            device,
            benchmark_recorder=benchmark_recorder,
        )
    if config.method.name == "cyclegan":
        raise NotImplementedError(
            "method.name='cyclegan' is recognized, but its training runtime is not implemented yet"
        )
    raise AssertionError(f"Unsupported built-in method: {config.method.name!r}")
