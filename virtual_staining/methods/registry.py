from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from virtual_staining.config.run import RunConfig
from virtual_staining.methods.cyclegan import CycleGANMethod
from virtual_staining.methods.pix2pix import Pix2PixMethod
from virtual_staining.training.runtime import TrainingMethodRuntime

if TYPE_CHECKING:
    from virtual_staining.training.benchmarking import TrainingBenchmarkRecorder


def resolve_training_method(
    config: RunConfig,
    device: torch.device,
    *,
    seed: int | None = None,
    benchmark_recorder: TrainingBenchmarkRecorder | None = None,
) -> TrainingMethodRuntime:
    """Construct the selected built-in training method.

    ``seed`` is the resolved experiment seed; CycleGAN derives its replay-pool RNGs from it.
    """
    if config.method.name == "pix2pix":
        return Pix2PixMethod(
            config,
            device,
            benchmark_recorder=benchmark_recorder,
        )
    if config.method.name == "cyclegan":
        if seed is None:
            raise ValueError("method.name='cyclegan' requires the resolved training seed")
        return CycleGANMethod(config, device, seed=seed)
    raise AssertionError(f"Unsupported built-in method: {config.method.name!r}")
