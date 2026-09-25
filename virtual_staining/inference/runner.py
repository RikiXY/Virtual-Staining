from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import autocast
from torchvision import transforms

from virtual_staining.checkpoint_selection import resolve_checkpoint_path
from virtual_staining.config.inference import InferenceDirection
from virtual_staining.config.run import RunConfig
from virtual_staining.experiment.run_layout import RunLayout
from virtual_staining.methods.cyclegan import load_cyclegan_inference_generator
from virtual_staining.methods.pix2pix import load_pix2pix_inference_generator
from virtual_staining.models.io_contract import (
    build_model_input_transform,
    denormalize_model_output,
)


@dataclass
class InferenceResult:
    output_dir: Path
    generated_paths: list[Path] = field(default_factory=list)
    num_samples: int = 0


@torch.no_grad()
def predict_batch(
    generator: nn.Module,
    inputs: dict[str, torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    with autocast(device_type=device.type, enabled=device.type == "cuda"):
        output = generator({name: value.to(device) for name, value in inputs.items()})
    return denormalize_model_output(output)


def resolve_inference_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_inference_transform(image_size: tuple[int, int]) -> transforms.Compose:
    return build_model_input_transform(image_size)


def _resolve_checkpoint(config: RunConfig, paths: RunLayout) -> Path:
    if config.inference is None:
        raise ValueError("RunConfig.inference is required to run inference.")

    if config.inference.checkpoint_path is not None:
        checkpoint_path = config.inference.checkpoint_path
        if not checkpoint_path.is_absolute():
            checkpoint_path = paths.root / checkpoint_path
        return checkpoint_path
    if config.inference.checkpoint_policy is None:
        raise ValueError(
            "inference.checkpoint_path or inference.checkpoint_policy must be set in the config."
        )
    return resolve_checkpoint_path(
        paths.checkpoints_dir,
        policy=config.inference.checkpoint_policy,
        metric=config.inference.checkpoint_metric,
        rank=config.inference.checkpoint_rank or 1,
    )


def inference_direction(config: RunConfig) -> InferenceDirection | None:
    """Return the CycleGAN translation direction (default A_to_B), or None for Pix2Pix."""
    if config.method.name != "cyclegan":
        return None
    configured = config.inference.direction if config.inference is not None else None
    return configured or "A_to_B"


def inference_input_names(config: RunConfig) -> tuple[str, ...]:
    """Return the named inputs the configured inference direction consumes."""
    direction = inference_direction(config)
    if direction is None:
        return tuple(config.model.inputs)
    return (config.model.inputs[0],) if direction == "A_to_B" else (config.model.target,)


def load_inference_generator(
    config: RunConfig,
    paths: RunLayout,
    device: torch.device,
) -> tuple[nn.Module, Path]:
    checkpoint_path = _resolve_checkpoint(config, paths)
    direction = inference_direction(config)
    if direction is None:
        generator = load_pix2pix_inference_generator(checkpoint_path, config, device)
    else:
        generator = load_cyclegan_inference_generator(checkpoint_path, config, direction, device)
    return generator, checkpoint_path
