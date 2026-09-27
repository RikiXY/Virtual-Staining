from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import autocast
from torchvision import transforms

from virtual_staining.checkpoint_contract import read_checkpoint, validate_checkpoint
from virtual_staining.checkpoint_selection import resolve_checkpoint_path
from virtual_staining.config.run import RunConfig
from virtual_staining.experiment.run_layout import RunLayout
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


def resolve_inference_checkpoint(config: RunConfig, paths: RunLayout) -> Path:
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


def inference_direction(config: RunConfig) -> str | None:
    """Return the selected prediction direction, or None for single-direction methods.

    Methods with several directions default to their first declared direction.
    """
    directions = config.method.definition.prediction_directions
    if len(directions) < 2:
        return None
    configured = config.inference.direction if config.inference is not None else None
    return configured or directions[0]


def inference_input_names(config: RunConfig) -> tuple[str, ...]:
    """Return the named inputs the configured inference direction consumes."""
    return config.method.definition.prediction_inputs(config, inference_direction(config))


def load_inference_generator(
    config: RunConfig,
    paths: RunLayout,
    device: torch.device,
    checkpoint_path: Path | None = None,
) -> tuple[nn.Module, Path]:
    """Build the prediction network through the selected method definition.

    The checkpoint must name registered definitions and match the config's semantic
    identity before the definition builds anything; only the prediction network is
    constructed (no optimizer, scheduler, objective or unused network).
    """
    if checkpoint_path is None:
        checkpoint_path = resolve_inference_checkpoint(config, paths)
    definition = config.method.definition
    payload = read_checkpoint(checkpoint_path)
    config.definitions.require_checkpoint(payload, checkpoint_path)
    checkpoint = validate_checkpoint(
        payload, definition.checkpoint_identity(config), checkpoint_path
    )
    generator = definition.build_inference_model(
        config, checkpoint, direction=inference_direction(config), device=device
    )
    generator.eval()
    return generator, checkpoint_path
