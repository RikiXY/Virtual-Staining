from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import autocast
from torchvision import transforms

from virtual_staining.checkpoint_contract import (
    check_generator_arch,
    validate_checkpoint_metadata,
)
from virtual_staining.checkpoint_selection import resolve_checkpoint_path
from virtual_staining.config.run import RunConfig
from virtual_staining.experiment.run_layout import RunLayout
from virtual_staining.models.factory import build_generator
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
    """Return the device used by inference entry points."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_inference_transform(image_size: tuple[int, int]) -> transforms.Compose:
    """Build the image transform expected by the generator."""
    return build_model_input_transform(image_size)


def _resolve_checkpoint(config: RunConfig, paths: RunLayout) -> Path:
    """Resolve the inference checkpoint path from RunConfig."""
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


def load_inference_generator(
    config: RunConfig,
    paths: RunLayout,
    device: torch.device,
) -> tuple[nn.Module, Path]:
    """Load and validate the configured generator checkpoint."""
    checkpoint_path = _resolve_checkpoint(config, paths)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    stored_size = checkpoint.get("image_size")
    if stored_size is not None and tuple(stored_size) != tuple(config.project.image_size):
        raise ValueError(
            "Image size mismatch between checkpoint and inference config. "
            f"Checkpoint image_size={tuple(stored_size)}, "
            f"config image_size={tuple(config.project.image_size)}."
        )

    checkpoint_arch = validate_checkpoint_metadata(checkpoint, checkpoint_path)

    generator = build_generator(config.model).to(device)
    check_generator_arch(checkpoint_arch, generator, target_modality=config.model.target)
    generator.load_state_dict(checkpoint["generator_state_dict"])
    generator.eval()
    return generator, checkpoint_path
