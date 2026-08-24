from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import autocast
from torchvision import transforms

from virtual_staining.config.run import RunConfig
from virtual_staining.experiment.run_paths import RunPaths
from virtual_staining.models.generator import ConcatUNetGenerator
from virtual_staining.training.checkpoint_selection import resolve_best_checkpoint_path
from virtual_staining.training.checkpoints import (
    _check_generator_arch,
    _validate_checkpoint_metadata,
)
from virtual_staining.utils.dimensions import to_torchvision_hw


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
    return (output * 0.5 + 0.5).clamp(0, 1)


def resolve_inference_device() -> torch.device:
    """Return the device used by inference entry points."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_inference_transform(image_size: tuple[int, int]) -> transforms.Compose:
    """Build the image transform expected by the generator."""
    return transforms.Compose(
        [
            transforms.Resize(to_torchvision_hw(image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )


def _resolve_checkpoint(config: RunConfig, paths: RunPaths) -> Path:
    """Resolve the inference checkpoint path from RunConfig."""
    if config.inference is None:
        raise ValueError("RunConfig.inference is required to run inference.")

    if config.inference.checkpoint_path is not None:
        checkpoint_path = config.inference.checkpoint_path
        if not checkpoint_path.is_absolute():
            checkpoint_path = paths.root / checkpoint_path
        return checkpoint_path

    if config.inference.checkpoint_policy == "latest":
        candidates = sorted(paths.checkpoints_dir.glob("ep*.pth"))
        if not candidates:
            raise FileNotFoundError(
                f"checkpoint_policy='latest' but no checkpoints found in {paths.checkpoints_dir}"
            )
        return candidates[-1]

    if config.inference.checkpoint_policy in {"best", "top_k"}:
        return resolve_best_checkpoint_path(
            paths.checkpoints_dir,
            policy=config.inference.checkpoint_policy,
            metric=config.inference.checkpoint_metric,
            rank=config.inference.checkpoint_rank or 1,
        )

    raise ValueError(
        "inference.checkpoint_path or inference.checkpoint_policy must be set in the config."
    )


def load_inference_generator(
    config: RunConfig,
    paths: RunPaths,
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

    checkpoint_arch = _validate_checkpoint_metadata(checkpoint, checkpoint_path)

    generator_config = config.model.generator
    generator = ConcatUNetGenerator(
        config.model.inputs,
        base_channels=generator_config.base_channels,
        norm=generator_config.norm,
        dropout=generator_config.dropout,
        bilinear=generator_config.bilinear,
    ).to(device)
    _check_generator_arch(checkpoint_arch, generator, target_modality=config.model.target)
    generator.load_state_dict(checkpoint["generator_state_dict"])
    generator.eval()
    return generator, checkpoint_path
