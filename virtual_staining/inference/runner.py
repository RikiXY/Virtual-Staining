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
from virtual_staining.checkpoint_selection import resolve_best_checkpoint_path
from virtual_staining.config.run import RunConfig
from virtual_staining.experiment.run_paths import RunPaths
from virtual_staining.models.generator import ConcatUNetGenerator
from virtual_staining.utils.dimensions import to_torchvision_hw


@dataclass
class InferenceResult:
    output_dir: Path
    generated_paths: list[Path] = field(default_factory=list)
    num_samples: int = 0


@dataclass(frozen=True)
class LoadedCheckpointGenerator:
    """Generator and inference metadata reconstructed from one checkpoint."""

    generator: nn.Module
    checkpoint_path: Path
    image_size: tuple[int, int]
    input_names: tuple[str, ...]
    target_modality: str
    channels_per_input: int
    device: torch.device


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


def load_checkpoint_generator(
    checkpoint_path: Path,
    device: torch.device,
) -> LoadedCheckpointGenerator:
    """Reconstruct a generator using only a current-format checkpoint."""
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint '{checkpoint_path}' must contain a mapping.")

    checkpoint_arch = validate_checkpoint_metadata(checkpoint, checkpoint_path)
    generator_arch = checkpoint_arch["generator"]

    input_names_value = generator_arch.get("input_names")
    if not isinstance(input_names_value, list) or any(
        not isinstance(name, str) or not name.strip() for name in input_names_value
    ):
        raise ValueError("Checkpoint generator input_names must contain non-empty strings.")
    input_names = tuple(input_names_value)
    if not input_names or len(set(input_names)) != len(input_names):
        raise ValueError("Checkpoint generator input_names must be non-empty and unique.")

    image_size_value = checkpoint.get("image_size")
    if (
        not isinstance(image_size_value, (list, tuple))
        or len(image_size_value) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in image_size_value)
        or any(value <= 0 for value in image_size_value)
    ):
        raise ValueError(
            "Checkpoint image_size must contain two positive integers for "
            "checkpoint-only inference."
        )
    image_size = (image_size_value[0], image_size_value[1])

    if generator_arch.get("class") != "ConcatUNetGenerator":
        raise ValueError(
            "Checkpoint generator class is not supported for checkpoint-only inference: "
            f"{generator_arch.get('class')!r}."
        )
    in_channels = generator_arch.get("in_channels")
    if (
        isinstance(in_channels, bool)
        or not isinstance(in_channels, int)
        or in_channels <= 0
        or in_channels % len(input_names) != 0
    ):
        raise ValueError(
            "Checkpoint generator in_channels must be a positive multiple of its input count."
        )
    if generator_arch.get("out_channels") != 3:
        raise ValueError("Checkpoint-only inference currently requires three output channels.")

    base_channels = generator_arch.get("base_channels")
    if isinstance(base_channels, bool) or not isinstance(base_channels, int) or base_channels <= 0:
        raise ValueError("Checkpoint generator base_channels must be a positive integer.")
    norm = generator_arch.get("norm")
    if norm not in {"batch", "instance"}:
        raise ValueError(f"Checkpoint generator norm is not supported: {norm!r}.")
    dropout = generator_arch.get("dropout")
    bilinear = generator_arch.get("bilinear")
    if not isinstance(dropout, bool) or not isinstance(bilinear, bool):
        raise ValueError("Checkpoint generator dropout and bilinear metadata must be booleans.")

    target_modality = generator_arch.get("target_modality")
    if not isinstance(target_modality, str) or not target_modality.strip():
        raise ValueError("Checkpoint generator target_modality must be a non-empty string.")

    channels_per_input = in_channels // len(input_names)
    generator = ConcatUNetGenerator(
        input_names,
        channels_per_input=channels_per_input,
        base_channels=base_channels,
        norm=norm,
        dropout=dropout,
        bilinear=bilinear,
    ).to(device)
    check_generator_arch(checkpoint_arch, generator, target_modality=target_modality)

    state_dict = checkpoint.get("generator_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint has no valid generator_state_dict.")
    try:
        generator.load_state_dict(state_dict)
    except RuntimeError as exc:
        raise ValueError(
            "Checkpoint generator weights do not match its architecture metadata."
        ) from exc
    generator.eval()

    return LoadedCheckpointGenerator(
        generator=generator,
        checkpoint_path=checkpoint_path,
        image_size=image_size,
        input_names=input_names,
        target_modality=target_modality,
        channels_per_input=channels_per_input,
        device=device,
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

    checkpoint_arch = validate_checkpoint_metadata(checkpoint, checkpoint_path)

    generator_config = config.model.generator
    generator = ConcatUNetGenerator(
        config.model.inputs,
        base_channels=generator_config.base_channels,
        norm=generator_config.norm,
        dropout=generator_config.dropout,
        bilinear=generator_config.bilinear,
    ).to(device)
    check_generator_arch(checkpoint_arch, generator, target_modality=config.model.target)
    generator.load_state_dict(checkpoint["generator_state_dict"])
    generator.eval()
    return generator, checkpoint_path
