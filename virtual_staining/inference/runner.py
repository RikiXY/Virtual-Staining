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


class _TensorGeneratorAdapter(nn.Module):
    def __init__(self, generator: nn.Module, input_name: str) -> None:
        super().__init__()
        self.generator = generator
        self.input_names = (input_name,)

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        if tuple(inputs) != self.input_names:
            raise ValueError(f"Expected inference input {self.input_names}, got {tuple(inputs)}")
        return self.generator(inputs[self.input_names[0]])


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

    if checkpoint.get("format_version") == 3:
        if config.method.name != "pix2pix":
            raise ValueError("Legacy v3 checkpoints can only be loaded as Pix2Pix")
        checkpoint_arch = validate_checkpoint_metadata(checkpoint, checkpoint_path)
        generator = build_generator(config.model).to(device)
        check_generator_arch(checkpoint_arch, generator, target_modality=config.model.target)
        generator.load_state_dict(checkpoint["generator_state_dict"])
        generator.eval()
        return generator, checkpoint_path

    if checkpoint.get("format_version") != 4:
        raise ValueError(
            f"Unsupported checkpoint format version {checkpoint.get('format_version')!r}; "
            "supported versions are Pix2Pix v3 and method checkpoint v4"
        )
    method = checkpoint.get("method")
    if not isinstance(method, dict) or method.get("name") != config.method.name:
        raise ValueError("Checkpoint method does not match the configured method")
    state = checkpoint.get("method_state")
    if not isinstance(state, dict) or not isinstance(state.get("models"), dict):
        raise ValueError("Method checkpoint is missing model state")
    generator = build_generator(config.model).to(device)
    if config.method.name == "cyclegan":
        assert config.inference is not None
        direction = config.inference.direction or "A_to_B"
        state_name = f"G_{direction}"
        if state_name not in state["models"]:
            raise ValueError(f"CycleGAN checkpoint has no generator for direction {direction}")
        generator.load_state_dict(state["models"][state_name])
        input_name = config.model.inputs[0] if direction == "A_to_B" else config.model.target
        generator = _TensorGeneratorAdapter(generator, input_name).to(device)
    else:
        if "generator" not in state["models"]:
            raise ValueError("Pix2Pix method checkpoint is missing generator state")
        generator.load_state_dict(state["models"]["generator"])
    generator.eval()
    return generator, checkpoint_path


def inference_input_names(config: RunConfig) -> tuple[str, ...]:
    if (
        config.method.name == "cyclegan"
        and config.inference is not None
        and config.inference.direction == "B_to_A"
    ):
        return (config.model.target,)
    return config.model.inputs
