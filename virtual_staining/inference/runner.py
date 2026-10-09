from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeAlias

import torch
import torch.nn as nn
from torch.amp import autocast
from torchvision import transforms

from virtual_staining.checkpoint_contract import read_checkpoint, validate_checkpoint
from virtual_staining.checkpoint_selection import resolve_checkpoint_path
from virtual_staining.config.run import RunConfig
from virtual_staining.experiment.run_layout import RunLayout
from virtual_staining.models.io_contract import (
    MODEL_OUTPUT_RANGE,
    build_model_input_transform,
    denormalize_model_output,
)

#: ``{input_name: NCHW tensor in [-1, 1]}`` -> ``{output_name: NCHW RGB tensor in [-1, 1]}``.
Predictor: TypeAlias = Callable[[dict[str, torch.Tensor]], Mapping[str, torch.Tensor]]
#: Slack for float noise around the declared output range; anything further out is rejected.
OUTPUT_RANGE_TOLERANCE = 1e-3


@dataclass
class InferenceResult:
    output_dir: Path
    generated_paths: list[Path] = field(default_factory=list)
    num_samples: int = 0


def validate_prediction(
    outputs: object, reference: torch.Tensor, output_names: tuple[str, ...]
) -> dict[str, torch.Tensor]:
    """Enforce the named same-grid RGB output contract before anything is published.

    ``reference`` is one NCHW predictor input. The predictor must return a mapping with
    exactly ``output_names`` in order (one output is a one-item mapping), each value a
    finite ``(N, 3, H, W)`` tensor on exactly that grid within the declared range. Nothing
    is cropped, padded, resized, selected or clamped away.
    """
    if not isinstance(outputs, Mapping):
        raise TypeError(
            "Predictor must return a mapping of output name to RGB tensor, got "
            f"{type(outputs).__name__}"
        )
    if tuple(outputs) != output_names:
        raise ValueError(
            f"Predictor outputs {tuple(outputs)} must be exactly {output_names} in that order"
        )
    expected = (reference.shape[0], 3, *reference.shape[-2:])
    low, high = MODEL_OUTPUT_RANGE
    for name, output in outputs.items():
        if not isinstance(output, torch.Tensor):
            raise TypeError(f"Predictor output {name!r} must be a tensor")
        if tuple(output.shape) != expected:
            raise ValueError(
                f"Predictor output {name!r} shape {tuple(output.shape)} violates the same-grid "
                f"RGB contract: expected {expected} (input batch, 3 channels, input height/width)"
            )
        if not output.is_floating_point():
            raise TypeError(
                f"Predictor output {name!r} must be a floating tensor, got {output.dtype}"
            )
        if not bool(torch.isfinite(output).all()):
            raise ValueError(f"Predictor output {name!r} contains NaN or Inf values")
        if (
            float(output.min()) < low - OUTPUT_RANGE_TOLERANCE
            or float(output.max()) > high + OUTPUT_RANGE_TOLERANCE
        ):
            raise ValueError(
                f"Predictor output {name!r} must lie in [{low}, {high}], got "
                f"[{float(output.min()):.4g}, {float(output.max()):.4g}]"
            )
    return dict(outputs)


@torch.no_grad()
def predict_batch(
    predictor: Predictor,
    inputs: dict[str, torch.Tensor],
    device: torch.device,
    output_names: tuple[str, ...],
) -> dict[str, torch.Tensor]:
    """Run a caller-prepared predictor once; the predictor itself is never moved or mutated.

    Returns every named output denormalized to [0, 1], in ``output_names`` order.
    """
    moved = {name: value.to(device) for name, value in inputs.items()}
    with autocast(device_type=device.type, enabled=device.type == "cuda"):
        outputs = predictor(moved)
    validated = validate_prediction(outputs, next(iter(moved.values())), output_names)
    return {name: denormalize_model_output(value) for name, value in validated.items()}


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


def inference_output_dir(config: RunConfig, paths: RunLayout) -> Path:
    """Return ``inference.output_dir``, defaulting to the run's test output directory."""
    configured = config.inference.output_dir if config.inference is not None else None
    return configured or paths.output_test_dir


def inference_direction(config: RunConfig) -> str | None:
    """Return the selected prediction direction, or None for single-direction methods.

    Methods with several directions default to their first declared direction.
    """
    assert config.method is not None
    directions = config.method.definition.prediction_directions
    if len(directions) < 2:
        return None
    configured = config.inference.direction if config.inference is not None else None
    return configured or directions[0]


def inference_input_names(config: RunConfig) -> tuple[str, ...]:
    """Return the named inputs the configured inference direction consumes."""
    assert config.method is not None
    return config.method.definition.prediction_inputs(config, inference_direction(config))


def inference_output_names(config: RunConfig) -> tuple[str, ...]:
    """Return the ordered named outputs the configured inference direction produces."""
    assert config.method is not None
    return config.method.definition.prediction_outputs(config, inference_direction(config))


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
    assert config.method is not None
    if checkpoint_path is None:
        checkpoint_path = resolve_inference_checkpoint(config, paths)
    definition = config.method.definition
    payload = read_checkpoint(checkpoint_path)
    config.definitions.require_checkpoint(payload, checkpoint_path)
    checkpoint = validate_checkpoint(
        payload,
        definition.inference_checkpoint_identity(config, payload, checkpoint_path),
        checkpoint_path,
    )
    generator = definition.build_inference_model(
        config, checkpoint, direction=inference_direction(config), device=device
    )
    generator.eval()
    return generator, checkpoint_path
