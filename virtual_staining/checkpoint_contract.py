from __future__ import annotations

from pathlib import Path
from typing import Any

import torch.nn as nn

from virtual_staining.models.io_contract import GENERATOR_OUTPUT_ACTIVATION, NORMALIZATION_CONTRACT

CHECKPOINT_FORMAT_VERSION: int = 3


def make_arch_metadata(
    generator: nn.Module,
    discriminator: nn.Module,
    *,
    target_modality: str | None = None,
) -> dict[str, Any]:
    input_names = tuple(getattr(generator, "input_names", ()))
    return {
        "generator": {
            "class": type(generator).__name__,
            "architecture": "concat_unet",
            "input_names": list(input_names),
            "target_modality": target_modality,
            "in_channels": getattr(getattr(generator, "unet", generator), "in_channels", None),
            "out_channels": getattr(getattr(generator, "unet", generator), "out_channels", 3),
            "base_channels": getattr(getattr(generator, "unet", generator), "base_channels", None),
            "norm": getattr(getattr(generator, "unet", generator), "norm", None),
            "dropout": getattr(getattr(generator, "unet", generator), "dropout", None),
            "bilinear": getattr(getattr(generator, "unet", generator), "bilinear", None),
            "output_activation": GENERATOR_OUTPUT_ACTIVATION,
        },
        "discriminator": {
            "class": type(discriminator).__name__,
            "in_channels": getattr(discriminator, "in_channels", None),
            "ndf": getattr(discriminator, "ndf", None),
            "norm": getattr(discriminator, "norm", None),
            "use_sigmoid": getattr(discriminator, "use_sigmoid", None),
        },
    }


def validate_checkpoint_metadata(checkpoint: dict[str, Any], path: Path) -> dict[str, Any]:
    if checkpoint.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"Checkpoint format version {checkpoint.get('format_version')!r} does not "
            f"match current version {CHECKPOINT_FORMAT_VERSION}. Re-train from scratch "
            "with the current code."
        )
    arch = checkpoint.get("architecture")
    if not isinstance(arch, dict):
        raise ValueError(
            f"Checkpoint '{path}' has no architecture metadata. "
            "Only current checkpoints are supported."
        )
    generator_arch = arch.get("generator")
    if not isinstance(generator_arch, dict):
        raise ValueError("Checkpoint generator architecture metadata must be a mapping.")
    if (
        generator_arch.get("architecture") != "concat_unet"
        or not isinstance(generator_arch.get("input_names"), list)
        or not generator_arch.get("target_modality")
    ):
        raise ValueError(
            "Checkpoint is missing current named-generator metadata; retrain from scratch."
        )
    if generator_arch.get("output_activation") != GENERATOR_OUTPUT_ACTIVATION:
        raise ValueError("Checkpoint output activation does not match current code.")
    if checkpoint.get("normalization_contract") != NORMALIZATION_CONTRACT:
        raise ValueError("Checkpoint normalization contract does not match current code.")
    return arch


def check_generator_arch(
    checkpoint_arch: dict[str, Any],
    generator: nn.Module,
    *,
    target_modality: str | None = None,
) -> None:
    if (
        target_modality is not None
        and checkpoint_arch.get("generator", {}).get("target_modality") != target_modality
    ):
        raise ValueError("Checkpoint target modality does not match current model.")
    current = getattr(generator, "unet", generator)
    gen_arch = checkpoint_arch.get("generator", {})
    for key in (
        "class",
        "architecture",
        "input_names",
        "in_channels",
        "out_channels",
        "base_channels",
        "norm",
        "dropout",
        "bilinear",
    ):
        if key == "class":
            curr_val = type(generator).__name__
        elif key == "architecture":
            curr_val = "concat_unet"
        elif key == "input_names":
            curr_val = list(getattr(generator, "input_names", ()))
        else:
            curr_val = getattr(current, key, 3 if key == "out_channels" else None)
        if gen_arch.get(key) != curr_val:
            raise ValueError(
                f"Architecture mismatch for generator.{key}: checkpoint has "
                f"{gen_arch.get(key)!r}, inference model has {curr_val!r}."
            )


def check_discriminator_arch(checkpoint_arch: dict[str, Any], discriminator: nn.Module) -> None:
    disc_arch = checkpoint_arch.get("discriminator", {})
    for key in ("class", "in_channels", "ndf", "norm", "use_sigmoid"):
        current = (
            type(discriminator).__name__ if key == "class" else getattr(discriminator, key, None)
        )
        if disc_arch.get(key) != current:
            raise ValueError(
                f"Architecture mismatch for discriminator.{key}: checkpoint has "
                f"{disc_arch.get(key)!r}, current model has {current!r}."
            )
