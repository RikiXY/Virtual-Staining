from __future__ import annotations

from importlib import import_module
from typing import Any

import torch.nn as nn

from virtual_staining.config.model import ModelConfig
from virtual_staining.models.discriminator import PatchGANDiscriminator
from virtual_staining.models.generator import ConcatUNetGenerator, ResNetGenerator


def load_custom_class(class_path: str, *, contract: str) -> type[nn.Module]:
    module_name, separator, class_name = class_path.partition(":")
    if not separator or not module_name or not class_name:
        raise ValueError(f"{contract} class_path must use 'package.module:ClassName' syntax")
    try:
        candidate: Any = getattr(import_module(module_name), class_name)
    except (ImportError, AttributeError) as exc:
        raise ValueError(f"Could not import {contract} class '{class_path}': {exc}") from exc
    if not isinstance(candidate, type) or not issubclass(candidate, nn.Module):
        raise TypeError(f"Custom {contract} '{class_path}' must be a torch.nn.Module class")
    return candidate


def build_generator(config: ModelConfig) -> nn.Module:
    generator = config.generator
    if generator.architecture == "custom":
        assert generator.class_path is not None
        generator_class = load_custom_class(generator.class_path, contract="generator")
        return generator_class(**generator.params)
    if generator.architecture == "resnet":
        return ResNetGenerator(
            base_channels=generator.base_channels,
            blocks=generator.blocks,
            norm=generator.norm,
        )
    return ConcatUNetGenerator(
        config.inputs,
        base_channels=generator.base_channels,
        norm=generator.norm,
        dropout=generator.dropout,
        bilinear=generator.bilinear,
    )


def build_discriminator(config: ModelConfig, *, conditional: bool = True) -> nn.Module:
    discriminator = config.discriminator
    if discriminator.architecture == "custom":
        assert discriminator.class_path is not None
        return load_custom_class(discriminator.class_path, contract="discriminator")(
            **discriminator.params
        )
    return PatchGANDiscriminator(
        in_channels=(3 * len(config.inputs)) + 3 if conditional else 3,
        ndf=discriminator.ndf,
        norm=discriminator.norm,
        use_sigmoid=discriminator.use_sigmoid,
    )
