from __future__ import annotations

import torch.nn as nn

from virtual_staining.models.config import DiscriminatorConfig, GeneratorConfig
from virtual_staining.models.discriminator import PatchGANDiscriminator
from virtual_staining.models.generator import UNetGenerator


def build_generator(config: GeneratorConfig) -> nn.Module:
    if config.name == "unet":
        return UNetGenerator(
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            base_channels=config.base_channels,
            norm=config.norm,
            dropout=config.dropout,
            bilinear=config.bilinear,
        )
    raise ValueError(f"Unknown generator name: {config.name!r}")


def build_discriminator(config: DiscriminatorConfig) -> nn.Module:
    if config.name == "patchgan":
        return PatchGANDiscriminator(
            in_channels=config.in_channels,
            ndf=config.ndf,
            norm=config.norm,
            use_sigmoid=config.use_sigmoid,
        )
    raise ValueError(f"Unknown discriminator name: {config.name!r}")
