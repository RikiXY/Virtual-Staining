from __future__ import annotations

from virtual_staining.config.model import ModelConfig
from virtual_staining.models.discriminator import PatchGANDiscriminator
from virtual_staining.models.generator import ConcatUNetGenerator


def build_generator(config: ModelConfig) -> ConcatUNetGenerator:
    generator = config.generator
    return ConcatUNetGenerator(
        config.inputs,
        base_channels=generator.base_channels,
        norm=generator.norm,
        dropout=generator.dropout,
        bilinear=generator.bilinear,
    )


def build_discriminator(config: ModelConfig) -> PatchGANDiscriminator:
    discriminator = config.discriminator
    return PatchGANDiscriminator(
        in_channels=(3 * len(config.inputs)) + 3,
        ndf=discriminator.ndf,
        norm=discriminator.norm,
        use_sigmoid=discriminator.use_sigmoid,
    )
