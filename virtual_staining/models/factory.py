from __future__ import annotations

from virtual_staining.config.model import ModelConfig
from virtual_staining.models.discriminator import PatchGANDiscriminator
from virtual_staining.models.generator import ConcatUNetGenerator, ResnetGenerator


def build_generator(config: ModelConfig) -> ConcatUNetGenerator:
    generator = config.generator
    return ConcatUNetGenerator(
        config.inputs,
        base_channels=generator.base_channels,
        norm=generator.norm,
        dropout=generator.dropout,
        bilinear=generator.bilinear,
    )


def build_resnet_generator(config: ModelConfig) -> ResnetGenerator:
    generator = config.generator
    if generator.architecture != "resnet" or generator.blocks is None:
        raise ValueError("build_resnet_generator requires model.generator.architecture='resnet'")
    return ResnetGenerator(base_channels=generator.base_channels, blocks=generator.blocks)


def build_discriminator(config: ModelConfig, *, conditional: bool = True) -> PatchGANDiscriminator:
    """Build a PatchGAN; conditional input is inputs+target, unconditional is one RGB image."""
    discriminator = config.discriminator
    return PatchGANDiscriminator(
        in_channels=(3 * len(config.inputs)) + 3 if conditional else 3,
        ndf=discriminator.ndf,
        norm=discriminator.norm,
        use_sigmoid=discriminator.use_sigmoid,
    )
