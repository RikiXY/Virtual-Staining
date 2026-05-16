from __future__ import annotations

import pytest
import torch

from virtual_staining.models.config import DiscriminatorConfig, GeneratorConfig, ModelConfig
from virtual_staining.models.discriminator import PatchGANDiscriminator
from virtual_staining.models.factory import build_discriminator, build_generator
from virtual_staining.models.generator import UNetGenerator


def test_model_config_defaults_match_previous_hardcoded_models() -> None:
    config = ModelConfig()

    generator = build_generator(config.generator)
    discriminator = build_discriminator(config.discriminator)

    assert isinstance(generator, UNetGenerator)
    assert generator.in_channels == 3
    assert generator.out_channels == 3
    assert generator.base_channels == 64
    assert generator.norm == "batch"
    assert generator.dropout is False
    assert generator.bilinear is False

    assert isinstance(discriminator, PatchGANDiscriminator)
    assert discriminator.in_channels == 6
    assert discriminator.ndf == 64
    assert discriminator.norm == "instance"
    assert discriminator.use_sigmoid is False


def test_build_generator_uses_configured_parameters() -> None:
    generator = build_generator(
        GeneratorConfig(
            in_channels=1,
            out_channels=2,
            base_channels=32,
            norm="instance",
            dropout=True,
            bilinear=False,
        )
    )

    assert isinstance(generator, UNetGenerator)
    assert generator.in_channels == 1
    assert generator.out_channels == 2
    assert generator.base_channels == 32
    assert generator.norm == "instance"
    assert generator.dropout is True
    assert generator.bilinear is False


def test_build_discriminator_uses_configured_parameters() -> None:
    discriminator = build_discriminator(
        DiscriminatorConfig(in_channels=4, ndf=32, norm="batch", use_sigmoid=True)
    )

    assert isinstance(discriminator, PatchGANDiscriminator)
    assert discriminator.in_channels == 4
    assert discriminator.ndf == 32
    assert discriminator.norm == "batch"
    assert discriminator.use_sigmoid is True


def test_build_generator_raises_on_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unknown generator name"):
        build_generator(GeneratorConfig(name="unknown"))  # type: ignore[arg-type]


def test_unet_generator_output_range_with_tanh() -> None:
    generator = build_generator(
        GeneratorConfig(in_channels=3, out_channels=3, base_channels=16, bilinear=False)
    )
    generator.eval()

    with torch.no_grad():
        output = generator(torch.randn(1, 3, 64, 64))

    assert output.shape == (1, 3, 64, 64)
    assert output.min().item() >= -1.0 - 1e-5
    assert output.max().item() <= 1.0 + 1e-5


def test_build_discriminator_raises_on_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unknown discriminator name"):
        build_discriminator(DiscriminatorConfig(name="unknown"))  # type: ignore[arg-type]
