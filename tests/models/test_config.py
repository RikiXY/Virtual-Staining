from __future__ import annotations

import pytest
import torch

from virtual_staining.models.config import DiscriminatorConfig, GeneratorConfig, ModelConfig
from virtual_staining.models.discriminator import PatchGANDiscriminator
from virtual_staining.models.generator import UNetGenerator


def test_model_config_defaults_match_models() -> None:
    config = ModelConfig()

    generator = UNetGenerator(
        in_channels=config.generator.in_channels,
        out_channels=config.generator.out_channels,
        base_channels=config.generator.base_channels,
        norm=config.generator.norm,
        dropout=config.generator.dropout,
        bilinear=config.generator.bilinear,
    )
    discriminator = PatchGANDiscriminator(
        in_channels=config.discriminator.in_channels,
        ndf=config.discriminator.ndf,
        norm=config.discriminator.norm,
        use_sigmoid=config.discriminator.use_sigmoid,
    )

    assert generator.in_channels == 3
    assert generator.out_channels == 3
    assert generator.base_channels == 64
    assert generator.norm == "batch"
    assert generator.dropout is False
    assert generator.bilinear is False
    assert discriminator.in_channels == 6
    assert discriminator.ndf == 64
    assert discriminator.norm == "instance"
    assert discriminator.use_sigmoid is False


def test_models_use_configured_parameters() -> None:
    generator_config = GeneratorConfig(
        in_channels=1,
        out_channels=2,
        base_channels=32,
        norm="instance",
        dropout=True,
    )
    generator = UNetGenerator(
        in_channels=generator_config.in_channels,
        out_channels=generator_config.out_channels,
        base_channels=generator_config.base_channels,
        norm=generator_config.norm,
        dropout=generator_config.dropout,
        bilinear=generator_config.bilinear,
    )
    discriminator_config = DiscriminatorConfig(in_channels=4, ndf=32, norm="batch")
    discriminator = PatchGANDiscriminator(
        in_channels=discriminator_config.in_channels,
        ndf=discriminator_config.ndf,
        norm=discriminator_config.norm,
        use_sigmoid=discriminator_config.use_sigmoid,
    )

    assert generator.in_channels == 1
    assert generator.out_channels == 2
    assert generator.base_channels == 32
    assert generator.norm == "instance"
    assert generator.dropout is True
    assert discriminator.in_channels == 4
    assert discriminator.ndf == 32
    assert discriminator.norm == "batch"


@pytest.mark.parametrize(
    "mapping",
    [
        {"name": "pix2pix"},
        {"generator": {"name": "unet"}},
        {"discriminator": {"name": "patchgan"}},
    ],
)
def test_model_config_rejects_removed_names(mapping: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="Unknown key"):
        ModelConfig.from_mapping(mapping)


def test_model_config_dict_omits_architecture_names() -> None:
    config = ModelConfig().to_dict()

    assert "name" not in config
    assert "name" not in config["generator"]
    assert "name" not in config["discriminator"]


def test_unet_generator_output_range_with_tanh() -> None:
    generator = UNetGenerator(base_channels=16)
    generator.eval()

    with torch.no_grad():
        output = generator(torch.randn(1, 3, 64, 64))

    assert output.shape == (1, 3, 64, 64)
    assert output.min().item() >= -1.0 - 1e-5
    assert output.max().item() <= 1.0 + 1e-5
