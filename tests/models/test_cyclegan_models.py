from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from virtual_staining.config.model import ModelConfig
from virtual_staining.methods.cyclegan import init_cyclegan_weights
from virtual_staining.methods.pix2pix import pix2pix_component_metadata
from virtual_staining.models.discriminator import PatchGANDiscriminator
from virtual_staining.models.factory import (
    build_discriminator,
    build_generator,
    build_resnet_generator,
)
from virtual_staining.models.generator import ResnetBlock, ResnetGenerator


def _resnet_config(**generator: object) -> ModelConfig:
    return ModelConfig.from_mapping(
        {
            "inputs": ["a"],
            "target": "b",
            "generator": {"architecture": "resnet", **generator},
            "discriminator": {"ndf": 4},
        }
    )


def test_resnet_generator_maps_rgb_to_bounded_rgb_of_same_size() -> None:
    generator = build_resnet_generator(_resnet_config(base_channels=4, blocks=1))
    x = torch.randn(2, 3, 32, 24) * 10

    y = generator(x)

    assert y.shape == x.shape
    assert y.abs().max() <= 1.0


def test_resnet_generator_defaults_to_nine_blocks() -> None:
    generator = build_resnet_generator(_resnet_config(base_channels=4))

    assert generator.blocks == 9
    assert sum(isinstance(module, ResnetBlock) for module in generator.modules()) == 9
    assert any(isinstance(module, nn.ReflectionPad2d) for module in generator.modules())
    assert not any(isinstance(module, nn.BatchNorm2d) for module in generator.modules())


def test_resnet_generator_construction_is_deterministic_under_torch_seed() -> None:
    config = _resnet_config(base_channels=4, blocks=1)
    torch.manual_seed(3)
    first = build_resnet_generator(config).state_dict()
    torch.manual_seed(3)
    second = build_resnet_generator(config).state_dict()

    assert all(torch.equal(first[key], second[key]) for key in first)


def test_resnet_generator_rejects_invalid_block_count() -> None:
    with pytest.raises(ValueError, match="at least one residual block"):
        ResnetGenerator(blocks=0)


@pytest.mark.parametrize(
    ("shape", "match"),
    [((1, 1, 32, 32), "3 channels"), ((1, 3, 30, 32), "multiples of 4"), ((3, 32, 32), "NCHW")],
)
def test_resnet_generator_rejects_invalid_shapes(shape: tuple[int, ...], match: str) -> None:
    generator = ResnetGenerator(base_channels=4, blocks=1)
    with pytest.raises(ValueError, match=match):
        generator(torch.zeros(shape))


def test_patchgan_conditional_and_unconditional_inputs() -> None:
    config = ModelConfig.from_mapping(
        {"inputs": ["a", "c"], "target": "b", "discriminator": {"ndf": 4}}
    )
    conditional = build_discriminator(config)
    unconditional = build_discriminator(config, conditional=False)

    assert conditional.in_channels == 9
    assert unconditional.in_channels == 3
    condition, image = torch.randn(2, 6, 32, 32), torch.randn(2, 3, 32, 32)
    assert conditional(condition, image).shape == (2, 1, 2, 2)
    assert unconditional(image).shape == (2, 1, 2, 2)


def test_patchgan_conditional_forward_is_unchanged() -> None:
    torch.manual_seed(0)
    discriminator = PatchGANDiscriminator(in_channels=6, ndf=4)
    x, y = torch.randn(1, 3, 32, 32), torch.randn(1, 3, 32, 32)

    assert torch.equal(discriminator(x, y), discriminator.model(torch.cat([x, y], dim=1)))


def test_cyclegan_initializer_is_applied_to_convs_and_affine_norms() -> None:
    torch.manual_seed(0)
    module = nn.Sequential(
        nn.Conv2d(3, 64, 3),
        nn.ConvTranspose2d(64, 64, 3),
        nn.BatchNorm2d(64),
        nn.InstanceNorm2d(64),
    )
    for layer in module:
        for parameter in layer.parameters():
            nn.init.constant_(parameter, 5.0)

    init_cyclegan_weights(module)

    conv, transpose, norm = module[0], module[1], module[2]
    assert isinstance(conv, nn.Conv2d) and isinstance(transpose, nn.ConvTranspose2d)
    assert isinstance(norm, nn.BatchNorm2d)
    for layer in (conv, transpose):
        assert abs(layer.weight.mean().item()) < 0.01
        assert 0.015 < layer.weight.std().item() < 0.025
        assert layer.bias is not None and torch.count_nonzero(layer.bias) == 0
    assert abs(norm.weight.mean().item() - 1.0) < 0.02
    assert norm.weight.std().item() < 0.05
    assert torch.count_nonzero(norm.bias) == 0


def test_pix2pix_component_metadata_is_unaffected_by_resnet_fields() -> None:
    config = ModelConfig.from_mapping({"inputs": ["a"], "target": "b"})

    assert build_generator(config).unet.base_channels == 64
    assert pix2pix_component_metadata(config)["generator"] == {
        "class": "ConcatUNetGenerator",
        "output_activation": "tanh",
        "architecture": "concat_unet",
        "base_channels": 64,
        "norm": "batch",
        "dropout": False,
        "bilinear": False,
    }
