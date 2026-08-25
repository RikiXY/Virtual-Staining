from __future__ import annotations

import pytest
import torch

from virtual_staining.config.model import ModelConfig
from virtual_staining.models.discriminator import PatchGANDiscriminator
from virtual_staining.models.generator import ConcatUNetGenerator


def test_model_config_defaults_match_models() -> None:
    config = ModelConfig.from_mapping({"inputs": ["LF", "AF"], "target": "stained"})
    generator = ConcatUNetGenerator(config.inputs, base_channels=config.generator.base_channels)
    PatchGANDiscriminator(
        in_channels=3 * len(config.inputs) + 3,
        ndf=config.discriminator.ndf,
        norm=config.discriminator.norm,
        use_sigmoid=config.discriminator.use_sigmoid,
    )
    assert generator.input_names == ("LF", "AF")
    assert generator.unet.in_channels == 6
    assert generator.unet.out_channels == 3


@pytest.mark.parametrize(
    "mapping",
    [
        {"target": "stained"},
        {"inputs": ["LF"], "target": ""},
        {"inputs": ["LF"], "target": "stained", "generator": {"architecture": "unet"}},
        {"inputs": ["LF"], "target": "stained", "generator": {"in_channels": 3}},
        {"inputs": ["LF"], "target": "stained", "discriminator": {"in_channels": 6}},
    ],
)
def test_model_config_rejects_removed_or_invalid_fields(mapping: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError), match="inputs|target|architecture|Unknown key"):
        ModelConfig.from_mapping(mapping)


def test_model_config_dict_has_named_contract() -> None:
    config = ModelConfig.from_mapping({"inputs": ["LF", "AF"], "target": "stained"}).to_dict()
    assert config["inputs"] == ["LF", "AF"]
    assert config["target"] == "stained"
    assert config["generator"]["architecture"] == "concat_unet"
    assert "in_channels" not in config["generator"]
    assert "in_channels" not in config["discriminator"]


def test_concat_generator_output_range_with_tanh() -> None:
    generator = ConcatUNetGenerator(("LF",), base_channels=16)
    generator.eval()
    with torch.no_grad():
        output = generator({"LF": torch.randn(1, 3, 64, 64)})
    assert output.shape == (1, 3, 64, 64)
    assert output.min().item() >= -1.0 - 1e-5
    assert output.max().item() <= 1.0 + 1e-5
