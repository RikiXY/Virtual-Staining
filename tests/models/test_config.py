from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from tests.config_helpers import pix2pix_config_data
from virtual_staining.config.model import ModelConfig
from virtual_staining.config.run import RunConfig
from virtual_staining.definitions import ComponentContext
from virtual_staining.models.components import CONCAT_UNET, PATCHGAN
from virtual_staining.models.generator import ConcatUNetGenerator

_GENERATOR = ComponentContext(field="model.generator", image_size=(32, 32))
_DISCRIMINATOR = ComponentContext(field="model.discriminator", image_size=(32, 32))


def _resolve(**model: Any) -> RunConfig:
    data = pix2pix_config_data(Path("unused"), inputs=("LF",))
    data["model"] = {"inputs": ["LF"], "outputs": ["stained"], **model}
    return RunConfig.from_mapping(data)


@pytest.mark.parametrize("section", ["generator", "discriminator"])
@pytest.mark.parametrize("norm", ["batch", "instance"])
def test_builtin_components_accept_norm_choices(section: str, norm: str) -> None:
    assert _resolve(**{section: {"norm": norm}}).to_dict()["model"][section]["norm"] == norm


@pytest.mark.parametrize(
    ("section", "field"),
    [("generator", "norm"), ("discriminator", "norm")],
)
@pytest.mark.parametrize(("value", "error"), [("unknown", ValueError), (False, TypeError)])
def test_builtin_components_reject_invalid_choices(
    section: str, field: str, value: object, error: type[Exception]
) -> None:
    with pytest.raises(error, match=rf"model\.{section}\.{field} must be"):
        _resolve(**{section: {field: value}})


def test_generator_architecture_must_name_a_registered_component() -> None:
    with pytest.raises(ValueError, match="'unknown' is not a registered component"):
        _resolve(generator={"architecture": "unknown"})
    with pytest.raises(TypeError, match=r"model\.generator\.architecture must be a string"):
        _resolve(generator={"architecture": False})


def test_component_defaults_build_the_models() -> None:
    names = {"input_names": ("LF", "AF"), "output_names": ("PAS", "HE")}
    generator = CONCAT_UNET.resolve({}, _GENERATOR).build(**names)
    discriminator = PATCHGAN.resolve({}, _DISCRIMINATOR).build(**names)

    assert isinstance(generator, ConcatUNetGenerator)
    assert generator.input_names == ("LF", "AF")
    assert generator.output_names == ("PAS", "HE")
    assert generator.unet.in_channels == 6
    assert generator.unet.out_channels == 6
    assert discriminator.in_channels == 12


@pytest.mark.parametrize(
    ("mapping", "match"),
    [
        ({"outputs": ["stained"]}, "model requires inputs"),
        ({"inputs": ["LF"]}, "model requires outputs"),
        ({"inputs": ["LF"], "outputs": []}, "at least one name"),
        ({"inputs": "LF", "outputs": ["x"]}, "sequence of names"),
        ({"inputs": ["LF"], "outputs": ["HE", "HE"]}, "duplicate names"),
        ({"inputs": ["LF"], "outputs": ["LF"]}, "disjoint"),
        ({"inputs": ["LF"], "outputs": ["H&E"]}, "invalid identifiers"),
        ({"inputs": ["1LF"], "outputs": ["HE"]}, "invalid identifiers"),
        ({"inputs": ["LF"], "outputs": [3]}, "invalid identifiers"),
        ({"inputs": ["LF"], "target": "HE"}, "model.target is not part of the current schema"),
    ],
)
def test_model_config_rejects_invalid_io(mapping: dict[str, object], match: str) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        ModelConfig.from_mapping(mapping)


def test_model_config_keeps_the_authored_order_of_safe_identifiers() -> None:
    config = ModelConfig.from_mapping({"inputs": ["LF", "AF"], "outputs": ["PAS", "H-E_2"]})

    assert config.outputs == ("PAS", "H-E_2")
    assert config.to_dict() == {"inputs": ["LF", "AF"], "outputs": ["PAS", "H-E_2"]}


def test_model_config_is_only_the_named_io_contract() -> None:
    with pytest.raises(ValueError, match="Unknown key.*generator"):
        ModelConfig.from_mapping({"inputs": ["LF"], "outputs": ["stained"], "generator": {}})
    config = ModelConfig.from_mapping({"inputs": ["LF", "AF"], "outputs": ["stained"]})
    assert config.to_dict() == {"inputs": ["LF", "AF"], "outputs": ["stained"]}


@pytest.mark.parametrize(
    "model",
    [{"generator": {"in_channels": 3}}, {"discriminator": {"in_channels": 6}}],
)
def test_builtin_components_reject_unknown_options(model: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="Unknown key"):
        _resolve(**model)


def test_resolved_builtin_model_spelling() -> None:
    model = _resolve().to_dict()["model"]
    assert model["inputs"] == ["LF"]
    assert model["generator"]["architecture"] == "concat_unet"
    assert "in_channels" not in model["generator"]
    assert "in_channels" not in model["discriminator"]


def test_concat_generator_output_range_with_tanh() -> None:
    generator = ConcatUNetGenerator(("LF",), ("HE",), base_channels=16)
    generator.eval()
    with torch.no_grad():
        outputs = generator({"LF": torch.randn(1, 3, 64, 64)})
    assert list(outputs) == ["HE"]
    assert outputs["HE"].shape == (1, 3, 64, 64)
    assert outputs["HE"].min().item() >= -1.0 - 1e-5
    assert outputs["HE"].max().item() <= 1.0 + 1e-5
