"""Omitted training components never bypass the complete checkpoint identity check."""

from copy import deepcopy
from pathlib import Path

import pytest

from tests.config_helpers import cyclegan_config_data, pix2pix_config_data
from virtual_staining.checkpoint_contract import (
    CheckpointCompatibilityError,
    build_checkpoint_payload,
    validate_checkpoint,
)
from virtual_staining.config.run import RunConfig


@pytest.mark.parametrize("method", ["pix2pix", "cyclegan"])
def test_inference_restores_only_omitted_discriminator_identity(
    tmp_path: Path, method: str
) -> None:
    raw = (pix2pix_config_data if method == "pix2pix" else cyclegan_config_data)(tmp_path)
    trained = RunConfig.from_mapping(raw, stages=("train",))
    assert trained.method is not None
    definition = trained.method.definition
    payload = build_checkpoint_payload(
        definition.checkpoint_identity(trained), epoch=0, state={"models": {}}
    )
    raw.pop("training")
    raw["model"].pop("discriminator")
    raw["data"] = {"pairing": definition.pairing}
    raw["inference"] = {"checkpoint_policy": "latest"}
    config = RunConfig.from_mapping(raw, stages=("infer",))
    assert config.method is not None and config.method.options.discriminator is None
    path = tmp_path / "checkpoint.pth"
    identity = definition.inference_checkpoint_identity(config, payload, path)
    assert identity == definition.checkpoint_identity(trained)
    validate_checkpoint(payload, identity, path)
    # Resolving checkpoint metadata does not mutate the effective YAML or its identity.
    assert "discriminator" not in config.to_dict()["model"]

    role = "discriminator" if method == "pix2pix" else "D_A"
    for key, value in (
        ("version", "unsupported"),
        ("source", "other_provider"),
        ("name", "other_network"),
        ("options", {"unknown": True}),
    ):
        invalid = deepcopy(payload)
        invalid["method"]["components"][role][key] = value
        with pytest.raises(CheckpointCompatibilityError, match="components"):
            expected = definition.inference_checkpoint_identity(config, invalid, path)
            validate_checkpoint(invalid, expected, path)
    invalid = deepcopy(payload)
    del invalid["method"]["components"][role]
    with pytest.raises(CheckpointCompatibilityError, match="components"):
        definition.inference_checkpoint_identity(config, invalid, path)

    # Explicitly supplied discriminator options still constrain the checkpoint exactly.
    raw["model"]["discriminator"] = {"ndf": 8}
    explicit = RunConfig.from_mapping(raw, stages=("infer",))
    with pytest.raises(CheckpointCompatibilityError, match="ndf"):
        validate_checkpoint(
            payload, definition.inference_checkpoint_identity(explicit, payload, path), path
        )
