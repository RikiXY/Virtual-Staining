from __future__ import annotations

import copy
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from tests.config_helpers import cyclegan_config_data, write_config_data, write_run_config
from virtual_staining.config.run import RunConfig

Mutation = Callable[[dict[str, Any]], None]


def _load(tmp_path: Path, mutate: Mutation | None = None) -> RunConfig:
    data = cyclegan_config_data(tmp_path)
    if mutate is not None:
        mutate(data)
    return RunConfig.from_yaml(write_config_data(tmp_path / "run.yaml", data))


def _pix2pix(tmp_path: Path, extra: str = "") -> RunConfig:
    return RunConfig.from_yaml(
        write_run_config(
            tmp_path,
            f"""
model:
  inputs: [label_free]
  target: stained
training:
  epochs: 1
  losses:
    generator:
      - name: l1
        weight: 1.0
    discriminator: []
{extra}
""",
        )
    )


def test_pix2pix_config_without_data_resolves_as_paired(tmp_path: Path) -> None:
    config = _pix2pix(tmp_path)

    assert config.data.pairing == "paired"
    assert config.to_dict()["data"] == {"pairing": "paired"}
    assert config.to_dict()["method"] == {"name": "pix2pix"}
    assert config.to_dict()["model"]["generator"] == {
        "architecture": "concat_unet",
        "base_channels": 64,
        "norm": "batch",
        "dropout": False,
        "bilinear": False,
    }


def test_valid_cyclegan_config_resolves_defaults(tmp_path: Path) -> None:
    config = _load(tmp_path)

    resolved = config.to_dict()
    assert resolved["method"] == {"name": "cyclegan", "replay_buffer_size": 50}
    assert resolved["data"] == {
        "pairing": "unpaired",
        "domains": {"label_free": "domains/label_free", "stained": "domains/stained"},
    }
    assert resolved["model"]["generator"] == {
        "architecture": "resnet",
        "base_channels": 4,
        "norm": "instance",
        "blocks": 1,
    }
    assert RunConfig.from_yaml(write_config_data(tmp_path / "again.yaml", resolved)) == config


def test_resnet_generator_defaults_to_nine_instance_norm_blocks(tmp_path: Path) -> None:
    def mutate(data: dict[str, Any]) -> None:
        data["model"]["generator"] = {"architecture": "resnet"}

    generator = _load(tmp_path, mutate).model.generator
    assert (generator.blocks, generator.norm, generator.base_channels) == (9, "instance", 64)


def test_split_pattern_domain_form_is_accepted(tmp_path: Path) -> None:
    def mutate(data: dict[str, Any]) -> None:
        data["data"]["domains"]["stained"] = "prepared/{split}/stained/**/*.tif"

    assert _load(tmp_path, mutate).data.domains["stained"] == "prepared/{split}/stained/**/*.tif"


def test_replay_buffer_size_zero_is_allowed(tmp_path: Path) -> None:
    def mutate(data: dict[str, Any]) -> None:
        data["method"]["replay_buffer_size"] = 0

    assert _load(tmp_path, mutate).method.replay_buffer_size == 0


def _set(path: tuple[str | int, ...], value: object) -> Mutation:
    def mutate(data: dict[str, Any]) -> None:
        target: Any = data
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value

    return mutate


def _delete(path: tuple[str | int, ...]) -> Mutation:
    def mutate(data: dict[str, Any]) -> None:
        target: Any = data
        for key in path[:-1]:
            target = target[key]
        del target[path[-1]]

    return mutate


def _drop_loss(role: str, name: str) -> Mutation:
    def mutate(data: dict[str, Any]) -> None:
        losses = data["training"]["losses"]
        losses[role] = [term for term in losses[role] if term["name"] != name]

    return mutate


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (_set(("data",), {"pairing": "paired"}), "requires data.pairing='unpaired'"),
        (_set(("model", "inputs"), []), "model.inputs must be a non-empty"),
        (
            _set(("model", "inputs"), ["label_free", "other"]),
            "exactly one model.inputs entry",
        ),
        (_set(("model", "target"), "label_free"), "must name different domains"),
        (_delete(("data", "domains", "stained")), r"missing=\['stained'\]"),
        (_set(("data", "domains", "extra"), "domains/extra"), r"extra=\['extra'\]"),
        (
            _set(("model", "generator"), {"architecture": "concat_unet"}),
            "requires model.generator.architecture='resnet'",
        ),
        (_set(("model", "generator", "blocks"), 0), "blocks must be an integer >= 1"),
        (_set(("model", "generator", "blocks"), "many"), "blocks must be an integer"),
        (_set(("model", "generator", "norm"), "batch"), "norm must be 'instance'"),
        (_set(("model", "generator", "dropout"), True), "Unknown key.*dropout"),
        (_set(("image_size",), [30, 32]), "invalid for the resnet generator"),
        (_set(("image_size",), [4, 4]), "invalid for the resnet generator"),
        (
            _set(("training", "augmentation"), {"enabled": True}),
            "augmentation.enabled=false",
        ),
        (_drop_loss("generator", "adversarial_lsgan"), "generator term 'adversarial_lsgan'"),
        (_drop_loss("generator", "cycle_l1"), "generator term 'cycle_l1'"),
        (
            _drop_loss("discriminator", "adversarial_lsgan"),
            "discriminator term 'adversarial_lsgan'",
        ),
        (
            _set(("training", "losses", "generator", 1), {"name": "cycle_l1", "weight": 0.0}),
            "generator term 'cycle_l1'",
        ),
        (
            _set(("training", "losses", "generator", 2), {"name": "l1", "weight": 1.0}),
            r"\['l1'\] are not supported by method.name='cyclegan'",
        ),
        (
            _set(("training", "losses", "discriminator", 0), {"name": "cycle_l1", "weight": 1.0}),
            "supported only in losses.generator",
        ),
        (_set(("method", "replay_buffer_size"), -1), "replay_buffer_size must be >= 0"),
        (_set(("method", "replay_buffer_size"), "50"), "replay_buffer_size must be an integer"),
        (_set(("method", "class_path"), "pkg.mod:Method"), "Unknown key.*class_path"),
        (_set(("method", "params"), {}), "Unknown key.*params"),
        (_set(("data", "custom"), True), "Unknown key.*custom"),
        (_set(("data", "pairing"), "custom"), "data.pairing must be one of"),
        (_set(("inference", "direction"), "sideways"), "inference.direction must be one of"),
        (
            _set(("inference",), {"checkpoint_policy": "best", "checkpoint_metric": "val_ssim"}),
            "inference.checkpoint_metric='val_ssim' is a paired image-fidelity metric",
        ),
        (
            _set(("training", "early_stopping"), {"monitor": "val_ssim"}),
            "early_stopping.monitor='val_ssim'",
        ),
    ],
)
def test_invalid_cyclegan_configs_are_rejected(
    tmp_path: Path, mutate: Mutation, match: str
) -> None:
    with pytest.raises((ValueError, TypeError), match=match):
        _load(tmp_path, mutate)


def test_pix2pix_rejects_unpaired_data(tmp_path: Path) -> None:
    extra = "data:\n  pairing: unpaired\n  domains:\n    label_free: a\n    stained: b"
    with pytest.raises(ValueError, match="pix2pix' requires data.pairing='paired'"):
        _pix2pix(tmp_path, extra)


def test_paired_data_rejects_domains(tmp_path: Path) -> None:
    extra = "data:\n  pairing: paired\n  domains:\n    label_free: a"
    with pytest.raises(ValueError, match="data.domains is supported only with"):
        _pix2pix(tmp_path, extra)


def test_pix2pix_rejects_resnet_generator(tmp_path: Path) -> None:
    config = cyclegan_config_data(tmp_path)
    data = {key: config[key] for key in ("dataset_root", "results_path", "run_name")}
    data["model"] = copy.deepcopy(config["model"])
    with pytest.raises(ValueError, match="pix2pix' requires model.generator.architecture"):
        RunConfig.from_yaml(write_config_data(tmp_path / "run.yaml", data))


def test_pix2pix_rejects_cyclegan_only_method_settings(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="replay_buffer_size is supported only"):
        _pix2pix(tmp_path, "method:\n  name: pix2pix\n  replay_buffer_size: 50")


def test_pix2pix_rejects_concat_unet_incompatible_resnet_fields(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown key.*blocks"):
        RunConfig.from_yaml(
            write_run_config(
                tmp_path,
                "model:\n  inputs: [a]\n  target: b\n  generator:\n    blocks: 3",
            )
        )


def test_pix2pix_rejects_inference_direction(tmp_path: Path) -> None:
    extra = "inference:\n  checkpoint_policy: latest\n  direction: B_to_A"
    with pytest.raises(ValueError, match="inference.direction is supported only"):
        _pix2pix(tmp_path, extra)


def test_pix2pix_rejects_cyclegan_losses(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not supported by method.name='pix2pix'"):
        RunConfig.from_yaml(
            write_run_config(
                tmp_path,
                """
training:
  epochs: 1
  losses:
    generator:
      - name: cycle_l1
        weight: 1.0
""",
            )
        )
