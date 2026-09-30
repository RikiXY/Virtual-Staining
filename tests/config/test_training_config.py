"""Generic ``training.*`` lifecycle keys plus the Pix2Pix-owned optimization keys.

Both are resolved through ``RunConfig.from_mapping``: the framework parses the lifecycle
keys and the selected method definition parses the keys it owns.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, get_args

import pytest

from tests.config_helpers import pix2pix_config_data
from virtual_staining.config.losses import LossName, parse_loss_config
from virtual_staining.config.run import RunConfig
from virtual_staining.config.training import TrainingConfig
from virtual_staining.loss_definitions import LOSS_DEFINITIONS
from virtual_staining.methods.builtin import GanTrainingOptions, Pix2PixDefinition

_LOSSES = {
    "generator": [
        {"name": "adversarial_bce", "weight": 1.0},
        {"name": "l1", "weight": 25.0},
    ],
    "discriminator": [{"name": "adversarial_bce", "weight": 1.0}],
}


def _resolve(**overrides: object) -> RunConfig:
    data = pix2pix_config_data(Path("unused"))
    data["training"] = {"epochs": 100, "losses": _LOSSES, **overrides}
    return RunConfig.from_mapping(data)


def _training(**overrides: object) -> TrainingConfig:
    config = _resolve(**overrides)
    assert config.training is not None
    return config.training


def _optimization(**overrides: object) -> GanTrainingOptions:
    optimization = _resolve(**overrides).method.options.training
    assert optimization is not None
    return optimization


def test_training_sections_round_trip() -> None:
    config = _resolve(
        scheduler={"name": "linear_decay", "decay_start_epoch": 50},
        early_stopping={"monitor": "val_ssim__stained", "patience": 10},
        augmentation={"enabled": True, "expansion_factor": 3, "intensity": "medium"},
    )
    training = config.training
    optimization = config.method.options.training

    assert optimization.scheduler.name == "linear_decay"
    assert training is not None and training.early_stopping is not None
    assert training.augmentation.effective_expansion_factor == 3
    assert optimization.losses.generator[1].name == "l1"
    data = pix2pix_config_data(Path("unused"))
    data["training"] = config.to_dict()["training"]
    assert RunConfig.from_mapping(data).to_dict() == config.to_dict()


def test_generic_training_config_holds_only_lifecycle_fields() -> None:
    assert set(_training().to_dict()) == {
        "batch_size",
        "epochs",
        "num_workers",
        "validate_rate",
        "checkpoint_rate",
        "checkpoint_top_k",
        "log_rate",
        "augmentation",
    }


def test_training_resolves_defaults() -> None:
    data = _resolve().to_dict()["training"]
    assert data["batch_size"] == 8
    assert data["scheduler"] == {"name": "none"}
    assert data["augmentation"] == {
        "enabled": False,
        "expansion_factor": 1,
        "intensity": "light",
        "photometric_inputs": [],
    }


@pytest.mark.parametrize("name", ["none", "linear_decay", "reduce_on_plateau"])
def test_training_accepts_scheduler_choices(name: str) -> None:
    optimization = _optimization(scheduler={"name": name, "decay_start_epoch": 50})
    assert optimization.scheduler.name == name


@pytest.mark.parametrize("intensity", ["light", "medium", "strong"])
def test_training_accepts_augmentation_choices(intensity: str) -> None:
    assert _training(augmentation={"intensity": intensity}).augmentation.intensity == intensity


@pytest.mark.parametrize("mode", ["min", "max"])
def test_training_accepts_checkpoint_modes(mode: str) -> None:
    config = _resolve(
        scheduler={"name": "reduce_on_plateau", "mode": mode}, early_stopping={"mode": mode}
    )
    assert config.method.options.training.scheduler.mode == mode
    assert config.training is not None and config.training.early_stopping is not None
    assert config.training.early_stopping.mode == mode


def test_early_stopping_default_monitor_and_mode_come_from_the_method() -> None:
    early_stopping = _training(early_stopping={}).early_stopping

    assert early_stopping is not None
    assert early_stopping.monitor == "val_ssim__stained"
    assert Pix2PixDefinition().resolve_default_monitor(("stained",)) == early_stopping.monitor
    assert early_stopping.mode == "max"


def test_several_outputs_require_an_explicit_early_stopping_monitor() -> None:
    data = pix2pix_config_data(Path("unused"), outputs=("HE", "PAS"))
    data["training"] = {"epochs": 10, "losses": _LOSSES, "early_stopping": {}}
    with pytest.raises(ValueError, match="early_stopping.monitor is required"):
        RunConfig.from_mapping(data)

    data["training"]["early_stopping"] = {"monitor": "val_mae__PAS"}
    config = RunConfig.from_mapping(data)
    assert config.training is not None and config.training.early_stopping is not None
    assert config.training.early_stopping.mode == "min"

    data["training"]["early_stopping"] = {"monitor": "val_mae__XX"}
    with pytest.raises(ValueError, match="names output 'XX'"):
        RunConfig.from_mapping(data)


def _augmentation(inputs: tuple[str, ...], **augmentation: object) -> tuple[str, ...] | None:
    data = pix2pix_config_data(Path("unused"), inputs=inputs)
    data["training"] = {"epochs": 10, "losses": _LOSSES, "augmentation": augmentation}
    training = RunConfig.from_mapping(data).training
    assert training is not None
    return training.augmentation.photometric_inputs


def test_photometric_inputs_resolve_to_the_effective_list() -> None:
    # light has no photometric transforms.
    assert _augmentation(("LF", "AF"), intensity="light") == ()
    # Without a preparation reference the first selected input is the default.
    assert _augmentation(("AF", "LF"), intensity="medium") == ("AF",)
    assert _augmentation(("LF", "AF"), intensity="strong", photometric_inputs=[]) == ()
    assert _augmentation(("LF", "AF"), intensity="medium", photometric_inputs=["LF"]) == ("LF",)
    assert _augmentation(("LF", "AF"), intensity="medium", photometric_inputs=["AF", "LF"]) == (
        "AF",
        "LF",
    )


def test_photometric_inputs_default_to_the_selected_preparation_reference() -> None:
    def resolve(inputs: list[str]) -> tuple[str, ...] | None:
        data = pix2pix_config_data(Path("unused"), inputs=tuple(inputs))
        data["training"] = {
            "epochs": 10,
            "losses": _LOSSES,
            "augmentation": {"intensity": "medium"},
        }
        data["preprocessing"] = {
            "inputs": {
                "inventory": "i.csv",
                "modalities": ["LF", "AF"],
                "reference": "LF",
                "target_modalities": ["stained"],
            },
            "split": {"unit": "patch", "train": 0.8, "val": 0.1, "test": 0.1},
        }
        training = RunConfig.from_mapping(data).training
        assert training is not None
        resolved = training.augmentation.photometric_inputs
        assert RunConfig.from_mapping(data).to_dict()["training"]["augmentation"][
            "photometric_inputs"
        ] == list(resolved or ())
        return resolved

    assert resolve(["AF", "LF"]) == ("LF",)  # the reference is selected
    assert resolve(["AF"]) == ("AF",)  # reference absent -> first selected input


@pytest.mark.parametrize(
    ("augmentation", "message"),
    [
        ({"intensity": "light", "photometric_inputs": ["LF"]}, "must be empty for"),
        ({"intensity": "medium", "photometric_inputs": ["LF", "LF"]}, "duplicate names"),
        ({"intensity": "medium", "photometric_inputs": ["XX"]}, "not selected model.inputs"),
        ({"intensity": "medium", "photometric_inputs": ["stained"]}, "not selected model.inputs"),
        ({"intensity": "medium", "photometric_inputs": "LF"}, "list of input names"),
    ],
)
def test_photometric_inputs_are_validated(augmentation: dict[str, Any], message: str) -> None:
    with pytest.raises((ValueError, TypeError), match=message):
        _augmentation(("LF", "AF"), **augmentation)


def test_enabled_augmentation_requires_a_square_image_size() -> None:
    data = pix2pix_config_data(Path("unused"), image_size=(48, 32))
    data["training"] = {"epochs": 10, "losses": _LOSSES, "augmentation": {"enabled": True}}
    with pytest.raises(ValueError, match="requires a square image_size"):
        RunConfig.from_mapping(data)

    data["training"]["augmentation"] = {"enabled": False, "intensity": "medium"}
    assert RunConfig.from_mapping(data).project.image_size == (48, 32)


@pytest.mark.parametrize(
    ("section", "field"),
    [
        ("scheduler", "name"),
        ("scheduler", "mode"),
        ("early_stopping", "mode"),
        ("augmentation", "intensity"),
    ],
)
@pytest.mark.parametrize(("value", "error"), [("unknown", ValueError), (False, TypeError)])
def test_training_rejects_invalid_choices(
    section: str, field: str, value: object, error: type[Exception]
) -> None:
    with pytest.raises(error, match=rf"{section}\.{field} must be"):
        _resolve(**{section: {field: value}})


def test_static_loss_name_alias_matches_canonical_definitions() -> None:
    assert set(get_args(LossName)) == set(LOSS_DEFINITIONS)


@pytest.mark.parametrize("name", get_args(LossName))
@pytest.mark.parametrize(
    "schedule_type",
    [
        "constant",
        "linear_warmup",
        "linear_decay",
        "step",
        "cosine",
        "turn_on_after_epoch",
        "turn_off_after_epoch",
    ],
)
def test_loss_config_accepts_loss_and_schedule_choices(name: str, schedule_type: str) -> None:
    losses = parse_loss_config(
        {
            "generator": [
                {
                    "name": name,
                    "weight": 1.0,
                    "schedule": {"type": schedule_type, "end_epoch": 10, "epoch": 5},
                }
            ]
        }
    )
    term = losses.generator[0]
    assert term.name == name and term.schedule.type == schedule_type


@pytest.mark.parametrize("field", ["name", "schedule"])
@pytest.mark.parametrize(("value", "error"), [("unknown", ValueError), (False, TypeError)])
def test_training_rejects_invalid_loss_choices(
    field: str, value: object, error: type[Exception]
) -> None:
    term = {"name": "l1", "weight": 1.0, field: {"type": value} if field == "schedule" else value}
    with pytest.raises(error, match=r"losses\.generator\[0\]\.(name|schedule.type) must be"):
        _resolve(losses={"generator": [term]})


def test_training_requires_epochs_and_the_method_requires_losses() -> None:
    data: dict[str, Any] = pix2pix_config_data(Path("unused"))
    data["training"] = {"losses": {}}
    with pytest.raises(ValueError, match="epochs"):
        RunConfig.from_mapping(data)
    data["training"] = {"epochs": 1}
    with pytest.raises(ValueError, match="losses"):
        RunConfig.from_mapping(data)


@pytest.mark.parametrize("legacy_key", ["lr_schedule", "decay_start_epoch"])
def test_training_rejects_legacy_scheduler_keys(legacy_key: str) -> None:
    with pytest.raises(ValueError, match=legacy_key):
        _resolve(**{legacy_key: "linear_decay"})


def test_training_rejects_top_level_unknown_key() -> None:
    with pytest.raises(ValueError, match="unexpected"):
        _resolve(unexpected=True)


@pytest.mark.parametrize(
    "scheduler",
    [
        {"name": "linear_decay"},
        {"name": "linear_decay", "decay_start_epoch": 100},
        {"name": "reduce_on_plateau", "factor": 1.0},
    ],
)
def test_invalid_scheduler_is_rejected(scheduler: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        _resolve(scheduler=scheduler)


def test_loss_term_schedule_and_mask_are_preserved() -> None:
    config = _resolve(
        losses={
            "generator": [
                {
                    "name": "ssim",
                    "weight": 2.5,
                    "params": {
                        "mask": {
                            "enabled": True,
                            "source": "foreground_mask",
                            "background_weight": 0.25,
                        }
                    },
                    "schedule": {
                        "type": "linear_warmup",
                        "start_epoch": 0,
                        "end_epoch": 10,
                    },
                }
            ],
            "discriminator": [],
        }
    )

    term = config.method.options.training.losses.generator[0]
    assert term.requires_mask is True
    assert config.method.definition.requires_foreground_mask(config) is True
    assert term.current_weight(epoch=5) == pytest.approx(1.25)
    assert config.to_dict()["training"]["losses"]["generator"][0]["weight"] == 2.5


def test_strict_augmentation_boolean_is_preserved() -> None:
    with pytest.raises(TypeError, match="YAML boolean"):
        _resolve(augmentation={"enabled": "false"})
