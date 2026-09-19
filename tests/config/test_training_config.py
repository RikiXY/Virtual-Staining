from __future__ import annotations

import pytest

from virtual_staining.config.training import TrainingConfig


def _mapping(**overrides: object) -> dict[str, object]:
    data: dict[str, object] = {
        "epochs": 100,
        "losses": {
            "generator": [
                {"name": "adversarial_bce", "weight": 1.0},
                {"name": "l1", "weight": 25.0},
            ],
            "discriminator": [{"name": "adversarial_bce", "weight": 1.0}],
        },
    }
    data.update(overrides)
    return data


def test_training_owns_nested_configs_and_round_trips() -> None:
    config = TrainingConfig.from_mapping(
        _mapping(
            scheduler={"name": "linear_decay", "decay_start_epoch": 50},
            early_stopping={"monitor": "val_ssim", "patience": 10},
            augmentation={"enabled": True, "expansion_factor": 3, "intensity": "medium"},
        )
    )

    assert config.scheduler.name == "linear_decay"
    assert config.early_stopping is not None
    assert config.augmentation.effective_expansion_factor == 3
    assert config.losses.generator[1].name == "l1"
    assert TrainingConfig.from_mapping(config.to_dict()) == config


def test_training_resolves_defaults() -> None:
    data = TrainingConfig.from_mapping(_mapping()).to_dict()
    assert data["batch_size"] == 8
    assert data["scheduler"] == {"name": "none"}
    assert data["augmentation"] == {
        "enabled": False,
        "expansion_factor": 1,
        "intensity": "light",
    }


@pytest.mark.parametrize("name", ["none", "linear_decay", "reduce_on_plateau"])
def test_training_accepts_scheduler_choices(name: str) -> None:
    config = TrainingConfig.from_mapping(
        _mapping(scheduler={"name": name, "decay_start_epoch": 50})
    )
    assert config.scheduler.name == name


@pytest.mark.parametrize("intensity", ["light", "medium", "strong"])
def test_training_accepts_augmentation_choices(intensity: str) -> None:
    config = TrainingConfig.from_mapping(_mapping(augmentation={"intensity": intensity}))
    assert config.augmentation.intensity == intensity


@pytest.mark.parametrize("mode", ["min", "max"])
def test_training_accepts_checkpoint_modes(mode: str) -> None:
    config = TrainingConfig.from_mapping(
        _mapping(
            scheduler={"name": "reduce_on_plateau", "mode": mode}, early_stopping={"mode": mode}
        )
    )
    assert config.scheduler.mode == mode
    assert config.early_stopping is not None and config.early_stopping.mode == mode


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
        TrainingConfig.from_mapping(_mapping(**{section: {field: value}}))


@pytest.mark.parametrize("name", ["adversarial_bce", "l1", "ssim"])
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
def test_training_accepts_loss_and_schedule_choices(name: str, schedule_type: str) -> None:
    config = TrainingConfig.from_mapping(
        _mapping(
            losses={
                "generator": [
                    {
                        "name": name,
                        "weight": 1.0,
                        "schedule": {"type": schedule_type, "end_epoch": 10, "epoch": 5},
                    }
                ]
            }
        )
    )
    term = config.losses.generator[0]
    assert term.name == name and term.schedule.type == schedule_type


@pytest.mark.parametrize("field", ["name", "schedule"])
@pytest.mark.parametrize(("value", "error"), [("unknown", ValueError), (False, TypeError)])
def test_training_rejects_invalid_loss_choices(
    field: str, value: object, error: type[Exception]
) -> None:
    term = {"name": "l1", "weight": 1.0, field: {"type": value} if field == "schedule" else value}
    with pytest.raises(error, match=r"losses\.generator\[0\]\.(name|schedule.type) must be"):
        TrainingConfig.from_mapping(_mapping(losses={"generator": [term]}))


def test_training_requires_epochs_and_losses() -> None:
    with pytest.raises(ValueError, match="epochs"):
        TrainingConfig.from_mapping({"losses": {}})
    with pytest.raises(ValueError, match="losses"):
        TrainingConfig.from_mapping({"epochs": 1})


@pytest.mark.parametrize("legacy_key", ["lr_schedule", "decay_start_epoch"])
def test_training_rejects_legacy_scheduler_keys(legacy_key: str) -> None:
    with pytest.raises(ValueError, match=legacy_key):
        TrainingConfig.from_mapping(_mapping(**{legacy_key: "linear_decay"}))


def test_training_rejects_top_level_unknown_key() -> None:
    with pytest.raises(ValueError, match="unexpected"):
        TrainingConfig.from_mapping(_mapping(unexpected=True))


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
        TrainingConfig.from_mapping(_mapping(scheduler=scheduler))


def test_loss_term_schedule_and_mask_are_preserved() -> None:
    config = TrainingConfig.from_mapping(
        _mapping(
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
    )

    term = config.losses.generator[0]
    assert term.requires_mask is True
    assert term.current_weight(epoch=5) == pytest.approx(1.25)
    assert config.to_dict()["losses"]["generator"][0]["weight"] == 2.5


def test_strict_augmentation_boolean_is_preserved() -> None:
    with pytest.raises(TypeError, match="YAML boolean"):
        TrainingConfig.from_mapping(_mapping(augmentation={"enabled": "false"}))
