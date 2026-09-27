from __future__ import annotations

import dataclasses
import math
from types import MappingProxyType
from typing import Any

import pytest

import virtual_staining.config.losses as config_losses
from virtual_staining.config.losses import LossScheduleConfig, LossTermConfig, parse_loss_config
from virtual_staining.loss_definitions import LOSS_DEFINITIONS

_ROLES = {
    "adversarial_bce": {"generator", "discriminator"},
    "l1": {"generator"},
    "ssim": {"generator"},
    "adversarial_lsgan": {"generator", "discriminator"},
    "cycle_l1": {"generator"},
    "identity_l1": {"generator"},
}


def _parse(role: str = "generator", **term: Any) -> Any:
    return parse_loss_config({role: [{"name": "l1", "weight": 1.0, **term}]})


@pytest.mark.parametrize("name", list(_ROLES))
@pytest.mark.parametrize("role", ["generator", "discriminator"])
def test_every_loss_accepts_only_its_roles(name: str, role: str) -> None:
    if role in _ROLES[name]:
        assert getattr(_parse(role, name=name), role)[0].name == name
    else:
        with pytest.raises(ValueError, match=f"loss '{name}' is supported only in losses.gen"):
            _parse(role, name=name)


def test_config_validation_uses_canonical_definitions(monkeypatch: pytest.MonkeyPatch) -> None:
    l1 = LOSS_DEFINITIONS["l1"]
    patched = {
        **LOSS_DEFINITIONS,
        "l1": dataclasses.replace(l1, roles=frozenset({"discriminator"}), param_keys=frozenset()),
    }
    monkeypatch.setattr(config_losses, "LOSS_DEFINITIONS", MappingProxyType(patched))

    assert _parse("discriminator").discriminator[0].name == "l1"
    with pytest.raises(ValueError, match="supported only in losses.discriminator"):
        _parse()
    with pytest.raises(ValueError, match="Unknown key.*reduction"):
        _parse("discriminator", params={"reduction": "mean"})


@pytest.mark.parametrize(
    ("raw", "error", "match"),
    [
        ({"generator": [{"name": "l2", "weight": 1.0}]}, ValueError, "name must be one of"),
        (
            {"generator": [{"name": "l1", "weight": 1.0}, {"name": "l1", "weight": 2.0}]},
            ValueError,
            "Duplicate loss name",
        ),
        ({"generator": [{"name": "l1", "weight": 1.0, "scale": 2}]}, ValueError, "Unknown key"),
        (
            {"generator": [{"name": "l1", "weight": 1.0, "params": {"alpha": 1}}]},
            ValueError,
            r"Unknown key\(s\) in loss 'l1' params: alpha",
        ),
        ({"generator": [{"name": "l1", "weight": -0.1}]}, ValueError, "greater than or equal"),
        ({"generator": [{"name": "l1", "weight": math.nan}]}, ValueError, "finite"),
        ({"generator": [{"name": "l1", "weight": math.inf}]}, ValueError, "finite"),
        ({"generator": [{"name": "l1", "weight": "-inf"}]}, ValueError, "finite"),
    ],
)
def test_loss_terms_reject_invalid_configuration(
    raw: dict[str, Any], error: type[Exception], match: str
) -> None:
    with pytest.raises(error, match=match):
        parse_loss_config(raw)


def test_direct_term_validation_rejects_unknown_name_and_non_finite_weight() -> None:
    with pytest.raises(ValueError, match="loss name must be one of"):
        LossTermConfig(name="l2", weight=1.0).validate("generator")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="weight must be a finite number"):
        LossTermConfig(name="l1", weight=math.nan).validate("generator")


@pytest.mark.parametrize(
    ("schedule", "match"),
    [
        ({"type": "step", "epoch": 2, "factor": math.nan}, "factor must be a finite"),
        ({"type": "step", "epoch": 2, "factor": math.inf}, "factor must be a finite"),
        ({"type": "step", "epoch": 2, "factor": -0.5}, "factor must be greater"),
        ({"type": "step", "epoch": math.inf}, "epoch must be a finite"),
        ({"type": "cosine", "end_epoch": math.nan}, "end_epoch must be a finite"),
        ({"type": "cosine", "start_epoch": -math.inf, "end_epoch": 2}, "start_epoch must be"),
        ({"type": "cosine", "start_epoch": -1, "end_epoch": 2}, "start_epoch must be greater"),
        ({"type": "linear_warmup", "start_epoch": 3, "end_epoch": 2}, "end_epoch must be"),
        ({"type": "linear_decay"}, "requires end_epoch"),
        ({"type": "turn_on_after_epoch"}, "requires epoch"),
        ({"type": "turn_off_after_epoch", "epoch": -1}, "epoch must be greater"),
        ({"type": "exponential"}, "type must be one of"),
    ],
)
def test_schedules_reject_invalid_and_non_finite_values(
    schedule: dict[str, Any], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        _parse(schedule=schedule)


@pytest.mark.parametrize(
    ("params", "match"),
    [
        ({"mask": {"foreground_weight": math.nan}}, "foreground_weight must be a finite"),
        ({"mask": {"background_weight": -math.inf}}, "background_weight must be a finite"),
        ({"mask": {"background_weight": -1}}, "greater than or equal to 0"),
        ({"reduction": "max"}, "reduction must be one of"),
    ],
)
def test_l1_params_reject_invalid_values(params: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _parse(params=params)


@pytest.mark.parametrize(
    ("params", "match"),
    [
        ({"data_range": math.nan}, "data_range must be a finite"),
        ({"data_range": -1}, "data_range must be greater than 0"),
        ({"sigma": math.inf}, "sigma must be a finite"),
        ({"sigma": 0}, "sigma must be greater than 0"),
        ({"window_size": 10}, "positive odd integer"),
        ({"window_size": math.nan}, "window_size must be a finite"),
        ({"channel_mode": "lab"}, "channel_mode must be one of"),
        ({"mask": {"foreground_weight": math.inf}}, "foreground_weight must be a finite"),
    ],
)
def test_ssim_params_reject_invalid_values(params: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _parse(name="ssim", params=params)


def _multipliers(schedule: LossScheduleConfig, epochs: range) -> list[float]:
    schedule.validate()
    return [schedule.multiplier(epoch=epoch, global_step=99) for epoch in epochs]


def test_constant_schedule_is_always_one() -> None:
    assert _multipliers(LossScheduleConfig(), range(4)) == [1.0] * 4


def test_linear_schedules_boundaries() -> None:
    warmup = LossScheduleConfig(type="linear_warmup", start_epoch=2, end_epoch=6)
    decay = LossScheduleConfig(type="linear_decay", start_epoch=2, end_epoch=6)
    assert _multipliers(warmup, range(8)) == [0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0]
    assert _multipliers(decay, range(8)) == [1.0, 1.0, 1.0, 0.75, 0.5, 0.25, 0.0, 0.0]


def test_cosine_schedule_interpolation() -> None:
    cosine = LossScheduleConfig(type="cosine", start_epoch=1, end_epoch=5)
    expected = [1.0, 1.0] + [0.5 * (1 + math.cos(math.pi * p)) for p in (0.25, 0.5, 0.75)]
    assert _multipliers(cosine, range(7)) == pytest.approx([*expected, 0.0, 0.0])


def test_epoch_triggered_schedules_switch_at_epoch() -> None:
    step = LossScheduleConfig(type="step", epoch=3, factor=0.1)
    on = LossScheduleConfig(type="turn_on_after_epoch", epoch=3)
    off = LossScheduleConfig(type="turn_off_after_epoch", epoch=3)
    assert _multipliers(step, range(2, 5)) == [1.0, 0.1, 0.1]
    assert _multipliers(on, range(2, 5)) == [0.0, 1.0, 1.0]
    assert _multipliers(off, range(2, 5)) == [1.0, 0.0, 0.0]


@pytest.mark.parametrize(
    ("schedule_type", "expected"),
    [
        ("linear_warmup", [0.0, 0.0, 1.0]),
        ("linear_decay", [1.0, 1.0, 0.0]),
        ("cosine", [1.0, 1.0, 0.0]),
    ],
)
def test_equal_start_and_end_epoch_jumps_after_the_boundary(
    schedule_type: str, expected: list[float]
) -> None:
    schedule = LossScheduleConfig(type=schedule_type, start_epoch=3, end_epoch=3)  # type: ignore[arg-type]
    assert _multipliers(schedule, range(2, 5)) == expected


def test_disabled_and_zero_weight_terms_are_inactive_but_configured() -> None:
    config = parse_loss_config(
        {
            "generator": [
                {"name": "l1", "weight": 2.0, "enabled": False},
                {"name": "ssim", "weight": 0.0},
                {"name": "adversarial_bce", "weight": 1.0},
            ]
        }
    )
    disabled, zero, active = config.generator
    assert [term.name for term in config.active_generator] == ["adversarial_bce"]
    assert (disabled.is_active, zero.is_active, active.is_active) == (False, False, True)
    assert disabled.current_weight(epoch=5) == 0.0
    assert zero.current_weight(epoch=5) == 0.0
    assert config.to_dict()["generator"][0]["enabled"] is False
