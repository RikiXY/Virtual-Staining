from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal, cast, get_args

from virtual_staining.config.validation import (
    parse_bool_strict,
    parse_choice,
    parse_int,
    reject_unknown_keys,
    require_finite,
)
from virtual_staining.loss_definitions import (
    LOSS_DEFINITIONS,
    LossDefinition,
    LossMaskConfig,
    LossParams,
    LossRole,
)

_LOSS_CONFIG_KEYS: frozenset[str] = frozenset({"generator", "discriminator"})
_LOSS_TERM_KEYS: frozenset[str] = frozenset({"name", "weight", "enabled", "params", "schedule"})
_LOSS_SCHEDULE_KEYS: frozenset[str] = frozenset(
    {"type", "start_epoch", "end_epoch", "epoch", "factor"}
)

# Static typing alias only; runtime validation uses LOSS_DEFINITIONS.
LossName = Literal["adversarial_bce", "l1", "ssim", "adversarial_lsgan", "cycle_l1", "identity_l1"]
LossScheduleType = Literal[
    "constant",
    "linear_warmup",
    "linear_decay",
    "step",
    "cosine",
    "turn_on_after_epoch",
    "turn_off_after_epoch",
]


@dataclass(frozen=True)
class LossScheduleConfig:
    type: LossScheduleType = "constant"
    start_epoch: int = 0
    end_epoch: int | None = None
    epoch: int | None = None
    factor: float = 0.0

    def validate(self) -> None:
        valid = sorted(get_args(LossScheduleType))
        if self.type not in valid:
            raise ValueError(f"loss schedule type must be one of {valid}")
        if self.start_epoch < 0:
            raise ValueError("loss schedule start_epoch must be greater than or equal to 0")
        if self.end_epoch is not None and self.end_epoch < self.start_epoch:
            raise ValueError("loss schedule end_epoch must be greater than or equal to start_epoch")
        if self.epoch is not None and self.epoch < 0:
            raise ValueError("loss schedule epoch must be greater than or equal to 0")
        require_finite(self.factor, "loss schedule factor")
        if self.factor < 0:
            raise ValueError("loss schedule factor must be greater than or equal to 0")
        if self.type in {"linear_warmup", "linear_decay", "cosine"} and self.end_epoch is None:
            raise ValueError(f"loss schedule '{self.type}' requires end_epoch")
        if self.type in {"step", "turn_on_after_epoch", "turn_off_after_epoch"} and (
            self.epoch is None
        ):
            raise ValueError(f"loss schedule '{self.type}' requires epoch")

    def multiplier(self, *, epoch: int, global_step: int | None = None) -> float:
        del global_step
        if self.type == "constant":
            return 1.0
        if self.type == "turn_on_after_epoch":
            assert self.epoch is not None
            return 1.0 if epoch >= self.epoch else 0.0
        if self.type == "turn_off_after_epoch":
            assert self.epoch is not None
            return 0.0 if epoch >= self.epoch else 1.0
        if self.type == "step":
            assert self.epoch is not None
            return self.factor if epoch >= self.epoch else 1.0

        assert self.end_epoch is not None
        if epoch <= self.start_epoch:
            progress = 0.0
        elif epoch >= self.end_epoch:
            progress = 1.0
        else:
            progress = (epoch - self.start_epoch) / max(1, self.end_epoch - self.start_epoch)
        if self.type == "linear_warmup":
            return progress
        if self.type == "linear_decay":
            return 1.0 - progress
        if self.type == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        raise ValueError(f"Unsupported loss schedule type: {self.type!r}")

    def current_weight(
        self,
        base_weight: float,
        *,
        epoch: int,
        global_step: int | None = None,
    ) -> float:
        return base_weight * self.multiplier(epoch=epoch, global_step=global_step)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"type": self.type}
        if self.type in {"linear_warmup", "linear_decay", "cosine"}:
            data["start_epoch"] = self.start_epoch
            data["end_epoch"] = self.end_epoch
        if self.type in {"step", "turn_on_after_epoch", "turn_off_after_epoch"}:
            data["epoch"] = self.epoch
        if self.type == "step":
            data["factor"] = self.factor
        return data


@dataclass(frozen=True)
class LossTermConfig:
    name: LossName
    weight: float
    enabled: bool = True
    params: dict[str, Any] = field(default_factory=dict)
    schedule: LossScheduleConfig = field(default_factory=LossScheduleConfig)

    def validate(self, role: LossRole) -> None:
        self.definition.validate_role(role)
        require_finite(self.weight, f"loss '{self.name}' weight")
        if self.weight < 0:
            raise ValueError(f"loss '{self.name}' weight must be greater than or equal to 0")
        self.schedule.validate()
        self.resolved_params()

    @property
    def definition(self) -> LossDefinition:
        if self.name not in LOSS_DEFINITIONS:
            raise ValueError(f"loss name must be one of {sorted(LOSS_DEFINITIONS)}")
        return LOSS_DEFINITIONS[self.name]

    def resolved_params(self) -> LossParams:
        return self.definition.parse_params(self.params)

    @property
    def is_active(self) -> bool:
        return self.enabled and self.weight != 0.0

    def current_weight(self, *, epoch: int, global_step: int | None = None) -> float:
        if not self.enabled:
            return 0.0
        return self.schedule.current_weight(self.weight, epoch=epoch, global_step=global_step)

    @property
    def mask(self) -> LossMaskConfig:
        return self.resolved_params().mask

    @property
    def requires_mask(self) -> bool:
        return self.mask.enabled

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "weight": self.weight,
            "enabled": self.enabled,
            "params": dict(self.params),
            "schedule": self.schedule.to_dict(),
        }


@dataclass(frozen=True)
class LossConfig:
    generator: tuple[LossTermConfig, ...] = ()
    discriminator: tuple[LossTermConfig, ...] = ()

    def validate(self) -> None:
        _validate_unique_loss_names(self.generator, "generator")
        _validate_unique_loss_names(self.discriminator, "discriminator")
        for term in self.generator:
            term.validate("generator")
        for term in self.discriminator:
            term.validate("discriminator")

    @property
    def active_generator(self) -> tuple[LossTermConfig, ...]:
        return tuple(term for term in self.generator if term.is_active)

    @property
    def active_discriminator(self) -> tuple[LossTermConfig, ...]:
        return tuple(term for term in self.discriminator if term.is_active)

    def to_dict(self) -> dict[str, Any]:
        return {
            "generator": [term.to_dict() for term in self.generator],
            "discriminator": [term.to_dict() for term in self.discriminator],
        }


def configured_loss_names(losses: LossConfig | None) -> list[str]:
    """Role-qualified term names, the component columns of the training history."""
    if losses is None:
        return []
    names = [f"generator_{term.name}" for term in losses.generator]
    names.extend(f"discriminator_{term.name}" for term in losses.discriminator)
    return names


def parse_loss_config(raw: Any) -> LossConfig:
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise TypeError("losses must be a YAML mapping")
    reject_unknown_keys(raw, _LOSS_CONFIG_KEYS, "losses")
    config = LossConfig(
        generator=_parse_loss_terms(raw.get("generator", []), "losses.generator"),
        discriminator=_parse_loss_terms(raw.get("discriminator", []), "losses.discriminator"),
    )
    config.validate()
    return config


def _parse_loss_terms(raw: Any, context: str) -> tuple[LossTermConfig, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise TypeError(f"{context} must be a YAML list")
    return tuple(_parse_loss_term(item, f"{context}[{index}]") for index, item in enumerate(raw))


def _parse_loss_term(raw: Any, context: str) -> LossTermConfig:
    if not isinstance(raw, dict):
        raise TypeError(f"{context} must be a YAML mapping")
    reject_unknown_keys(raw, _LOSS_TERM_KEYS, context)
    if "name" not in raw:
        raise ValueError(f"{context}.name is required")
    if "weight" not in raw:
        raise ValueError(f"{context}.weight is required")

    name = parse_choice(raw["name"], f"{context}.name", set(LOSS_DEFINITIONS))
    params = raw.get("params", {})
    if params is None:
        params = {}
    if not isinstance(params, dict):
        raise TypeError(f"{context}.params must be a YAML mapping")
    return LossTermConfig(
        name=cast(LossName, name),
        weight=float(raw["weight"]),
        enabled=parse_bool_strict(raw.get("enabled", True), f"{context}.enabled"),
        params=dict(params),
        schedule=_parse_loss_schedule(raw.get("schedule", {}), f"{context}.schedule"),
    )


def _parse_loss_schedule(raw: Any, context: str) -> LossScheduleConfig:
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise TypeError(f"{context} must be a YAML mapping")
    reject_unknown_keys(raw, _LOSS_SCHEDULE_KEYS, context)
    schedule_type = parse_choice(
        raw.get("type", "constant"),
        f"{context}.type",
        set(get_args(LossScheduleType)),
    )
    config = LossScheduleConfig(
        type=cast(LossScheduleType, schedule_type),
        start_epoch=parse_int(raw.get("start_epoch", 0), f"{context}.start_epoch"),
        end_epoch=(
            parse_int(raw["end_epoch"], f"{context}.end_epoch")
            if raw.get("end_epoch") is not None
            else None
        ),
        epoch=parse_int(raw["epoch"], f"{context}.epoch") if raw.get("epoch") is not None else None,
        factor=float(raw.get("factor", 0.0)),
    )
    config.validate()
    return config


def _validate_unique_loss_names(terms: tuple[LossTermConfig, ...], role: str) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for term in terms:
        if term.name in seen:
            duplicates.add(term.name)
        seen.add(term.name)
    if duplicates:
        raise ValueError(
            f"Duplicate loss name(s) in losses.{role}: {', '.join(sorted(duplicates))}"
        )
