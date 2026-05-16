from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, cast

from virtual_staining.config.validation import parse_bool_strict, reject_unknown_keys

_TRAINING_KEYS: frozenset[str] = frozenset(
    {
        "batch_size",
        "epochs",
        "lr_g",
        "lr_d",
        "beta1",
        "beta2",
        "l1_weight",
        "seed",
        "num_workers",
        "validate_rate",
        "checkpoint_rate",
        "log_rate",
        "resume",
    }
)

_LOSS_CONFIG_KEYS: frozenset[str] = frozenset({"generator", "discriminator"})
_LOSS_TERM_KEYS: frozenset[str] = frozenset({"name", "weight", "enabled", "params", "schedule"})
_LOSS_SCHEDULE_KEYS: frozenset[str] = frozenset({"type"})
_SSIM_PARAM_KEYS: frozenset[str] = frozenset(
    {"data_range", "window_size", "channel_mode", "reduction"}
)

LossName = Literal["ssim"]
LossScheduleType = Literal["constant"]
LossRole = Literal["generator", "discriminator"]


@dataclass(frozen=True)
class LossScheduleConfig:
    type: LossScheduleType = "constant"

    def validate(self) -> None:
        if self.type != "constant":
            raise ValueError("loss schedule type must be one of ['constant']")

    def to_yaml_dict(self) -> dict[str, str]:
        return {"type": self.type}


@dataclass(frozen=True)
class LossTermConfig:
    name: LossName
    weight: float
    enabled: bool = True
    params: dict[str, Any] = field(default_factory=dict)
    schedule: LossScheduleConfig = field(default_factory=LossScheduleConfig)

    def validate(self, role: LossRole) -> None:
        if self.name != "ssim":
            raise ValueError("loss name must be one of ['ssim']")
        if role != "generator":
            raise ValueError("loss 'ssim' is supported only in losses.generator")
        if self.weight < 0:
            raise ValueError(f"loss '{self.name}' weight must be greater than or equal to 0")
        self.schedule.validate()
        reject_unknown_keys(self.params, _SSIM_PARAM_KEYS, f"loss '{self.name}' params")

    @property
    def is_active(self) -> bool:
        return self.enabled and self.weight != 0.0

    def to_yaml_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "weight": self.weight,
            "enabled": self.enabled,
            "params": dict(self.params),
            "schedule": self.schedule.to_yaml_dict(),
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

    def to_yaml_dict(self) -> dict[str, Any]:
        return {
            "generator": [term.to_yaml_dict() for term in self.generator],
            "discriminator": [term.to_yaml_dict() for term in self.discriminator],
        }


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

    name = _parse_choice(raw["name"], f"{context}.name", {"ssim"})
    params = raw.get("params", {})
    if params is None:
        params = {}
    if not isinstance(params, dict):
        raise TypeError(f"{context}.params must be a YAML mapping")

    schedule = _parse_loss_schedule(raw.get("schedule", {}), f"{context}.schedule")
    return LossTermConfig(
        name=cast(LossName, name),
        weight=float(raw["weight"]),
        enabled=parse_bool_strict(raw.get("enabled", True), f"{context}.enabled"),
        params=dict(params),
        schedule=schedule,
    )


def _parse_loss_schedule(raw: Any, context: str) -> LossScheduleConfig:
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise TypeError(f"{context} must be a YAML mapping")
    reject_unknown_keys(raw, _LOSS_SCHEDULE_KEYS, context)
    schedule_type = _parse_choice(raw.get("type", "constant"), f"{context}.type", {"constant"})
    return LossScheduleConfig(type=cast(LossScheduleType, schedule_type))


def _parse_choice(raw: Any, field_name: str, choices: set[str]) -> str:
    if not isinstance(raw, str):
        raise TypeError(f"{field_name} must be a string. Supported values: {sorted(choices)}.")
    if raw not in choices:
        raise ValueError(f"{field_name} must be one of {sorted(choices)}. Got {raw!r}.")
    return raw


def _validate_unique_loss_names(terms: tuple[LossTermConfig, ...], role: str) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for term in terms:
        if term.name in seen:
            duplicates.add(term.name)
        seen.add(term.name)
    if duplicates:
        joined = ", ".join(sorted(duplicates))
        raise ValueError(f"Duplicate loss name(s) in losses.{role}: {joined}")


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int
    epochs: int
    lr_g: float
    lr_d: float
    beta1: float
    beta2: float
    l1_weight: float
    seed: int | None
    num_workers: int
    validate_rate: int
    checkpoint_rate: int
    log_rate: int = 15
    resume: str | None = None

    def validate(self) -> None:
        for field_name, value in (
            ("batch_size", self.batch_size),
            ("epochs", self.epochs),
            ("validate_rate", self.validate_rate),
            ("checkpoint_rate", self.checkpoint_rate),
            ("log_rate", self.log_rate),
        ):
            if value <= 0:
                raise ValueError(f"{field_name} must be greater than 0")
        if self.num_workers < 0:
            raise ValueError("num_workers must be >= 0")
        for field_name, value in (("lr_g", self.lr_g), ("lr_d", self.lr_d)):
            if value <= 0:
                raise ValueError(f"{field_name} must be greater than 0")
        for field_name, value in (("beta1", self.beta1), ("beta2", self.beta2)):
            if not (0.0 <= value < 1.0):
                raise ValueError(f"{field_name} must be in [0, 1)")
        if self.l1_weight < 0:
            raise ValueError("l1_weight must be greater than or equal to 0")
