from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from virtual_staining.config.validation import parse_bool_strict, reject_unknown_keys
from virtual_staining.utils.metrics import is_higher_better_metric

_TRAINING_KEYS: frozenset[str] = frozenset(
    {
        "batch_size",
        "epochs",
        "lr_g",
        "lr_d",
        "beta1",
        "beta2",
        "seed",
        "num_workers",
        "validate_rate",
        "checkpoint_rate",
        "checkpoint_metric",
        "checkpoint_mode",
        "checkpoint_top_k",
        "log_rate",
        "resume",
    }
)
_AUGMENTATION_KEYS: frozenset[str] = frozenset({"enabled", "expansion_factor", "intensity"})

_LOSS_CONFIG_KEYS: frozenset[str] = frozenset({"generator", "discriminator"})
_LOSS_TERM_KEYS: frozenset[str] = frozenset({"name", "weight", "enabled", "params", "schedule"})
_LOSS_SCHEDULE_KEYS: frozenset[str] = frozenset(
    {"type", "start_epoch", "end_epoch", "epoch", "factor"}
)
_LOSS_MASK_KEYS: frozenset[str] = frozenset(
    {"enabled", "source", "foreground_weight", "background_weight", "ignore_empty_mask"}
)
_SSIM_PARAM_KEYS: frozenset[str] = frozenset(
    {"data_range", "window_size", "sigma", "channel_mode", "reduction", "mask"}
)
_L1_PARAM_KEYS: frozenset[str] = frozenset({"reduction", "mask"})
_ADVERSARIAL_BCE_PARAM_KEYS: frozenset[str] = frozenset()

LossName = Literal["adversarial_bce", "l1", "ssim"]
AugmentationIntensity = Literal["light", "medium", "strong"]
LossScheduleType = Literal[
    "constant",
    "linear_warmup",
    "linear_decay",
    "step",
    "cosine",
    "turn_on_after_epoch",
    "turn_off_after_epoch",
]
LossRole = Literal["generator", "discriminator"]
LossMaskSource = Literal["foreground_mask"]
CheckpointMetric = Literal[
    "loss_G_val",
    "val_ssim",
    "val_mae",
    "val_rmse",
    "val_psnr",
    "val_pcc_gray",
    "val_pcc_rgb_mean",
]
CheckpointMode = Literal["min", "max"]
SUPPORTED_CHECKPOINT_METRICS: frozenset[str] = frozenset(
    {
        "loss_G_val",
        "val_ssim",
        "val_mae",
        "val_rmse",
        "val_psnr",
        "val_pcc_gray",
        "val_pcc_rgb_mean",
    }
)
SUPPORTED_CHECKPOINT_MODES: frozenset[str] = frozenset({"min", "max"})


@dataclass(frozen=True)
class AugmentationConfig:
    enabled: bool = False
    expansion_factor: int = 1
    intensity: AugmentationIntensity = "light"

    def validate(self) -> None:
        if self.expansion_factor < 1:
            raise ValueError("augmentation.expansion_factor must be greater than or equal to 1")
        if self.intensity not in {"light", "medium", "strong"}:
            raise ValueError("augmentation.intensity must be one of ['light', 'medium', 'strong']")

    @property
    def effective_expansion_factor(self) -> int:
        return self.expansion_factor if self.enabled else 1

    def to_yaml_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "expansion_factor": self.expansion_factor,
            "intensity": self.intensity,
        }


def parse_augmentation_config(raw: Any) -> AugmentationConfig:
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise TypeError("augmentation must be a YAML mapping")
    reject_unknown_keys(raw, _AUGMENTATION_KEYS, "augmentation")

    expansion_factor_raw = raw.get("expansion_factor", 1)
    if isinstance(expansion_factor_raw, bool) or not isinstance(expansion_factor_raw, int):
        raise TypeError("augmentation.expansion_factor must be an integer")

    intensity = _parse_choice(
        raw.get("intensity", "light"),
        "augmentation.intensity",
        {"light", "medium", "strong"},
    )
    config = AugmentationConfig(
        enabled=parse_bool_strict(raw.get("enabled", False), "augmentation.enabled"),
        expansion_factor=expansion_factor_raw,
        intensity=cast(AugmentationIntensity, intensity),
    )
    config.validate()
    return config


@dataclass(frozen=True)
class LossScheduleConfig:
    type: LossScheduleType = "constant"
    start_epoch: int = 0
    end_epoch: int | None = None
    epoch: int | None = None
    factor: float = 0.0

    def validate(self) -> None:
        valid = [
            "constant",
            "cosine",
            "linear_decay",
            "linear_warmup",
            "step",
            "turn_off_after_epoch",
            "turn_on_after_epoch",
        ]
        if self.type not in valid:
            raise ValueError(f"loss schedule type must be one of {valid}")
        if self.start_epoch < 0:
            raise ValueError("loss schedule start_epoch must be greater than or equal to 0")
        if self.end_epoch is not None and self.end_epoch < self.start_epoch:
            raise ValueError("loss schedule end_epoch must be greater than or equal to start_epoch")
        if self.epoch is not None and self.epoch < 0:
            raise ValueError("loss schedule epoch must be greater than or equal to 0")
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
            span = max(1, self.end_epoch - self.start_epoch)
            progress = (epoch - self.start_epoch) / span
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

    def to_yaml_dict(self) -> dict[str, Any]:
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
class LossMaskConfig:
    enabled: bool = False
    source: LossMaskSource = "foreground_mask"
    foreground_weight: float = 1.0
    background_weight: float = 1.0
    ignore_empty_mask: bool = True

    def validate(self) -> None:
        if self.source != "foreground_mask":
            raise ValueError("loss mask source must be one of ['foreground_mask']")
        if self.foreground_weight < 0:
            raise ValueError("loss mask foreground_weight must be greater than or equal to 0")
        if self.background_weight < 0:
            raise ValueError("loss mask background_weight must be greater than or equal to 0")

    def to_yaml_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "source": self.source,
            "foreground_weight": self.foreground_weight,
            "background_weight": self.background_weight,
            "ignore_empty_mask": self.ignore_empty_mask,
        }


@dataclass(frozen=True)
class LossTermConfig:
    name: LossName
    weight: float
    enabled: bool = True
    params: dict[str, Any] = field(default_factory=dict)
    schedule: LossScheduleConfig = field(default_factory=LossScheduleConfig)

    def validate(self, role: LossRole) -> None:
        if self.name not in {"adversarial_bce", "l1", "ssim"}:
            raise ValueError("loss name must be one of ['adversarial_bce', 'l1', 'ssim']")
        if self.name in {"l1", "ssim"} and role != "generator":
            raise ValueError(f"loss '{self.name}' is supported only in losses.generator")
        if self.name == "ssim" and role != "generator":
            raise ValueError("loss 'ssim' is supported only in losses.generator")
        if self.weight < 0:
            raise ValueError(f"loss '{self.name}' weight must be greater than or equal to 0")
        self.schedule.validate()
        if self.name == "ssim":
            reject_unknown_keys(self.params, _SSIM_PARAM_KEYS, f"loss '{self.name}' params")
            _validate_ssim_params(self.params)
        elif self.name == "l1":
            reject_unknown_keys(self.params, _L1_PARAM_KEYS, f"loss '{self.name}' params")
            _validate_l1_params(self.params)
        else:
            reject_unknown_keys(
                self.params, _ADVERSARIAL_BCE_PARAM_KEYS, f"loss '{self.name}' params"
            )

    @property
    def is_active(self) -> bool:
        return self.enabled and self.weight != 0.0

    def current_weight(self, *, epoch: int, global_step: int | None = None) -> float:
        if not self.enabled:
            return 0.0
        return self.schedule.current_weight(self.weight, epoch=epoch, global_step=global_step)

    @property
    def mask(self) -> LossMaskConfig:
        return parse_loss_mask_config(self.params.get("mask"), f"loss '{self.name}' params.mask")

    @property
    def requires_mask(self) -> bool:
        return self.mask.enabled

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

    name = _parse_choice(raw["name"], f"{context}.name", {"adversarial_bce", "l1", "ssim"})
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
    schedule_type = _parse_choice(
        raw.get("type", "constant"),
        f"{context}.type",
        {
            "constant",
            "linear_warmup",
            "linear_decay",
            "step",
            "cosine",
            "turn_on_after_epoch",
            "turn_off_after_epoch",
        },
    )
    config = LossScheduleConfig(
        type=cast(LossScheduleType, schedule_type),
        start_epoch=int(raw.get("start_epoch", 0)),
        end_epoch=int(raw["end_epoch"]) if raw.get("end_epoch") is not None else None,
        epoch=int(raw["epoch"]) if raw.get("epoch") is not None else None,
        factor=float(raw.get("factor", 0.0)),
    )
    config.validate()
    return config


def parse_loss_mask_config(raw: Any, context: str = "loss mask") -> LossMaskConfig:
    if raw is None:
        return LossMaskConfig()
    if not isinstance(raw, dict):
        raise TypeError(f"{context} must be a YAML mapping")
    reject_unknown_keys(raw, _LOSS_MASK_KEYS, context)
    source = _parse_choice(
        raw.get("source", "foreground_mask"),
        f"{context}.source",
        {"foreground_mask"},
    )
    config = LossMaskConfig(
        enabled=parse_bool_strict(raw.get("enabled", False), f"{context}.enabled"),
        source=cast(LossMaskSource, source),
        foreground_weight=float(raw.get("foreground_weight", 1.0)),
        background_weight=float(raw.get("background_weight", 1.0)),
        ignore_empty_mask=parse_bool_strict(
            raw.get("ignore_empty_mask", True), f"{context}.ignore_empty_mask"
        ),
    )
    config.validate()
    return config


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


def _validate_ssim_params(params: dict[str, Any]) -> None:
    data_range = float(params.get("data_range", 1.0))
    if data_range <= 0:
        raise ValueError("loss 'ssim' params.data_range must be greater than 0")

    window_size = int(params.get("window_size", 11))
    if window_size <= 0 or window_size % 2 == 0:
        raise ValueError("loss 'ssim' params.window_size must be a positive odd integer")

    sigma = float(params.get("sigma", 1.5))
    if sigma <= 0:
        raise ValueError("loss 'ssim' params.sigma must be greater than 0")

    channel_mode = params.get("channel_mode", "rgb")
    if channel_mode not in {"rgb", "gray"}:
        raise ValueError("loss 'ssim' params.channel_mode must be one of ['gray', 'rgb']")

    reduction = params.get("reduction", "mean")
    if reduction not in {"mean", "sum", "none"}:
        raise ValueError("loss 'ssim' params.reduction must be one of ['mean', 'none', 'sum']")

    parse_loss_mask_config(params.get("mask"), "loss 'ssim' params.mask")


def _validate_l1_params(params: dict[str, Any]) -> None:
    reduction = params.get("reduction", "mean")
    if reduction not in {"mean", "sum", "none"}:
        raise ValueError("loss 'l1' params.reduction must be one of ['mean', 'none', 'sum']")

    parse_loss_mask_config(params.get("mask"), "loss 'l1' params.mask")


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int
    epochs: int
    lr_g: float
    lr_d: float
    beta1: float
    beta2: float
    seed: int | None
    num_workers: int
    validate_rate: int
    checkpoint_rate: int
    checkpoint_metric: CheckpointMetric = "loss_G_val"
    checkpoint_mode: CheckpointMode = "min"
    checkpoint_top_k: int = 3
    log_rate: int = 15
    resume: str | None = None

    def validate(self) -> None:
        for field_name, value in (
            ("batch_size", self.batch_size),
            ("epochs", self.epochs),
            ("validate_rate", self.validate_rate),
            ("checkpoint_rate", self.checkpoint_rate),
            ("checkpoint_top_k", self.checkpoint_top_k),
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
        if self.checkpoint_metric not in SUPPORTED_CHECKPOINT_METRICS:
            raise ValueError(
                "checkpoint_metric must be one of "
                f"{sorted(SUPPORTED_CHECKPOINT_METRICS)}. Got {self.checkpoint_metric!r}."
            )
        if self.checkpoint_mode not in SUPPORTED_CHECKPOINT_MODES:
            raise ValueError(
                "checkpoint_mode must be one of "
                f"{sorted(SUPPORTED_CHECKPOINT_MODES)}. Got {self.checkpoint_mode!r}."
            )


def default_checkpoint_mode(metric: str) -> CheckpointMode:
    """Return the default comparison mode for a supported checkpoint metric."""
    if metric not in SUPPORTED_CHECKPOINT_METRICS:
        raise ValueError(
            f"Unsupported checkpoint_metric {metric!r}. "
            f"Supported metrics: {sorted(SUPPORTED_CHECKPOINT_METRICS)}."
        )
    if metric == "loss_G_val":
        return "min"
    if metric.startswith("val_"):
        evaluation_metric = metric.removeprefix("val_")
        return "max" if is_higher_better_metric(evaluation_metric) else "min"
    raise AssertionError(f"Unsupported checkpoint_metric slipped through validation: {metric!r}")
