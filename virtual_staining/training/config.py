from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from virtual_staining.config.validation import parse_bool_strict, parse_choice, reject_unknown_keys
from virtual_staining.metrics import is_higher_better_metric
from virtual_staining.training.loss_config import LossConfig, parse_loss_config

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
        "checkpoint_top_k",
        "log_rate",
        "resume",
        "scheduler",
        "early_stopping",
        "augmentation",
        "losses",
    }
)
_SCHEDULER_KEYS: frozenset[str] = frozenset(
    {"name", "decay_start_epoch", "monitor", "mode", "factor", "patience", "min_lr"}
)
_EARLY_STOPPING_KEYS: frozenset[str] = frozenset({"monitor", "mode", "patience", "min_delta"})
_AUGMENTATION_KEYS: frozenset[str] = frozenset({"enabled", "expansion_factor", "intensity"})

AugmentationIntensity = Literal["light", "medium", "strong"]
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
LearningRateSchedulerName = Literal["none", "linear_decay", "reduce_on_plateau"]
EarlyStoppingMonitor = str
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
_VALIDATION_LOSS_MONITOR_PATTERN = re.compile(
    r"^loss_val_(?:total_(?:generator|discriminator)|"
    r"(?:raw|weighted|current_weight)_(?:generator|discriminator)_[a-z0-9_]+)$"
)


def is_supported_early_stopping_monitor(monitor: str) -> bool:
    return (
        monitor in SUPPORTED_CHECKPOINT_METRICS
        or monitor == "loss_D_val"
        or bool(_VALIDATION_LOSS_MONITOR_PATTERN.fullmatch(monitor))
    )


@dataclass(frozen=True)
class LearningRateSchedulerConfig:
    name: LearningRateSchedulerName = "none"
    decay_start_epoch: int | None = None
    monitor: CheckpointMetric = "loss_G_val"
    mode: CheckpointMode = "min"
    factor: float = 0.1
    patience: int = 10
    min_lr: float = 0.0

    def validate(self, *, epochs: int) -> None:
        if self.name not in {"none", "linear_decay", "reduce_on_plateau"}:
            raise ValueError(
                "training.scheduler.name must be one of "
                "['linear_decay', 'none', 'reduce_on_plateau']"
            )
        if self.name == "linear_decay":
            if self.decay_start_epoch is None:
                raise ValueError("training.scheduler.decay_start_epoch is required")
            if self.decay_start_epoch < 0:
                raise ValueError("training.scheduler.decay_start_epoch must be >= 0")
            if self.decay_start_epoch >= epochs:
                raise ValueError("training.scheduler.decay_start_epoch must be less than epochs")
        elif self.decay_start_epoch is not None and self.decay_start_epoch < 0:
            raise ValueError("training.scheduler.decay_start_epoch must be >= 0")
        if self.name == "reduce_on_plateau":
            if self.monitor not in SUPPORTED_CHECKPOINT_METRICS:
                raise ValueError(
                    "training.scheduler.monitor must be one of "
                    f"{sorted(SUPPORTED_CHECKPOINT_METRICS)}"
                )
            if self.mode not in {"min", "max"}:
                raise ValueError("training.scheduler.mode must be one of ['max', 'min']")
            if not (0.0 < self.factor < 1.0):
                raise ValueError("training.scheduler.factor must be in (0, 1)")
            if self.patience < 0:
                raise ValueError("training.scheduler.patience must be >= 0")
            if self.min_lr < 0:
                raise ValueError("training.scheduler.min_lr must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"name": self.name}
        if self.name == "linear_decay":
            data["decay_start_epoch"] = self.decay_start_epoch
        elif self.name == "reduce_on_plateau":
            data.update(
                {
                    "monitor": self.monitor,
                    "mode": self.mode,
                    "factor": self.factor,
                    "patience": self.patience,
                    "min_lr": self.min_lr,
                }
            )
        return data


@dataclass(frozen=True)
class EarlyStoppingConfig:
    monitor: EarlyStoppingMonitor = "val_ssim"
    mode: CheckpointMode = "max"
    patience: int = 15
    min_delta: float = 0.0

    def validate(self) -> None:
        if not is_supported_early_stopping_monitor(self.monitor):
            raise ValueError(
                "training.early_stopping.monitor must be a validation CSV column "
                "such as loss_G_val, loss_D_val, val_ssim, val_mae, or a configured "
                "loss_val_* column"
            )
        if self.mode not in {"min", "max"}:
            raise ValueError("training.early_stopping.mode must be one of ['max', 'min']")
        if self.patience < 0:
            raise ValueError("training.early_stopping.patience must be >= 0")
        if self.min_delta < 0:
            raise ValueError("training.early_stopping.min_delta must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        return {
            "monitor": self.monitor,
            "mode": self.mode,
            "patience": self.patience,
            "min_delta": self.min_delta,
        }


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

    def to_dict(self) -> dict[str, Any]:
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
    expansion_factor = raw.get("expansion_factor", 1)
    if isinstance(expansion_factor, bool) or not isinstance(expansion_factor, int):
        raise TypeError("augmentation.expansion_factor must be an integer")
    config = AugmentationConfig(
        enabled=parse_bool_strict(raw.get("enabled", False), "augmentation.enabled"),
        expansion_factor=expansion_factor,
        intensity=cast(
            AugmentationIntensity,
            parse_choice(
                raw.get("intensity", "light"),
                "augmentation.intensity",
                {"light", "medium", "strong"},
            ),
        ),
    )
    config.validate()
    return config


def parse_learning_rate_scheduler_config(raw: Any, *, epochs: int) -> LearningRateSchedulerConfig:
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise TypeError("training.scheduler must be a YAML mapping")
    reject_unknown_keys(raw, _SCHEDULER_KEYS, "training.scheduler")
    name = parse_choice(
        raw.get("name", "none"),
        "training.scheduler.name",
        {"none", "linear_decay", "reduce_on_plateau"},
    )
    monitor = parse_choice(
        raw.get("monitor", "loss_G_val"),
        "training.scheduler.monitor",
        set(SUPPORTED_CHECKPOINT_METRICS),
    )
    config = LearningRateSchedulerConfig(
        name=cast(LearningRateSchedulerName, name),
        decay_start_epoch=int(raw["decay_start_epoch"])
        if raw.get("decay_start_epoch") is not None
        else None,
        monitor=cast(CheckpointMetric, monitor),
        mode=cast(
            CheckpointMode,
            parse_choice(
                raw.get("mode", default_checkpoint_mode(monitor)),
                "training.scheduler.mode",
                {"min", "max"},
            ),
        ),
        factor=float(raw.get("factor", 0.1)),
        patience=int(raw.get("patience", 10)),
        min_lr=float(raw.get("min_lr", 0.0)),
    )
    config.validate(epochs=epochs)
    return config


def parse_early_stopping_config(raw: Any) -> EarlyStoppingConfig | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise TypeError("training.early_stopping must be a YAML mapping")
    reject_unknown_keys(raw, _EARLY_STOPPING_KEYS, "training.early_stopping")
    monitor = raw.get("monitor", "val_ssim")
    if not isinstance(monitor, str):
        raise TypeError("training.early_stopping.monitor must be a string")
    if not is_supported_early_stopping_monitor(monitor):
        raise ValueError(
            "training.early_stopping.monitor must be a validation CSV column "
            "such as loss_G_val, loss_D_val, val_ssim, val_mae, or a configured "
            "loss_val_* column"
        )
    default_mode = (
        "min"
        if monitor in {"loss_G_val", "loss_D_val"} or monitor.startswith("loss_val_")
        else default_checkpoint_mode(monitor)
    )
    config = EarlyStoppingConfig(
        monitor=monitor,
        mode=cast(
            CheckpointMode,
            parse_choice(
                raw.get("mode", default_mode), "training.early_stopping.mode", {"min", "max"}
            ),
        ),
        patience=int(raw.get("patience", 15)),
        min_delta=float(raw.get("min_delta", 0.0)),
    )
    config.validate()
    return config


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
    losses: LossConfig = field(default_factory=LossConfig)
    checkpoint_top_k: int = 3
    log_rate: int = 15
    resume: str | None = None
    scheduler: LearningRateSchedulerConfig = field(default_factory=LearningRateSchedulerConfig)
    early_stopping: EarlyStoppingConfig | None = None
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)

    def __post_init__(self) -> None:
        self.validate()

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> TrainingConfig:
        reject_unknown_keys(data, _TRAINING_KEYS, "training")
        if "epochs" not in data:
            raise ValueError("training.epochs is required")
        if "losses" not in data:
            raise ValueError("training.losses is required")
        epochs = int(data["epochs"])
        return cls(
            batch_size=int(data.get("batch_size", 8)),
            epochs=epochs,
            lr_g=float(data.get("lr_g", 2e-4)),
            lr_d=float(data.get("lr_d", 2e-4)),
            beta1=float(data.get("beta1", 0.5)),
            beta2=float(data.get("beta2", 0.999)),
            seed=data.get("seed"),
            num_workers=int(data.get("num_workers", min(4, os.cpu_count() or 1))),
            validate_rate=int(data.get("validate_rate", 10)),
            checkpoint_rate=int(data.get("checkpoint_rate", 10)),
            checkpoint_top_k=int(data.get("checkpoint_top_k", 3)),
            log_rate=int(data.get("log_rate", 15)),
            resume=data.get("resume"),
            scheduler=parse_learning_rate_scheduler_config(
                data.get("scheduler", {}), epochs=epochs
            ),
            early_stopping=parse_early_stopping_config(data.get("early_stopping")),
            augmentation=parse_augmentation_config(data.get("augmentation", {})),
            losses=parse_loss_config(data["losses"]),
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "lr_g": self.lr_g,
            "lr_d": self.lr_d,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "seed": self.seed,
            "num_workers": self.num_workers,
            "validate_rate": self.validate_rate,
            "checkpoint_rate": self.checkpoint_rate,
            "checkpoint_top_k": self.checkpoint_top_k,
            "log_rate": self.log_rate,
            "resume": self.resume,
            "scheduler": self.scheduler.to_dict(),
            "augmentation": self.augmentation.to_dict(),
            "losses": self.losses.to_dict(),
        }
        if self.early_stopping is not None:
            data["early_stopping"] = self.early_stopping.to_dict()
        return {key: value for key, value in data.items() if value is not None}

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
        self.scheduler.validate(epochs=self.epochs)
        if self.early_stopping is not None:
            self.early_stopping.validate()
        self.augmentation.validate()
        self.losses.validate()


def default_checkpoint_mode(metric: str) -> CheckpointMode:
    if metric not in SUPPORTED_CHECKPOINT_METRICS:
        raise ValueError(
            f"Unsupported checkpoint_metric {metric!r}. "
            f"Supported metrics: {sorted(SUPPORTED_CHECKPOINT_METRICS)}."
        )
    if metric == "loss_G_val":
        return "min"
    if metric.startswith("val_"):
        return "max" if is_higher_better_metric(metric.removeprefix("val_")) else "min"
    raise AssertionError(f"Unsupported checkpoint_metric slipped through validation: {metric!r}")
