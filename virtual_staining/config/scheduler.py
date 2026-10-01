"""Learning-rate scheduler options for methods that opt into them.

The block is method-owned: a method definition parses it with its own monitor policy,
so no validation-metric catalogue lives here.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, cast, get_args

from virtual_staining.checkpoint_selection import CheckpointMode
from virtual_staining.config.validation import parse_choice, reject_unknown_keys

LearningRateSchedulerName = Literal["none", "linear_decay", "reduce_on_plateau"]
MonitorMode = Callable[[str, str], CheckpointMode]

_SCHEDULER_KEYS: frozenset[str] = frozenset(
    {"name", "decay_start_epoch", "monitor", "mode", "factor", "patience", "min_lr"}
)


@dataclass(frozen=True)
class LearningRateSchedulerConfig:
    name: LearningRateSchedulerName = "none"
    decay_start_epoch: int | None = None
    monitor: str | None = None
    mode: CheckpointMode = "min"
    factor: float = 0.1
    patience: int = 10
    min_lr: float = 0.0

    def validate(self, *, epochs: int) -> None:
        if self.name not in set(get_args(LearningRateSchedulerName)):
            raise ValueError(
                "training.scheduler.name must be one of "
                f"{sorted(get_args(LearningRateSchedulerName))}"
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
            if self.monitor is None:
                raise ValueError("training.scheduler.monitor is required for reduce_on_plateau")
            if self.mode not in set(get_args(CheckpointMode)):
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


def parse_learning_rate_scheduler_config(
    raw: Any,
    *,
    epochs: int,
    monitor_mode: MonitorMode,
    default_monitor: str,
) -> LearningRateSchedulerConfig:
    """Parse ``training.scheduler``; ``monitor_mode`` validates the monitor for the method."""
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise TypeError("training.scheduler must be a YAML mapping")
    reject_unknown_keys(raw, _SCHEDULER_KEYS, "training.scheduler")
    name = parse_choice(
        raw.get("name", "none"),
        "training.scheduler.name",
        set(get_args(LearningRateSchedulerName)),
    )
    monitor = raw.get("monitor", default_monitor)
    if not isinstance(monitor, str):
        raise TypeError("training.scheduler.monitor must be a string")
    natural_mode = monitor_mode(monitor, "training.scheduler.monitor")
    config = LearningRateSchedulerConfig(
        name=cast(LearningRateSchedulerName, name),
        decay_start_epoch=int(raw["decay_start_epoch"])
        if raw.get("decay_start_epoch") is not None
        else None,
        monitor=monitor,
        mode=cast(
            CheckpointMode,
            parse_choice(
                raw.get("mode", natural_mode),
                "training.scheduler.mode",
                set(get_args(CheckpointMode)),
            ),
        ),
        factor=float(raw.get("factor", 0.1)),
        patience=int(raw.get("patience", 10)),
        min_lr=float(raw.get("min_lr", 0.0)),
    )
    config.validate(epochs=epochs)
    return config
