from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class EpochMetrics:
    loss_G: float
    loss_D: float
    loss_L1: float | None = None
    loss_adv: float | None = None
    raw: dict[str, float] = field(default_factory=dict)
    weighted: dict[str, float] = field(default_factory=dict)
    current_weight: dict[str, float] = field(default_factory=dict)
    image: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainingResult:
    final_epoch: int
    best_checkpoint_path: Path | None
    stopped_early: bool = False
    stop_epoch: int | None = None
    stop_reason: str | None = None
    early_stopping_monitor: str | None = None
    early_stopping_mode: str | None = None
    early_stopping_best_epoch: int | None = None
    early_stopping_best_value: float | None = None
