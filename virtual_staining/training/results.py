from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EpochMetrics:
    loss_G: float
    loss_D: float
    loss_L1: float | None = None
    loss_adv: float | None = None


@dataclass(frozen=True)
class TrainingResult:
    final_epoch: int
    best_checkpoint_path: Path | None
