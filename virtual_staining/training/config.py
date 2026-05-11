from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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
        "train_dir",
        "val_dir",
    }
)


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
    train_dir: Path | None = None
    val_dir: Path | None = None

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
