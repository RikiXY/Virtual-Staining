from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

_TRAINING_KEYS: frozenset[str] = frozenset(
    {
        # shared fields and size aliases (accepted after section_with_shared_fields injects them)
        "dataset_root",
        "results_path",
        "run_name",
        "image_size",
        "model_image_size",
        # section-specific
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

_INFERENCE_KEYS: frozenset[str] = frozenset(
    {
        # shared fields and size aliases (accepted after section_with_shared_fields injects them)
        "dataset_root",
        "results_path",
        "run_name",
        "image_size",
        "model_image_size",
        # section-specific
        "checkpoint",
        "checkpoint_policy",
        "test_dir",
        "output_dir",
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

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainingConfig:
        """Compatibility shim — prefer RunConfig.from_yaml()."""
        from virtual_staining.config.run import RunConfig

        run_config = RunConfig.from_yaml(path)
        if run_config.training is None:
            raise ValueError(f"No 'training' section found in config: {path}")
        return run_config.training


@dataclass(frozen=True)
class InferenceConfig:
    checkpoint_policy: str | None = None
    checkpoint_path: Path | None = None
    test_dir: Path | None = None
    output_dir: Path | None = None

    def validate(self) -> None:
        if self.checkpoint_policy is None and self.checkpoint_path is None:
            raise ValueError(
                "Either inference.checkpoint or inference.checkpoint_policy must be set."
            )
        if self.checkpoint_policy is not None and self.checkpoint_policy != "latest":
            raise ValueError(
                f"Unknown checkpoint_policy: {self.checkpoint_policy!r}. "
                "Only 'latest' is supported."
            )
        if self.checkpoint_path is not None and not str(self.checkpoint_path).strip():
            raise ValueError("checkpoint must be a non-empty path")

    @classmethod
    def from_yaml(cls, path: str | Path) -> InferenceConfig:
        """Compatibility shim — prefer RunConfig.from_yaml()."""
        from virtual_staining.config.run import RunConfig

        run_config = RunConfig.from_yaml(path)
        if run_config.inference is None:
            raise ValueError(f"No 'inference' section found in config: {path}")
        return run_config.inference
