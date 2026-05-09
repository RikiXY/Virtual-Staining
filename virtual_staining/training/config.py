from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from virtual_staining.image_size import parse_wh_size, parse_wh_size_from_aliases
from virtual_staining.run_config import load_yaml_mapping, section_with_shared_fields


def _validate_non_empty_string(field_name: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _validate_positive_pair(field_name: str, value: tuple[int, int]) -> None:
    if len(value) != 2 or any(dimension <= 0 for dimension in value):
        raise ValueError(f"{field_name} must contain two positive integers")


def _resolve_checkpoint(data: dict[str, object], run_root: Path) -> Path:
    checkpoint = data.get("checkpoint")
    if checkpoint is not None:
        checkpoint_value = str(checkpoint)
        if not checkpoint_value.strip():
            raise ValueError("checkpoint must be a non-empty path")
        checkpoint_path = Path(checkpoint_value)
        if checkpoint_path.is_absolute():
            return checkpoint_path
        return run_root / checkpoint_path

    if data.get("checkpoint_policy") == "latest":
        checkpoint_dir = run_root / "checkpoints"
        checkpoints = sorted(checkpoint_dir.glob("ep*.pth"))
        if not checkpoints:
            raise FileNotFoundError(
                f"No checkpoints found for checkpoint_policy=latest in {checkpoint_dir}"
            )
        return checkpoints[-1]

    raise ValueError(
        "Config field inference.checkpoint or inference.checkpoint_policy is "
        "required for inference."
    )


@dataclass(frozen=True)
class TrainingConfig:
    dataset_root: Path
    results_path: Path
    run_name: str
    image_size: tuple[int, int]  # (width, height)
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

    @property
    def run_root(self) -> Path:
        return self.results_path / self.run_name

    @property
    def dataset_train_dir(self) -> Path:
        if self.train_dir is not None:
            return self.train_dir
        return self.dataset_root / "dataset_train"

    @property
    def dataset_val_dir(self) -> Path:
        if self.val_dir is not None:
            return self.val_dir
        return self.dataset_root / "dataset_val"

    def validate(self) -> None:
        _validate_non_empty_string("run_name", self.run_name)
        _validate_positive_pair("image_size", self.image_size)

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
            raise ValueError("num_workers must be greater than or equal to 0")

        for field_name, value in (("lr_g", self.lr_g), ("lr_d", self.lr_d)):
            if value <= 0:
                raise ValueError(f"{field_name} must be greater than 0")

        for field_name, value in (("beta1", self.beta1), ("beta2", self.beta2)):
            if value < 0 or value >= 1:
                raise ValueError(f"{field_name} must be greater than or equal to 0 and less than 1")

        if self.l1_weight < 0:
            raise ValueError("l1_weight must be greater than or equal to 0")

    def to_yaml(self, path: str | Path) -> None:
        import yaml

        data = {
            "dataset_root": str(self.dataset_root),
            "results_path": str(self.results_path),
            "run_name": self.run_name,
            "image_size": list(self.image_size),
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "lr_g": self.lr_g,
            "lr_d": self.lr_d,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "l1_weight": self.l1_weight,
            "seed": self.seed,
            "num_workers": self.num_workers,
            "validate_rate": self.validate_rate,
            "checkpoint_rate": self.checkpoint_rate,
            "log_rate": self.log_rate,
            "resume": self.resume,
        }
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, default_flow_style=False)

    @classmethod
    def from_args(cls, args) -> TrainingConfig:
        config = cls(
            dataset_root=Path(args.dataset_root),
            results_path=Path(getattr(args, "results_path", "local_workspace/results")),
            run_name=args.run_name,
            image_size=parse_wh_size(getattr(args, "image_size", (256, 256)), (256, 256)),
            batch_size=getattr(args, "batch_size", 8),
            epochs=args.epochs,
            lr_g=getattr(args, "lr_g", 2e-4),
            lr_d=getattr(args, "lr_d", 2e-4),
            beta1=getattr(args, "beta1", 0.5),
            beta2=getattr(args, "beta2", 0.999),
            l1_weight=getattr(args, "l1_weight", 25.0),
            seed=getattr(args, "seed", None),
            num_workers=getattr(args, "num_workers", min(4, os.cpu_count() or 1)),
            validate_rate=getattr(args, "validate_rate", 10),
            checkpoint_rate=getattr(args, "checkpoint_rate", 10),
            log_rate=getattr(args, "log_rate", 15),
            resume=getattr(args, "resume", None),
        )
        config.validate()
        return config

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainingConfig:
        raw_data = load_yaml_mapping(path)
        data = section_with_shared_fields(
            raw_data,
            "training",
            {"dataset_root", "results_path", "run_name", "image_size"},
        )

        config = cls(
            dataset_root=Path(data["dataset_root"]),
            results_path=Path(data["results_path"]),
            run_name=data["run_name"],
            image_size=parse_wh_size_from_aliases(
                data, ("model_image_size", "image_size"), (256, 256)
            ),
            batch_size=int(data.get("batch_size", 8)),
            epochs=int(data["epochs"]),
            lr_g=float(data.get("lr_g", 2e-4)),
            lr_d=float(data.get("lr_d", 2e-4)),
            beta1=float(data.get("beta1", 0.5)),
            beta2=float(data.get("beta2", 0.999)),
            l1_weight=float(data.get("l1_weight", 25.0)),
            seed=data.get("seed"),
            num_workers=int(data.get("num_workers", min(4, os.cpu_count() or 1))),
            validate_rate=int(data.get("validate_rate", 10)),
            checkpoint_rate=int(data.get("checkpoint_rate", 10)),
            log_rate=int(data.get("log_rate", 15)),
            resume=data.get("resume"),
            train_dir=Path(data["train_dir"]) if data.get("train_dir") else None,
            val_dir=Path(data["val_dir"]) if data.get("val_dir") else None,
        )
        config.validate()
        return config


@dataclass(frozen=True)
class InferenceConfig:
    dataset_root: Path
    results_path: Path
    run_name: str
    checkpoint: Path
    image_size: tuple[int, int]  # (width, height)
    test_dir_override: Path | None = None
    output_dir: Path | None = None

    @property
    def run_root(self) -> Path:
        return self.results_path / self.run_name

    @property
    def test_dir(self) -> Path:
        if self.test_dir_override is not None:
            return self.test_dir_override
        return self.dataset_root / "dataset_test"

    @property
    def output_test_dir(self) -> Path:
        if self.output_dir is not None:
            return self.output_dir
        return self.run_root / "output_test"

    def validate(self) -> None:
        _validate_non_empty_string("run_name", self.run_name)
        _validate_positive_pair("image_size", self.image_size)
        if str(self.checkpoint).strip() in {"", "."}:
            raise ValueError("checkpoint must be a non-empty path")

    @classmethod
    def from_yaml(cls, path: str | Path) -> InferenceConfig:
        raw_data = load_yaml_mapping(path)
        data = section_with_shared_fields(
            raw_data,
            "inference",
            {"dataset_root", "results_path", "run_name", "image_size"},
        )
        training_data = section_with_shared_fields(
            raw_data,
            "training",
            {"dataset_root", "results_path", "run_name", "image_size"},
        )

        run_root = Path(data["results_path"]) / data["run_name"]

        config = cls(
            dataset_root=Path(data["dataset_root"]),
            results_path=Path(data["results_path"]),
            run_name=data["run_name"],
            checkpoint=_resolve_checkpoint(data, run_root),
            image_size=parse_wh_size_from_aliases(
                data,
                ("model_image_size", "image_size"),
                parse_wh_size_from_aliases(
                    training_data, ("model_image_size", "image_size"), (256, 256)
                ),
            ),
            test_dir_override=Path(data["test_dir"]) if data.get("test_dir") else None,
            output_dir=Path(data["output_dir"]) if data.get("output_dir") else None,
        )
        config.validate()
        return config
