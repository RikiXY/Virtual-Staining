from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainingConfig:
    dataset_root: Path
    results_path: Path
    run_name: str
    image_size: tuple[int, int]
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

    @property
    def run_root(self) -> Path:
        return self.results_path / self.run_name

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
        return cls(
            dataset_root=Path(args.dataset_root),
            results_path=Path(args.results_path),
            run_name=args.run_name,
            image_size=tuple(args.image_size),
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr_g=args.lr_g,
            lr_d=args.lr_d,
            beta1=args.beta1,
            beta2=args.beta2,
            l1_weight=args.l1_weight,
            seed=args.seed,
            num_workers=args.num_workers,
            validate_rate=args.validate_rate,
            checkpoint_rate=args.checkpoint_rate,
            log_rate=args.log_rate,
            resume=args.resume,
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainingConfig:
        import yaml

        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        return cls(
            dataset_root=Path(data["dataset_root"]),
            results_path=Path(data["results_path"]),
            run_name=data["run_name"],
            image_size=tuple(data.get("image_size", [256, 256])),
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
        )
