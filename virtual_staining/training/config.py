from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from virtual_staining.config.project import ProjectConfig

_TRAINING_KEYS: frozenset[str] = frozenset(
    {
        # shared fields and size aliases
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
        # shared fields and size aliases
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


def _require_project(config_type: str, project: ProjectConfig | None) -> ProjectConfig:
    if project is None:
        raise ValueError(
            f"{config_type} requires a ProjectConfig for project-derived paths. "
            "Construct it via RunConfig.from_yaml() or pass project=ProjectConfig(...)."
        )
    return project


def _resolve_checkpoint(data: dict[str, object], project: ProjectConfig) -> Path:
    checkpoint = data.get("checkpoint")
    if checkpoint is not None:
        checkpoint_value = str(checkpoint)
        if not checkpoint_value.strip():
            raise ValueError("checkpoint must be a non-empty path")
        checkpoint_path = Path(checkpoint_value)
        if checkpoint_path.is_absolute():
            return checkpoint_path
        return project.run_root / checkpoint_path

    if data.get("checkpoint_policy") == "latest":
        checkpoint_dir = project.run_root / "checkpoints"
        checkpoints = sorted(checkpoint_dir.glob("ep*.pth"))
        if not checkpoints:
            raise FileNotFoundError(
                f"No checkpoints found for checkpoint_policy=latest in {checkpoint_dir}"
            )
        return checkpoints[-1]

    raise ValueError("Either inference.checkpoint or inference.checkpoint_policy must be set.")


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
    project: ProjectConfig | None = None

    @property
    def dataset_root(self) -> Path:
        return _require_project("TrainingConfig", self.project).dataset_root

    @property
    def results_path(self) -> Path:
        return _require_project("TrainingConfig", self.project).results_path

    @property
    def run_name(self) -> str:
        return _require_project("TrainingConfig", self.project).run_name

    @property
    def image_size(self) -> tuple[int, int]:
        return _require_project("TrainingConfig", self.project).image_size

    @property
    def run_root(self) -> Path:
        return _require_project("TrainingConfig", self.project).run_root

    @property
    def dataset_train_dir(self) -> Path:
        if self.train_dir is not None:
            return self.train_dir
        return _require_project("TrainingConfig", self.project).dataset_train_dir

    @property
    def dataset_val_dir(self) -> Path:
        if self.val_dir is not None:
            return self.val_dir
        return _require_project("TrainingConfig", self.project).dataset_val_dir

    def validate(self) -> None:
        if self.project is not None:
            if not self.project.run_name.strip():
                raise ValueError("run_name must be a non-empty string")
            width, height = self.project.image_size
            if width <= 0 or height <= 0:
                raise ValueError("image_size must contain two positive integers")

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

    def to_yaml(self, path: str | Path) -> None:
        import yaml

        project = _require_project("TrainingConfig", self.project)
        data = {
            "dataset_root": str(project.dataset_root),
            "results_path": str(project.results_path),
            "run_name": project.run_name,
            "image_size": list(project.image_size),
            "training": {
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
                "train_dir": str(self.train_dir) if self.train_dir is not None else None,
                "val_dir": str(self.val_dir) if self.val_dir is not None else None,
            },
        }
        with open(path, "w", encoding="utf-8") as file_obj:
            yaml.safe_dump(data, file_obj, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainingConfig:
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
    project: ProjectConfig | None = None

    @property
    def image_size(self) -> tuple[int, int]:
        return _require_project("InferenceConfig", self.project).image_size

    @property
    def run_root(self) -> Path:
        return _require_project("InferenceConfig", self.project).run_root

    @property
    def checkpoint(self) -> Path:
        project = _require_project("InferenceConfig", self.project)
        data: dict[str, object] = {"checkpoint_policy": self.checkpoint_policy}
        if self.checkpoint_path is not None:
            data["checkpoint"] = str(self.checkpoint_path)
        return _resolve_checkpoint(data, project)

    @property
    def output_test_dir(self) -> Path:
        if self.output_dir is not None:
            return self.output_dir
        return self.run_root / "output_test"

    def validate(self) -> None:
        if self.project is not None:
            if not self.project.run_name.strip():
                raise ValueError("run_name must be a non-empty string")
            width, height = self.project.image_size
            if width <= 0 or height <= 0:
                raise ValueError("image_size must contain two positive integers")
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
        from virtual_staining.config.run import RunConfig

        run_config = RunConfig.from_yaml(path)
        if run_config.inference is None:
            raise ValueError(f"No 'inference' section found in config: {path}")
        return run_config.inference
