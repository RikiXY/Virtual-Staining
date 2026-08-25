from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from virtual_staining.config.project import ProjectConfig
from virtual_staining.experiment.stages import RunStageName


@dataclass(frozen=True)
class StageLayout:
    input_config: Path
    resolved_config: Path
    environment: Path


@dataclass(frozen=True)
class RunLayout:
    root: Path

    @classmethod
    def from_project(cls, project: ProjectConfig) -> RunLayout:
        return cls(project.results_path / project.run_name)

    @classmethod
    def from_artifact_path(cls, path: Path) -> RunLayout:
        resolved = path.resolve()
        parts = resolved.parts
        try:
            index = parts.index("artifacts")
        except ValueError:
            raise ValueError(
                f"Cannot infer run layout from {path}; provide the explicit output/save path."
            ) from None
        if index == 0:
            raise ValueError(
                f"Cannot infer run layout from {path}; provide the explicit output/save path."
            )
        return cls(Path(*parts[:index]))

    @classmethod
    def from_evaluation_path(cls, path: Path) -> RunLayout:
        resolved = path.resolve()
        if resolved.name == "evaluation":
            return cls(resolved.parent)
        if resolved.parent.name == "evaluation":
            return cls(resolved.parent.parent)
        raise ValueError(
            f"Cannot infer run layout from {path}; provide the explicit output/save path."
        )

    @property
    def config_dir(self) -> Path:
        return self.root / "config"

    @property
    def metadata_dir(self) -> Path:
        return self.root / "metadata"

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    @property
    def checkpoints_dir(self) -> Path:
        return self.root / "checkpoints"

    @property
    def metrics_dir(self) -> Path:
        return self.root / "metrics"

    @property
    def artifacts_dir(self) -> Path:
        return self.root / "artifacts"

    @property
    def evaluation_dir(self) -> Path:
        return self.root / "evaluation"

    @property
    def comparisons_dir(self) -> Path:
        return self.root / "comparisons"

    @property
    def output_train_dir(self) -> Path:
        return self.artifacts_dir / "output_train"

    @property
    def output_val_dir(self) -> Path:
        return self.artifacts_dir / "output_val"

    @property
    def output_test_dir(self) -> Path:
        return self.artifacts_dir / "output_test"

    @property
    def per_image_metrics(self) -> Path:
        return self.evaluation_dir / "per_image_metrics.csv"

    @property
    def summary_csv(self) -> Path:
        return self.evaluation_dir / "summary.csv"

    @property
    def skipped_csv(self) -> Path:
        return self.evaluation_dir / "skipped.csv"

    @property
    def run_metadata(self) -> Path:
        return self.metadata_dir / "run.json"

    @property
    def events(self) -> Path:
        return self.metadata_dir / "events.jsonl"

    @property
    def run_log(self) -> Path:
        return self.logs_dir / "run.log"

    @property
    def epochs_csv(self) -> Path:
        return self.metrics_dir / "epochs.csv"

    @property
    def checkpoint_selection(self) -> Path:
        return self.checkpoints_dir / "best.json"

    def stage(self, stage: RunStageName) -> StageLayout:
        if stage not in {"train", "infer", "evaluate"}:
            raise ValueError(f"Unsupported run stage: {stage}")
        return StageLayout(
            self.config_dir / stage / "input.yaml",
            self.config_dir / stage / "resolved.yaml",
            self.metadata_dir / "environments" / f"{stage}.json",
        )


def ensure_run_directories(layout: RunLayout) -> None:
    for directory in (
        layout.config_dir,
        layout.metadata_dir,
        layout.logs_dir,
        layout.checkpoints_dir,
        layout.metrics_dir,
        layout.artifacts_dir,
        layout.output_train_dir,
        layout.output_val_dir,
        layout.output_test_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)
