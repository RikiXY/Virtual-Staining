from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunPaths:
    root: Path

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
    def output_train_dir(self) -> Path:
        return self.artifacts_dir / "output_train"

    @property
    def output_val_dir(self) -> Path:
        return self.artifacts_dir / "output_val"

    @property
    def output_test_dir(self) -> Path:
        return self.artifacts_dir / "output_test"

    @property
    def evaluation_dir(self) -> Path:
        return self.root / "evaluation"

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

    def stage_config_dir(self, stage: str) -> Path:
        return self.config_dir / stage

    def stage_environment(self, stage: str) -> Path:
        return self.metadata_dir / "environments" / f"{stage}.json"

    def create_directories(self) -> None:
        """Create directories used directly by training and inference."""
        for directory in [
            self.config_dir,
            self.metadata_dir,
            self.logs_dir,
            self.checkpoints_dir,
            self.metrics_dir,
            self.artifacts_dir,
            self.output_train_dir,
            self.output_val_dir,
            self.output_test_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)
