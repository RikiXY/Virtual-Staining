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
    def comparisons_dir(self) -> Path:
        return self.root / "comparisons"

    @property
    def input_config(self) -> Path:
        return self.config_dir / "input.yaml"

    @property
    def resolved_config(self) -> Path:
        return self.config_dir / "resolved.yaml"

    @property
    def run_metadata(self) -> Path:
        return self.metadata_dir / "run.json"

    @property
    def environment_metadata(self) -> Path:
        return self.metadata_dir / "environment.json"

    @property
    def config_hash(self) -> Path:
        return self.metadata_dir / "config_hash.txt"

    @property
    def stages_dir(self) -> Path:
        return self.metadata_dir / "stages"

    @property
    def events_log(self) -> Path:
        return self.metadata_dir / "events.jsonl"

    def create_directories(self) -> None:
        """Create all run sub-directories."""
        for directory in [
            self.config_dir,
            self.metadata_dir,
            self.stages_dir,
            self.logs_dir,
            self.checkpoints_dir,
            self.metrics_dir,
            self.artifacts_dir,
            self.output_train_dir,
            self.output_val_dir,
            self.output_test_dir,
            self.evaluation_dir,
            self.comparisons_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)
