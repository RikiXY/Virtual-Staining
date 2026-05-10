from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from virtual_staining.experiment.run_context import RunContext
    from virtual_staining.training.results import EpochMetrics, TrainingResult


class NullReporter:
    def info(self, message: str) -> None:
        pass

    def warning(self, message: str) -> None:
        pass

    def metric(self, name: str, value: float, step: int | None = None) -> None:
        pass

    def artifact(self, path: Path, kind: str) -> None:
        pass

    def on_training_started(self, context: RunContext) -> None:
        pass

    def on_epoch_completed(self, metrics: EpochMetrics) -> None:
        pass

    def on_checkpoint_saved(self, path: Path, epoch: int) -> None:
        pass

    def on_training_completed(self, result: TrainingResult) -> None:
        pass
