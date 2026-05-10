from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from virtual_staining.experiment.run_context import RunContext
    from virtual_staining.training.results import EpochMetrics, TrainingResult


class LoggingReporter:
    def __init__(self, logger: logging.Logger | None = None) -> None:
        self._logger = logger or logging.getLogger("virtual_staining")

    def info(self, message: str) -> None:
        self._logger.info(message)

    def warning(self, message: str) -> None:
        self._logger.warning(message)

    def metric(self, name: str, value: float, step: int | None = None) -> None:
        step_part = f" step={step}" if step is not None else ""
        self._logger.info(f"metric {name}={value:.6f}{step_part}")

    def artifact(self, path: Path, kind: str) -> None:
        self._logger.info(f"artifact [{kind}] {path}")

    def on_training_started(self, context: RunContext) -> None:
        self._logger.info(
            f"Training started: run={context.name!r}, hash={context.config_hash[:16]}"
        )

    def on_epoch_completed(self, metrics: EpochMetrics) -> None:
        self._logger.info(
            f"Epoch completed: loss_G={metrics.loss_G:.4f}, loss_D={metrics.loss_D:.4f}"
        )

    def on_checkpoint_saved(self, path: Path, epoch: int) -> None:
        self._logger.info(f"Checkpoint saved at epoch {epoch}: {path}")

    def on_training_completed(self, result: TrainingResult) -> None:
        self._logger.info(
            f"Training completed at epoch {result.final_epoch}. "
            f"Best checkpoint: {result.best_checkpoint_path}"
        )
