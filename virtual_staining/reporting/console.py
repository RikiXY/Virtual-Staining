from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from virtual_staining.utils.console import print_info, print_section, style

if TYPE_CHECKING:
    from virtual_staining.experiment.run_context import RunContext
    from virtual_staining.training.results import EpochMetrics, TrainingResult


class ConsoleReporter:
    def info(self, message: str) -> None:
        print(style("[INFO]", "cyan"), message)

    def warning(self, message: str) -> None:
        print(style("[WARN]", "yellow"), message)

    def metric(self, name: str, value: float, step: int | None = None) -> None:
        step_part = f" (step {step})" if step is not None else ""
        print(f"  {style(name, 'bold')}: {value:.6f}{step_part}")

    def artifact(self, path: Path, kind: str) -> None:
        print_info(f"Artifact [{kind}]", str(path))

    def on_training_started(self, context: RunContext) -> None:
        print_section("Training started")
        print_info("Run", context.name)
        print_info("Device", context.device or "cpu")
        print_info("Config hash", context.config_hash[:16])

    def on_epoch_completed(self, metrics: EpochMetrics) -> None:
        print(f"  loss_G: {metrics.loss_G:.4f}  loss_D: {metrics.loss_D:.4f}")

    def on_checkpoint_saved(self, path: Path, epoch: int) -> None:
        print_info(f"Checkpoint (ep {epoch})", str(path))

    def on_training_completed(self, result: TrainingResult) -> None:
        print_section("Training completed")
        print_info("Final epoch", str(result.final_epoch))
        if result.best_checkpoint_path:
            print_info("Best checkpoint", str(result.best_checkpoint_path))
