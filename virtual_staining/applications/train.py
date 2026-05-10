from __future__ import annotations

from pathlib import Path

from virtual_staining.config.run import RunConfig
from virtual_staining.training.results import TrainingResult
from virtual_staining.training.runner import run_training


def train(config: RunConfig, config_path: Path) -> TrainingResult:
    """Application-level training entry point."""
    return run_training(config, config_path)
