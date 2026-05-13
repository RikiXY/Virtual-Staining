from __future__ import annotations

from pathlib import Path

from virtual_staining.applications.evaluate import evaluate
from virtual_staining.applications.infer import infer
from virtual_staining.applications.prepare import prepare
from virtual_staining.applications.train import train
from virtual_staining.config.run import RunConfig


def complete_run(config: RunConfig, config_path: Path) -> None:
    """Run the canonical full pipeline in stage order."""
    prepare(config, config_path)
    train(config, config_path)
    infer(config, config_path)
    evaluate(config, config_path)
