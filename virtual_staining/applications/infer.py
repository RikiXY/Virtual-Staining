from __future__ import annotations

from pathlib import Path

from virtual_staining.config.run import RunConfig
from virtual_staining.inference.results import InferenceResult
from virtual_staining.inference.runner import run_inference


def infer(config: RunConfig, config_path: Path) -> InferenceResult:
    """Application-level inference entry point."""
    return run_inference(config, config_path)
