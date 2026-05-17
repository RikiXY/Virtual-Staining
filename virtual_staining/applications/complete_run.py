from __future__ import annotations

from pathlib import Path

from virtual_staining.applications.run_stages import DEFAULT_FULL_RUN_STAGES, run_stages
from virtual_staining.config.run import RunConfig


def complete_run(config: RunConfig, config_path: Path) -> None:
    """Run the canonical full pipeline in stage order."""
    run_stages(config, config_path, DEFAULT_FULL_RUN_STAGES)
