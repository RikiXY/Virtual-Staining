from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TypeAlias

from virtual_staining.applications.evaluate import evaluate
from virtual_staining.applications.infer import infer
from virtual_staining.applications.prepare import prepare
from virtual_staining.applications.train import train
from virtual_staining.config.run import RunConfig

VALID_STAGES = (
    "prepare",
    "train",
    "infer",
    "evaluate",
)

DEFAULT_FULL_RUN_STAGES = (
    "prepare",
    "train",
    "infer",
    "evaluate",
)

StageHandler: TypeAlias = Callable[[RunConfig, Path], object]


def run_stages(
    config: RunConfig,
    config_path: Path,
    stages: Sequence[str] = DEFAULT_FULL_RUN_STAGES,
) -> None:
    """Run selected pipeline stages in the requested order."""
    stage_handlers: dict[str, StageHandler] = {
        "prepare": prepare,
        "train": train,
        "infer": infer,
        "evaluate": evaluate,
    }

    unknown_stages = [stage for stage in stages if stage not in stage_handlers]
    if unknown_stages:
        allowed = ", ".join(VALID_STAGES)
        unknown = ", ".join(unknown_stages)
        raise ValueError(f"Unknown stage(s): {unknown}. Allowed stages: {allowed}")

    for stage in stages:
        stage_handlers[stage](config, config_path)
