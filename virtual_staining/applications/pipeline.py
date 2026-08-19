from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal, TypeAlias, cast

from virtual_staining.applications.evaluate import evaluate
from virtual_staining.applications.prepare import prepare
from virtual_staining.config.run import RunConfig
from virtual_staining.inference.runner import run_inference
from virtual_staining.training.runner import run_training

StageName = Literal["prepare", "train", "infer", "evaluate"]
VALID_STAGES: tuple[StageName, ...] = ("prepare", "train", "infer", "evaluate")
DEFAULT_FULL_RUN_STAGES = VALID_STAGES
StageHandler: TypeAlias = Callable[[RunConfig, Path], object]


def run_stage(config_path: Path, stage: str) -> object:
    """Run one user-visible experiment stage."""
    return run_stages(config_path, (stage,))[cast(StageName, stage)]


def run_stages(
    config_path: Path,
    stages: Sequence[str] = DEFAULT_FULL_RUN_STAGES,
) -> dict[StageName, object]:
    """Load one run config and execute selected stages in order."""
    unknown = [stage for stage in stages if stage not in VALID_STAGES]
    if unknown:
        raise ValueError(
            f"Unknown stage(s): {', '.join(unknown)}. Allowed stages: {', '.join(VALID_STAGES)}"
        )

    path = config_path.resolve()
    config = RunConfig.from_yaml(path)
    handlers: dict[StageName, StageHandler] = {
        "prepare": prepare,
        "train": run_training,
        "infer": run_inference,
        "evaluate": evaluate,
    }
    return {
        cast(StageName, stage): handlers[cast(StageName, stage)](config, path) for stage in stages
    }
