from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import cast

from virtual_staining.applications.evaluate import evaluate
from virtual_staining.applications.infer import infer
from virtual_staining.applications.prepare import prepare
from virtual_staining.applications.train import ProgressReporter, train
from virtual_staining.config.run import RunConfig
from virtual_staining.experiment.stages import VALID_STAGES, StageName

DEFAULT_FULL_RUN_STAGES = VALID_STAGES


def run_stage(
    config_path: Path,
    stage: str,
    *,
    progress_reporter: ProgressReporter | None = None,
) -> object:
    """Run one user-visible experiment stage."""
    return run_stages(config_path, (stage,), progress_reporter=progress_reporter)[
        cast(StageName, stage)
    ]


def run_stages(
    config_path: Path,
    stages: Sequence[str] = DEFAULT_FULL_RUN_STAGES,
    *,
    progress_reporter: ProgressReporter | None = None,
) -> dict[StageName, object]:
    """Load one run config and execute selected stages in order."""
    unknown = [stage for stage in stages if stage not in VALID_STAGES]
    if unknown:
        raise ValueError(
            f"Unknown stage(s): {', '.join(unknown)}. Allowed stages: {', '.join(VALID_STAGES)}"
        )

    path = config_path.resolve()
    config = RunConfig.from_yaml(path)
    results: dict[StageName, object] = {}
    for stage in stages:
        stage_name = cast(StageName, stage)
        if stage_name == "prepare":
            results[stage_name] = prepare(config, path)
        elif stage_name == "train":
            results[stage_name] = train(config, path, progress_reporter=progress_reporter)
        elif stage_name == "infer":
            results[stage_name] = infer(config, path)
        else:
            results[stage_name] = evaluate(config, path)
    return results
