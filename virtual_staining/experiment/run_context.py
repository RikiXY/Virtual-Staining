from __future__ import annotations

from dataclasses import dataclass

from virtual_staining.experiment.run_paths import RunPaths


@dataclass(frozen=True)
class RunContext:
    name: str
    paths: RunPaths
    seed: int | None
    device: str | None
    config_hash: str
