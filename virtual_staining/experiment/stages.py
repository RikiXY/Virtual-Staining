from __future__ import annotations

from typing import Literal

StageName = Literal["prepare", "train", "infer", "evaluate"]
RunStageName = Literal["train", "infer", "evaluate"]
VALID_STAGES: tuple[StageName, ...] = ("prepare", "train", "infer", "evaluate")
RUN_STAGES: tuple[RunStageName, ...] = ("train", "infer", "evaluate")
