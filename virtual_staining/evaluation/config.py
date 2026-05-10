from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EvaluationConfig:
    save_graphs: bool = False
    target_dir: Path | None = None
    generated_dir: Path | None = None
    output_dir: Path | None = None
