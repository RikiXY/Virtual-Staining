from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

_EVALUATION_KEYS: frozenset[str] = frozenset(
    {
        "dataset_root",
        "results_path",
        "run_name",
        "save_graphs",
        "target_dir",
        "generated_dir",
        "output_dir",
    }
)


@dataclass(frozen=True)
class EvaluationConfig:
    save_graphs: bool = False
    target_dir: Path | None = None
    generated_dir: Path | None = None
    output_dir: Path | None = None
