from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class InferenceResult:
    output_dir: Path
    generated_paths: list[Path] = field(default_factory=list)
    num_samples: int = 0
