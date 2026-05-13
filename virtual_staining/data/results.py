from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetBuildResult:
    train_count: int
    val_count: int
    test_count: int
    skipped_count: int
    output_root: Path
    reused: bool = False
