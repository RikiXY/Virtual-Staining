from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectConfig:
    dataset_root: Path
    results_path: Path
    run_name: str
    image_size: tuple[int, int]

    @property
    def run_root(self) -> Path:
        return self.results_path / self.run_name

    @property
    def dataset_train_dir(self) -> Path:
        return self.dataset_root / "dataset_train"

    @property
    def dataset_val_dir(self) -> Path:
        return self.dataset_root / "dataset_val"

    @property
    def dataset_test_dir(self) -> Path:
        return self.dataset_root / "dataset_test"

    @property
    def manifest_path(self) -> Path:
        return self.dataset_root / "manifests" / "manifest.csv"
