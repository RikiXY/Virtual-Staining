from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectConfig:
    dataset_root: Path
    results_path: Path
    run_name: str
    image_size: tuple[int, int]
    manifest_path_override: Path | None = None

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
        if self.manifest_path_override is not None:
            return self.manifest_path_override
        return self.dataset_root / "manifests" / "manifest.csv"

    def validate(self) -> None:
        if not self.run_name.strip():
            raise ValueError("run_name must be a non-empty string")
        width, height = self.image_size
        if width <= 0 or height <= 0:
            raise ValueError("image_size must contain two positive integers")
