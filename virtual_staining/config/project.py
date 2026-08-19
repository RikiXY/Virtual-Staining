from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from virtual_staining.config.validation import reject_unknown_keys
from virtual_staining.utils.dimensions import parse_wh_size

_PROJECT_KEYS = frozenset(
    {"dataset_root", "results_path", "run_name", "image_size", "manifest_path"}
)

SplitName = Literal["train", "val", "test"]


@dataclass(frozen=True)
class ProjectConfig:
    dataset_root: Path
    results_path: Path
    run_name: str
    image_size: tuple[int, int]
    manifest_path_override: Path | None = None

    def __post_init__(self) -> None:
        self.validate()

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> ProjectConfig:
        project_data = {key: value for key, value in data.items() if key in _PROJECT_KEYS}
        reject_unknown_keys(project_data, _PROJECT_KEYS, "project")
        manifest_path = project_data.get("manifest_path")
        return cls(
            dataset_root=Path(project_data["dataset_root"]),
            results_path=Path(project_data["results_path"]),
            run_name=str(project_data["run_name"]),
            image_size=parse_wh_size(project_data.get("image_size"), (256, 256)),
            manifest_path_override=Path(manifest_path) if manifest_path else None,
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "dataset_root": str(self.dataset_root),
            "results_path": str(self.results_path),
            "run_name": self.run_name,
            "image_size": list(self.image_size),
        }
        if self.manifest_path_override is not None:
            data["manifest_path"] = str(self.manifest_path_override)
        return data

    @property
    def run_root(self) -> Path:
        return self.results_path / self.run_name

    @property
    def splits_dir(self) -> Path:
        return self.dataset_root / "splits"

    def split_dir(self, split: SplitName) -> Path:
        return self.splits_dir / split

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
