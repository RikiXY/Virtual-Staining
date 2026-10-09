from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from virtual_staining.utils.dimensions import parse_wh_size

PROJECT_KEYS = frozenset(
    {"dataset_root", "results_path", "run_name", "image_size", "manifest_path"}
)


@dataclass(frozen=True)
class ProjectConfig:
    dataset_root: Path
    results_path: Path | None
    run_name: str | None
    image_size: tuple[int, int]
    manifest_path_override: Path | None = None

    def __post_init__(self) -> None:
        self.validate()

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> ProjectConfig:
        project_data = {key: value for key, value in data.items() if key in PROJECT_KEYS}
        for name in ("dataset_root", "results_path", "manifest_path", "run_name"):
            if name in project_data and (
                not isinstance(project_data[name], str) or not project_data[name].strip()
            ):
                raise TypeError(f"{name} must be a non-empty string")
        if "dataset_root" not in project_data:
            raise ValueError("dataset_root is required")
        manifest_path = project_data.get("manifest_path")
        return cls(
            dataset_root=Path(project_data["dataset_root"]),
            results_path=Path(project_data["results_path"])
            if "results_path" in project_data
            else None,
            run_name=project_data.get("run_name"),
            image_size=parse_wh_size(project_data.get("image_size"), (256, 256)),
            manifest_path_override=Path(manifest_path) if manifest_path else None,
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "dataset_root": str(self.dataset_root),
            "image_size": list(self.image_size),
        }
        if self.results_path is not None:
            data["results_path"] = str(self.results_path)
        if self.run_name is not None:
            data["run_name"] = self.run_name
        if self.manifest_path_override is not None:
            data["manifest_path"] = str(self.manifest_path_override)
        return data

    def validate(self) -> None:
        if self.run_name is not None and not self.run_name.strip():
            raise ValueError("run_name must be a non-empty string")
        width, height = self.image_size
        if width <= 0 or height <= 0:
            raise ValueError("image_size must contain two positive integers")
