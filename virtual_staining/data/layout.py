from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from virtual_staining.config.project import ProjectConfig

DatasetSplit = Literal["train", "val", "test"]


@dataclass(frozen=True)
class DatasetLayout:
    root: Path
    manifest_path_override: Path | None = None

    @classmethod
    def from_project(cls, project: ProjectConfig) -> DatasetLayout:
        return cls(project.dataset_root, project.manifest_path_override)

    @property
    def config_dir(self) -> Path:
        return self.root / "config"

    @property
    def metadata_dir(self) -> Path:
        return self.root / "metadata"

    @property
    def manifests_dir(self) -> Path:
        return self.root / "manifests"

    @property
    def splits_dir(self) -> Path:
        return self.root / "splits"

    @property
    def discarded_patches_dir(self) -> Path:
        return self.root / "discarded_patches"

    @property
    def manifest_path(self) -> Path:
        return self.manifest_path_override or self.manifests_dir / "manifest.csv"

    @property
    def discarded_manifest_path(self) -> Path:
        return self.manifests_dir / "discarded_manifest.csv"

    @property
    def slide_sets_path(self) -> Path:
        return self.manifests_dir / "slide_sets.csv"

    @property
    def manifest_metadata_path(self) -> Path:
        return self.manifests_dir / "manifest_metadata.json"

    @property
    def dataset_build_path(self) -> Path:
        return self.metadata_dir / "dataset_build.json"

    @property
    def dataset_fingerprint_path(self) -> Path:
        return self.metadata_dir / "dataset_fingerprint.json"

    @property
    def input_hashes_path(self) -> Path:
        return self.metadata_dir / "input_hashes.json"

    @property
    def split_assignment_path(self) -> Path:
        return self.metadata_dir / "split_assignment.csv"

    @property
    def input_config_path(self) -> Path:
        return self.config_dir / "input.yaml"

    @property
    def resolved_config_path(self) -> Path:
        return self.config_dir / "resolved.yaml"

    @property
    def environment_path(self) -> Path:
        return self.metadata_dir / "environment.json"

    @property
    def config_hash_path(self) -> Path:
        return self.metadata_dir / "config_hash.txt"

    def split_dir(self, split: DatasetSplit) -> Path:
        return self.splits_dir / split
