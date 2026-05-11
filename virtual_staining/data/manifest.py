from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

Split = Literal["train", "val", "test", "discarded"]

_VALID_SPLITS: frozenset[str] = frozenset({"train", "val", "test", "discarded"})


def _parse_split(value: str) -> Split:
    if value not in _VALID_SPLITS:
        raise ValueError(f"Invalid split {value!r}; expected one of {sorted(_VALID_SPLITS)}")
    return cast(Split, value)


@dataclass(frozen=True)
class ManifestRecord:
    sample_id: str
    split: Split
    input_path: Path
    target_path: Path
    input_modality: str
    target_modality: str
    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True)
class DatasetManifest:
    records: tuple[ManifestRecord, ...]
    dataset_root: Path

    _FIELDNAMES = (
        "sample_id",
        "split",
        "input_path",
        "target_path",
        "input_modality",
        "target_modality",
        "x",
        "y",
        "width",
        "height",
    )

    def filter_split(self, split: Split) -> DatasetManifest:
        """Return a new manifest containing only records with the given split."""
        return DatasetManifest(
            records=tuple(record for record in self.records if record.split == split),
            dataset_root=self.dataset_root,
        )

    def resolved_input_paths(self) -> list[Path]:
        return [self.dataset_root / record.input_path for record in self.records]

    def resolved_target_paths(self) -> list[Path]:
        return [self.dataset_root / record.target_path for record in self.records]

    def validate(self, check_files_exist: bool = False) -> None:
        """
        Raise if the manifest is inconsistent.

        Checks:
        - sample_ids are unique
        - no duplicate (split, input_path) pairs
        - if check_files_exist=True, every resolved path must exist on disk
        """
        sample_ids = [record.sample_id for record in self.records]
        if len(sample_ids) != len(set(sample_ids)):
            duplicate_ids = {
                sample_id for sample_id in sample_ids if sample_ids.count(sample_id) > 1
            }
            raise ValueError(f"Duplicate sample_ids in manifest: {duplicate_ids}")

        split_input_pairs = [(record.split, record.input_path) for record in self.records]
        if len(split_input_pairs) != len(set(split_input_pairs)):
            duplicate_pairs = {
                pair for pair in split_input_pairs if split_input_pairs.count(pair) > 1
            }
            raise ValueError(f"Duplicate (split, input_path) pairs in manifest: {duplicate_pairs}")

        if check_files_exist:
            for record in self.records:
                input_path = self.dataset_root / record.input_path
                target_path = self.dataset_root / record.target_path
                if not input_path.exists():
                    raise FileNotFoundError(f"Input file not found: {input_path}")
                if not target_path.exists():
                    raise FileNotFoundError(f"Target file not found: {target_path}")

    def to_csv(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self._FIELDNAMES)
            writer.writeheader()
            for record in self.records:
                writer.writerow(
                    {
                        "sample_id": record.sample_id,
                        "split": record.split,
                        "input_path": record.input_path.as_posix(),
                        "target_path": record.target_path.as_posix(),
                        "input_modality": record.input_modality,
                        "target_modality": record.target_modality,
                        "x": record.x,
                        "y": record.y,
                        "width": record.width,
                        "height": record.height,
                    }
                )

    @classmethod
    def from_csv(cls, path: Path, dataset_root: Path) -> DatasetManifest:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            records = tuple(
                ManifestRecord(
                    sample_id=row["sample_id"],
                    split=_parse_split(row["split"]),
                    input_path=Path(row["input_path"]),
                    target_path=Path(row["target_path"]),
                    input_modality=row["input_modality"],
                    target_modality=row["target_modality"],
                    x=int(row["x"]),
                    y=int(row["y"]),
                    width=int(row["width"]),
                    height=int(row["height"]),
                )
                for row in reader
            )
        return cls(records=records, dataset_root=dataset_root)

    def __len__(self) -> int:
        return len(self.records)
