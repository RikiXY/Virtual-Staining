from __future__ import annotations

import csv
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

if TYPE_CHECKING:
    from virtual_staining.config.project import ProjectConfig

Split = Literal["train", "val", "test", "discarded"]

_VALID_SPLITS: frozenset[str] = frozenset({"train", "val", "test", "discarded"})
MANIFEST_SCHEMA_VERSION = "1.0"
_REQUIRED_FIELDNAMES = (
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

logger = logging.getLogger(__name__)


def _parse_split(value: str) -> Split:
    if value not in _VALID_SPLITS:
        raise ValueError(f"Invalid split {value!r}; expected one of {sorted(_VALID_SPLITS)}")
    return cast(Split, value)


def _validate_manifest_path(path: Path, field_name: str) -> None:
    """Raise ValueError if a manifest path is empty, absolute, or contains traversal."""
    if not path.parts:
        raise ValueError(f"ManifestRecord.{field_name} must not be empty")
    if path.is_absolute():
        raise ValueError(
            f"ManifestRecord.{field_name} must be a relative path, got absolute path: {path!r}"
        )
    if ".." in path.parts:
        raise ValueError(
            f"ManifestRecord.{field_name} must not contain '..' components, got: {path!r}"
        )


def _parse_int_field(value: str, field: str, row_num: int, csv_path: Path) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"Manifest CSV {csv_path}, row {row_num}: "
            f"field '{field}' must be an integer, got {value!r}"
        ) from None


def _parse_split_field(value: str, row_num: int, csv_path: Path) -> Split:
    try:
        return _parse_split(value)
    except ValueError:
        raise ValueError(
            f"Manifest CSV {csv_path}, row {row_num}: "
            f"split must be one of {sorted(_VALID_SPLITS)}, got {value!r}"
        ) from None


def _require_nonempty(value: str, field: str, row_num: int, csv_path: Path) -> str:
    if not value.strip():
        raise ValueError(
            f"Manifest CSV {csv_path}, row {row_num}: field '{field}' must not be empty"
        )
    return value


def _parse_path_field(value: str, field: str, row_num: int, csv_path: Path) -> Path:
    path = Path(value)
    try:
        _validate_manifest_path(path, field)
    except ValueError as exc:
        raise ValueError(f"Manifest CSV {csv_path}, row {row_num}: {exc}") from None
    return path


def load_manifest_or_raise(project: ProjectConfig) -> DatasetManifest:
    """Load the project manifest, raising a clear error when it is missing."""
    manifest_path = project.manifest_path
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found at {manifest_path}. "
            "Run 'vs-prepare' or set 'manifest_path' in your run config."
        )
    return DatasetManifest.from_csv(manifest_path, dataset_root=project.dataset_root)


def _warn_on_schema_version_mismatch(manifest_path: Path, expected_version: str) -> None:
    metadata_path = manifest_path.parent / "manifest_metadata.json"
    if not metadata_path.exists():
        return

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    on_disk_version = metadata.get("schema_version", "unknown")
    if on_disk_version != expected_version:
        logger.warning(
            "Manifest schema version mismatch: file has '%s', code expects '%s'. "
            "Re-run 'vs-prepare' if you see unexpected errors.",
            on_disk_version,
            expected_version,
        )


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

    def __post_init__(self) -> None:
        if not self.sample_id.strip():
            raise ValueError("ManifestRecord.sample_id must be a non-empty string")
        if self.split not in _VALID_SPLITS:
            raise ValueError(
                f"ManifestRecord.split must be one of {sorted(_VALID_SPLITS)}, got {self.split!r}"
            )
        if not self.input_modality.strip():
            raise ValueError("ManifestRecord.input_modality must be a non-empty string")
        if not self.target_modality.strip():
            raise ValueError("ManifestRecord.target_modality must be a non-empty string")
        if self.x < 0:
            raise ValueError(f"ManifestRecord.x must be >= 0, got {self.x}")
        if self.y < 0:
            raise ValueError(f"ManifestRecord.y must be >= 0, got {self.y}")
        if self.width <= 0:
            raise ValueError(f"ManifestRecord.width must be > 0, got {self.width}")
        if self.height <= 0:
            raise ValueError(f"ManifestRecord.height must be > 0, got {self.height}")
        if self.input_path == self.target_path:
            raise ValueError(
                "ManifestRecord.input_path and target_path must be different, "
                f"got {self.input_path!r}"
            )
        _validate_manifest_path(self.input_path, "input_path")
        _validate_manifest_path(self.target_path, "target_path")


@dataclass(frozen=True)
class DatasetManifest:
    SCHEMA_VERSION = MANIFEST_SCHEMA_VERSION
    records: tuple[ManifestRecord, ...]
    dataset_root: Path

    _FIELDNAMES = _REQUIRED_FIELDNAMES

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

    def validate(
        self,
        check_files_exist: bool = False,
        require_splits: set[str] | None = None,
    ) -> None:
        """
        Raise if the manifest is inconsistent.

        Checks:
        - sample_ids are unique
        - no sample_id appears in more than one non-discarded split
        - no duplicate (split, input_path) pairs
        - no duplicate input_path values
        - no duplicate target_path values
        - if check_files_exist=True, every resolved path must exist on disk
        - if require_splits is provided, each listed split must have at least one record
        """
        splits_by_sample: dict[str, list[Split]] = defaultdict(list)
        for record in self.records:
            splits_by_sample[record.sample_id].append(record.split)

        multi_split = {
            sample_id: sorted({split for split in splits if split != "discarded"})
            for sample_id, splits in splits_by_sample.items()
            if len({split for split in splits if split != "discarded"}) > 1
        }
        if multi_split:
            raise ValueError(f"Some sample_ids appear in multiple splits: {multi_split}")

        duplicate_ids = {
            sample_id
            for sample_id, splits in splits_by_sample.items()
            if len(splits) > 1 and not (len(splits) == 2 and "discarded" in splits)
        }
        if duplicate_ids:
            raise ValueError(f"Duplicate sample_ids in manifest: {duplicate_ids}")

        split_input_pairs = [(record.split, record.input_path) for record in self.records]
        if len(split_input_pairs) != len(set(split_input_pairs)):
            duplicate_pairs = {
                pair for pair in split_input_pairs if split_input_pairs.count(pair) > 1
            }
            raise ValueError(f"Duplicate (split, input_path) pairs in manifest: {duplicate_pairs}")

        input_paths = [record.input_path for record in self.records]
        if len(input_paths) != len(set(input_paths)):
            duplicate_input_paths = {
                input_path for input_path in input_paths if input_paths.count(input_path) > 1
            }
            raise ValueError(f"Duplicate input_path values in manifest: {duplicate_input_paths}")

        target_paths = [record.target_path for record in self.records]
        if len(target_paths) != len(set(target_paths)):
            duplicate_target_paths = {
                target_path for target_path in target_paths if target_paths.count(target_path) > 1
            }
            raise ValueError(f"Duplicate target_path values in manifest: {duplicate_target_paths}")

        if check_files_exist:
            for record in self.records:
                input_path = self.dataset_root / record.input_path
                target_path = self.dataset_root / record.target_path
                if not input_path.exists():
                    raise FileNotFoundError(f"Input file not found: {input_path}")
                if not target_path.exists():
                    raise FileNotFoundError(f"Target file not found: {target_path}")

        if require_splits:
            present_splits = sorted({record.split for record in self.records})
            for split in sorted(require_splits):
                if split not in _VALID_SPLITS:
                    raise ValueError(
                        f"Invalid required split {split!r}; expected one of {sorted(_VALID_SPLITS)}"
                    )
                if len(self.filter_split(cast(Split, split)).records) == 0:
                    raise ValueError(
                        f"Manifest has no records for required split '{split}'. "
                        f"Present splits: {present_splits}. "
                        "Check that 'vs-prepare' completed successfully and that "
                        "the correct manifest is being used."
                    )

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
            fieldnames = reader.fieldnames or []
            missing = [field for field in cls._FIELDNAMES if field not in fieldnames]
            if missing:
                raise ValueError(f"Manifest CSV at {path} is missing required columns: {missing}")
            unexpected = [field for field in fieldnames if field not in cls._FIELDNAMES]
            if unexpected:
                raise ValueError(f"Manifest CSV at {path} has unexpected columns: {unexpected}")
            records_list: list[ManifestRecord] = []
            for row_num, row in enumerate(reader, start=2):
                records_list.append(
                    ManifestRecord(
                        sample_id=_require_nonempty(row["sample_id"], "sample_id", row_num, path),
                        split=_parse_split_field(row["split"], row_num, path),
                        input_path=_parse_path_field(
                            row["input_path"], "input_path", row_num, path
                        ),
                        target_path=_parse_path_field(
                            row["target_path"], "target_path", row_num, path
                        ),
                        input_modality=_require_nonempty(
                            row["input_modality"], "input_modality", row_num, path
                        ),
                        target_modality=_require_nonempty(
                            row["target_modality"], "target_modality", row_num, path
                        ),
                        x=_parse_int_field(row["x"], "x", row_num, path),
                        y=_parse_int_field(row["y"], "y", row_num, path),
                        width=_parse_int_field(row["width"], "width", row_num, path),
                        height=_parse_int_field(row["height"], "height", row_num, path),
                    )
                )
            records = tuple(records_list)
        _warn_on_schema_version_mismatch(path, cls.SCHEMA_VERSION)
        return cls(records=records, dataset_root=dataset_root)

    def __len__(self) -> int:
        return len(self.records)
