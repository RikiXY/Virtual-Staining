from __future__ import annotations

import csv
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from virtual_staining.data.consumption import AssetRow
from virtual_staining.data.layout import DatasetLayout
from virtual_staining.split_contract import (
    DISCARDED_SPLIT,
    MANIFEST_SPLITS,
)
from virtual_staining.split_contract import (
    ManifestSplit as Split,
)
from virtual_staining.utils.hashing import sha256_file

if TYPE_CHECKING:
    from virtual_staining.config.project import ProjectConfig

MANIFEST_SCHEMA_VERSION = "3.0"


def _validate_manifest_path(path: Path, field_name: str) -> None:
    if not path.parts or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{field_name} must be a relative, non-traversing path: {path!r}")


def _parse_split(value: str, *, row: int | None = None, path: Path | None = None) -> Split:
    if value not in MANIFEST_SPLITS:
        location = f" in {path}, row {row}" if row is not None and path is not None else ""
        raise ValueError(f"Invalid split{location}: {value!r}")
    return cast(Split, value)


def _nonempty(value: str, field: str, row: int, path: Path) -> str:
    if not value.strip():
        raise ValueError(f"Manifest CSV {path}, row {row}: {field} must not be empty")
    return value


def _parse_path(value: str, field: str, row: int, path: Path) -> Path:
    if not value.strip():
        raise ValueError(f"Manifest CSV {path}, row {row}: {field} must not be empty")
    result = Path(value)
    try:
        _validate_manifest_path(result, field)
    except ValueError as exc:
        raise ValueError(f"Manifest CSV {path}, row {row}: {exc}") from None
    return result


def _parse_int(value: str, field: str, row: int, path: Path) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ValueError(f"Manifest CSV {path}, row {row}: {field} must be an integer") from None


@dataclass(frozen=True)
class ManifestMetadata:
    schema_version: str
    input_modalities: tuple[str, ...]
    reference_modality: str
    target_modality: str

    def __post_init__(self) -> None:
        if self.schema_version != MANIFEST_SCHEMA_VERSION:
            raise ValueError(f"Manifest schema must be exactly {MANIFEST_SCHEMA_VERSION}")
        if not self.input_modalities or len(set(self.input_modalities)) != len(
            self.input_modalities
        ):
            raise ValueError("Manifest input_modalities must be non-empty and unique")
        if self.reference_modality not in self.input_modalities:
            raise ValueError("Manifest reference_modality must be one of input_modalities")
        if not self.target_modality.strip() or self.target_modality in self.input_modalities:
            raise ValueError("Manifest target_modality must differ from all input modalities")

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> ManifestMetadata:
        required = {"schema_version", "input_modalities", "reference_modality", "target_modality"}
        missing = required - value.keys()
        if missing:
            raise ValueError(f"Manifest metadata missing required fields: {sorted(missing)}")
        modalities = value["input_modalities"]
        if isinstance(modalities, str) or not isinstance(modalities, list | tuple):
            raise ValueError("Manifest metadata input_modalities must be a list")
        return cls(
            str(value["schema_version"]),
            tuple(str(item) for item in modalities),
            str(value["reference_modality"]),
            str(value["target_modality"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "input_modalities": list(self.input_modalities),
            "reference_modality": self.reference_modality,
            "target_modality": self.target_modality,
        }


@dataclass(frozen=True)
class ManifestRecord:
    sample_id: str
    set_id: str
    split: Split
    input_paths: dict[str, Path]
    target_path: Path
    x: int
    y: int
    width: int
    height: int
    foreground_mask_path: Path | None = None

    def __post_init__(self) -> None:
        if not self.sample_id.strip() or not self.set_id.strip():
            raise ValueError("ManifestRecord sample_id and set_id must be non-empty")
        if self.split not in MANIFEST_SPLITS:
            raise ValueError(f"ManifestRecord.split must be one of {sorted(MANIFEST_SPLITS)}")
        if self.x < 0 or self.y < 0 or self.width <= 0 or self.height <= 0:
            raise ValueError(
                "ManifestRecord coordinates must be nonnegative and dimensions positive"
            )
        if not self.input_paths:
            raise ValueError("ManifestRecord.input_paths must not be empty")
        for modality, input_path in self.input_paths.items():
            if not modality.strip():
                raise ValueError("ManifestRecord input modality names must be non-empty")
            _validate_manifest_path(input_path, f"input_paths[{modality!r}]")
        _validate_manifest_path(self.target_path, "target_path")
        if self.target_path in self.input_paths.values():
            raise ValueError("ManifestRecord.target_path must differ from every input path")
        if self.foreground_mask_path is not None:
            _validate_manifest_path(self.foreground_mask_path, "foreground_mask_path")


@dataclass(frozen=True)
class DatasetManifest:
    records: tuple[ManifestRecord, ...]
    dataset_root: Path
    metadata: ManifestMetadata

    SCHEMA_VERSION = MANIFEST_SCHEMA_VERSION

    @property
    def fieldnames(self) -> tuple[str, ...]:
        return (
            "sample_id",
            "set_id",
            "split",
            *(f"input__{name}" for name in self.metadata.input_modalities),
            "target_path",
            "foreground_mask_path",
            "x",
            "y",
            "width",
            "height",
        )

    def filter_split(self, split: Split) -> DatasetManifest:
        return DatasetManifest(
            tuple(record for record in self.records if record.split == split),
            self.dataset_root,
            self.metadata,
        )

    def validate(
        self, check_files_exist: bool = False, require_splits: set[str] | None = None
    ) -> None:
        allowed = set(self.metadata.input_modalities)
        for record in self.records:
            if tuple(record.input_paths) != self.metadata.input_modalities:
                raise ValueError("Manifest input keys must exactly match metadata order")
            if set(record.input_paths) != allowed:
                raise ValueError("Manifest input keys must exactly match metadata")
            for path in record.input_paths.values():
                _validate_manifest_path(path, "input_path")
            _validate_manifest_path(record.target_path, "target_path")
        samples: dict[str, list[Split]] = defaultdict(list)
        for record in self.records:
            samples[record.sample_id].append(record.split)
        if any(
            len({split for split in splits if split != DISCARDED_SPLIT}) > 1
            for splits in samples.values()
        ):
            raise ValueError("Some sample_ids appear in multiple splits")
        if any(
            len(splits) > 1 and not (len(splits) == 2 and DISCARDED_SPLIT in splits)
            for splits in samples.values()
        ):
            raise ValueError("Duplicate sample_ids in manifest")
        input_paths = [path for record in self.records for path in record.input_paths.values()]
        if len(input_paths) != len(set(input_paths)):
            raise ValueError("Duplicate input paths in manifest")
        target_paths = [record.target_path for record in self.records]
        if len(target_paths) != len(set(target_paths)):
            raise ValueError("Duplicate target paths in manifest")
        if check_files_exist:
            for record in self.records:
                for path in (
                    *record.input_paths.values(),
                    record.target_path,
                    *((record.foreground_mask_path,) if record.foreground_mask_path else ()),
                ):
                    if not (self.dataset_root / path).exists():
                        raise FileNotFoundError(
                            f"Manifest file not found: {self.dataset_root / path}"
                        )
        if require_splits:
            for split in require_splits:
                if split not in MANIFEST_SPLITS:
                    raise ValueError(f"Invalid required split {split!r}")
                if not any(record.split == split for record in self.records):
                    raise ValueError(f"Manifest has no records for required split {split!r}")

    def to_csv(self, path: Path) -> None:
        self.validate()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
            writer.writeheader()
            for record in self.records:
                row: dict[str, Any] = {
                    "sample_id": record.sample_id,
                    "set_id": record.set_id,
                    "split": record.split,
                }
                row.update(
                    {
                        f"input__{name}": record.input_paths[name].as_posix()
                        for name in self.metadata.input_modalities
                    }
                )
                row.update(
                    {
                        "target_path": record.target_path.as_posix(),
                        "foreground_mask_path": record.foreground_mask_path.as_posix()
                        if record.foreground_mask_path
                        else "",
                        "x": record.x,
                        "y": record.y,
                        "width": record.width,
                        "height": record.height,
                    }
                )
                writer.writerow(row)

    @classmethod
    def from_csv(
        cls, path: Path, dataset_root: Path, metadata: ManifestMetadata | None = None
    ) -> DatasetManifest:
        if metadata is None:
            raise ValueError("ManifestMetadata is required to parse a v3 manifest")
        expected = (
            "sample_id",
            "set_id",
            "split",
            *(f"input__{name}" for name in metadata.input_modalities),
            "target_path",
            "foreground_mask_path",
            "x",
            "y",
            "width",
            "height",
        )
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != expected:
                raise ValueError(
                    f"Manifest CSV at {path} must match exact v3 columns: {list(expected)}"
                )
            records: list[ManifestRecord] = []
            for row_num, row in enumerate(reader, start=2):
                records.append(
                    ManifestRecord(
                        sample_id=_nonempty(row["sample_id"], "sample_id", row_num, path),
                        set_id=_nonempty(row["set_id"], "set_id", row_num, path),
                        split=_parse_split(row["split"], row=row_num, path=path),
                        input_paths={
                            name: _parse_path(
                                row[f"input__{name}"], f"input__{name}", row_num, path
                            )
                            for name in metadata.input_modalities
                        },
                        target_path=_parse_path(row["target_path"], "target_path", row_num, path),
                        foreground_mask_path=_parse_path(
                            row["foreground_mask_path"], "foreground_mask_path", row_num, path
                        )
                        if row["foreground_mask_path"].strip()
                        else None,
                        x=_parse_int(row["x"], "x", row_num, path),
                        y=_parse_int(row["y"], "y", row_num, path),
                        width=_parse_int(row["width"], "width", row_num, path),
                        height=_parse_int(row["height"], "height", row_num, path),
                    )
                )
        result = cls(tuple(records), dataset_root, metadata)
        result.validate()
        return result

    def __len__(self) -> int:
        return len(self.records)


def load_manifest_or_raise(project: ProjectConfig) -> DatasetManifest:
    layout = DatasetLayout.from_project(project)
    manifest_path = layout.manifest_path
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found at {manifest_path}. Run 'vs prepare'.")
    metadata_path = layout.manifest_metadata_path
    try:
        metadata = ManifestMetadata.from_mapping(
            json.loads(metadata_path.read_text(encoding="utf-8"))
        )
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid manifest metadata at {metadata_path}") from exc
    return DatasetManifest.from_csv(
        manifest_path, dataset_root=project.dataset_root, metadata=metadata
    )


def load_set_groups(project: ProjectConfig) -> dict[str, tuple[str, str]]:
    """Map prepared ``set_id`` to ``(specimen_id, patient_id)`` from ``slide_sets.csv``.

    Returns an empty mapping when the prepared dataset has no slide-set metadata; the rows
    then carry only ``set_id`` and no stronger biological unit can be claimed.
    """
    path = DatasetLayout.from_project(project).slide_sets_path
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            row["set_id"]: (
                (row.get("specimen_id") or "").strip(),
                (row.get("patient_id") or "").strip(),
            )
            for row in csv.DictReader(handle)
        }


def manifest_sources(project: ProjectConfig) -> dict[str, object]:
    """Prepared-dataset lineage of a manifest-backed stage: manifest and preparation identity."""
    layout = DatasetLayout.from_project(project)
    fingerprint = None
    if layout.dataset_fingerprint_path.is_file():
        fingerprint = json.loads(layout.dataset_fingerprint_path.read_text(encoding="utf-8")).get(
            "fingerprint"
        )
    return {
        "manifest_path": str(layout.manifest_path),
        "manifest_sha256": sha256_file(layout.manifest_path),
        "dataset_fingerprint": fingerprint,
    }


def paired_record_rows(
    records: Sequence[ManifestRecord],
    *,
    input_names: Sequence[str],
    target: str | None,
    include_mask: bool,
    groups: Mapping[str, tuple[str, str]],
) -> list[AssetRow]:
    """Rows for the files a paired consumer actually reads from each manifest record."""
    rows: list[AssetRow] = []
    for record in records:
        specimen, patient = groups.get(record.set_id, ("", ""))
        common: dict[str, Any] = {
            "root": "dataset",
            "split": record.split,
            "sample_id": record.sample_id,
            "set_id": record.set_id,
            "specimen_id": specimen,
            "patient_id": patient,
        }
        for name in input_names:
            rows.append(
                AssetRow(
                    locator=record.input_paths[name].as_posix(),
                    role="input",
                    domain=name,
                    **common,
                )
            )
        if target is not None:
            rows.append(
                AssetRow(
                    locator=record.target_path.as_posix(), role="target", domain=target, **common
                )
            )
        if include_mask:
            if record.foreground_mask_path is None:
                raise FileNotFoundError(f"Foreground mask path is missing for {record.sample_id!r}")
            rows.append(
                AssetRow(
                    locator=record.foreground_mask_path.as_posix(),
                    role="mask",
                    domain="foreground_mask",
                    **common,
                )
            )
    return rows
