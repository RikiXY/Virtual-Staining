"""The prepared paired manifest, schema v4: ordered named inputs and targets per record.

Metadata (``manifest_metadata.json``) names ``input_modalities``, ``target_modalities``
and the ``reference_modality``; coordinates are upper-left patch origins and extents in
reference level-0 pixels with integer pixel centers, never model-resized tensor
coordinates. Every declared input and target is required in every record; a target's
foreground mask may be blank. Earlier schema versions are rejected, never converted.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from virtual_staining.config.validation import MODALITY_NAME_PATTERN
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

MANIFEST_SCHEMA_VERSION = "4.0"
COORDINATE_SPACE = "reference_level0_pixels"
PIXEL_CENTER = "integer"
_REQUIRED_METADATA = (
    "schema_version",
    "input_modalities",
    "target_modalities",
    "reference_modality",
    "coordinate_space",
    "pixel_center",
)
# Builder-owned producer fields that may accompany the canonical metadata.
_PRODUCER_METADATA: Mapping[str, type] = {"created_at": str, "record_count": int, "splits": dict}


def _validate_manifest_path(path: Path, field_name: str) -> None:
    if not path.parts or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{field_name} must be a relative, non-traversing path: {path!r}")


def _check_names(names: tuple[str, ...], field: str) -> None:
    if not names or len(set(names)) != len(names):
        raise ValueError(f"Manifest {field} must be non-empty and unique")
    invalid = [name for name in names if not MODALITY_NAME_PATTERN.match(name)]
    if invalid:
        raise ValueError(f"Manifest {field} contains unsafe identifiers {invalid}")


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


def _optional_path(row: Mapping[str, str], field: str, number: int, path: Path) -> Path | None:
    return _parse_path(row[field], field, number, path) if row[field].strip() else None


def _parse_int(value: str, field: str, row: int, path: Path) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ValueError(f"Manifest CSV {path}, row {row}: {field} must be an integer") from None


@dataclass(frozen=True)
class ManifestMetadata:
    schema_version: str
    input_modalities: tuple[str, ...]
    target_modalities: tuple[str, ...]
    reference_modality: str
    coordinate_space: str = COORDINATE_SPACE
    pixel_center: str = PIXEL_CENTER

    def __post_init__(self) -> None:
        if self.schema_version != MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                f"Manifest schema must be exactly {MANIFEST_SCHEMA_VERSION}, got "
                f"{self.schema_version!r}; earlier manifests are not supported, re-run "
                "'vs prepare'"
            )
        _check_names(self.input_modalities, "input_modalities")
        _check_names(self.target_modalities, "target_modalities")
        if set(self.input_modalities) & set(self.target_modalities):
            raise ValueError("Manifest target_modalities must differ from all input modalities")
        if self.reference_modality not in self.input_modalities:
            raise ValueError("Manifest reference_modality must be one of input_modalities")
        if self.coordinate_space != COORDINATE_SPACE:
            raise ValueError(f"Manifest coordinate_space must be {COORDINATE_SPACE!r}")
        if self.pixel_center != PIXEL_CENTER:
            raise ValueError(f"Manifest pixel_center must be {PIXEL_CENTER!r}")

    @classmethod
    def from_mapping(cls, value: object) -> ManifestMetadata:
        if not isinstance(value, dict):
            raise ValueError("Manifest metadata must be a JSON object")
        missing = [key for key in _REQUIRED_METADATA if key not in value]
        if missing:
            raise ValueError(f"Manifest metadata missing required fields: {missing}")
        unknown = sorted(set(value) - set(_REQUIRED_METADATA) - set(_PRODUCER_METADATA))
        if unknown:
            raise ValueError(f"Manifest metadata has unknown fields: {unknown}")
        for key, kind in _PRODUCER_METADATA.items():
            if key in value and (not isinstance(value[key], kind) or isinstance(value[key], bool)):
                raise ValueError(f"Manifest metadata {key} must be a {kind.__name__}")
        for key in ("schema_version", "reference_modality", "coordinate_space", "pixel_center"):
            if not isinstance(value[key], str):
                raise ValueError(f"Manifest metadata {key} must be a string")
        names: dict[str, tuple[str, ...]] = {}
        for key in ("input_modalities", "target_modalities"):
            items = value[key]
            if not isinstance(items, list) or not all(isinstance(item, str) for item in items):
                raise ValueError(f"Manifest metadata {key} must be a list of names")
            names[key] = tuple(items)
        return cls(
            schema_version=value["schema_version"],
            input_modalities=names["input_modalities"],
            target_modalities=names["target_modalities"],
            reference_modality=value["reference_modality"],
            coordinate_space=value["coordinate_space"],
            pixel_center=value["pixel_center"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "input_modalities": list(self.input_modalities),
            "target_modalities": list(self.target_modalities),
            "reference_modality": self.reference_modality,
            "coordinate_space": self.coordinate_space,
            "pixel_center": self.pixel_center,
        }


def manifest_fieldnames(metadata: ManifestMetadata) -> tuple[str, ...]:
    """The exact v4 CSV header: inputs, targets, then one mask column per target."""
    return (
        "sample_id",
        "set_id",
        "split",
        *(f"input__{name}" for name in metadata.input_modalities),
        *(f"target__{name}" for name in metadata.target_modalities),
        *(f"foreground_mask__{name}" for name in metadata.target_modalities),
        "x",
        "y",
        "width",
        "height",
    )


@dataclass(frozen=True)
class ManifestRecord:
    """One committed sample: every input and target on the same reference-level-0 grid.

    ``x``/``y`` are the patch's upper-left origin and ``width``/``height`` its extent in
    reference level-0 pixels.
    """

    sample_id: str
    set_id: str
    split: Split
    input_paths: Mapping[str, Path]
    target_paths: Mapping[str, Path]
    foreground_mask_paths: Mapping[str, Path | None]
    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if not self.sample_id.strip() or not self.set_id.strip():
            raise ValueError("ManifestRecord sample_id and set_id must be non-empty")
        if self.split not in MANIFEST_SPLITS:
            raise ValueError(f"ManifestRecord.split must be one of {sorted(MANIFEST_SPLITS)}")
        for field, value in (
            ("x", self.x),
            ("y", self.y),
            ("width", self.width),
            ("height", self.height),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"ManifestRecord.{field} must be an integer")
        if self.x < 0 or self.y < 0 or self.width <= 0 or self.height <= 0:
            raise ValueError(
                "ManifestRecord coordinates must be nonnegative and dimensions positive"
            )
        if not self.input_paths or not self.target_paths:
            raise ValueError("ManifestRecord input_paths and target_paths must not be empty")
        for role, paths in (("input_paths", self.input_paths), ("target_paths", self.target_paths)):
            for modality, path in paths.items():
                if not MODALITY_NAME_PATTERN.match(modality):
                    raise ValueError(f"ManifestRecord {role} has unsafe name {modality!r}")
                _validate_manifest_path(path, f"{role}[{modality!r}]")
        if set(self.input_paths) & set(self.target_paths):
            raise ValueError("ManifestRecord input and target names must be disjoint")
        if tuple(self.foreground_mask_paths) != tuple(self.target_paths):
            raise ValueError("ManifestRecord foreground_mask_paths must name exactly the targets")
        for modality, mask in self.foreground_mask_paths.items():
            if mask is not None:
                _validate_manifest_path(mask, f"foreground_mask_paths[{modality!r}]")
        images = [*self.input_paths.values(), *self.target_paths.values()]
        if len(set(images)) != len(images):
            raise ValueError("ManifestRecord input and target paths must all differ")

    def domain_path(self, name: str) -> Path:
        """The image of domain ``name``, an input or a target of this record."""
        if name in self.input_paths:
            return self.input_paths[name]
        if name in self.target_paths:
            return self.target_paths[name]
        raise KeyError(f"Manifest record {self.sample_id!r} has no domain {name!r}")


@dataclass(frozen=True)
class DatasetManifest:
    records: tuple[ManifestRecord, ...]
    dataset_root: Path
    metadata: ManifestMetadata

    SCHEMA_VERSION = MANIFEST_SCHEMA_VERSION

    @property
    def fieldnames(self) -> tuple[str, ...]:
        return manifest_fieldnames(self.metadata)

    def filter_split(self, split: Split) -> DatasetManifest:
        return DatasetManifest(
            tuple(record for record in self.records if record.split == split),
            self.dataset_root,
            self.metadata,
        )

    def validate(
        self, check_files_exist: bool = False, require_splits: set[str] | None = None
    ) -> None:
        inputs, targets = self.metadata.input_modalities, self.metadata.target_modalities
        for record in self.records:
            if tuple(record.input_paths) != inputs:
                raise ValueError(
                    f"Manifest record {record.sample_id!r} input keys "
                    f"{tuple(record.input_paths)} must exactly match {inputs}"
                )
            if tuple(record.target_paths) != targets:
                raise ValueError(
                    f"Manifest record {record.sample_id!r} target keys "
                    f"{tuple(record.target_paths)} must exactly match {targets}"
                )
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
        target_paths = [path for record in self.records for path in record.target_paths.values()]
        if len(target_paths) != len(set(target_paths)):
            raise ValueError("Duplicate target paths in manifest")
        if set(input_paths) & set(target_paths):
            raise ValueError("Manifest reuses a path as both an input and a target")
        if check_files_exist:
            root = self.dataset_root.resolve()
            for record in self.records:
                masks = (mask for mask in record.foreground_mask_paths.values() if mask)
                for path in (*record.input_paths.values(), *record.target_paths.values(), *masks):
                    full = self.dataset_root / path
                    if not full.exists():
                        raise FileNotFoundError(f"Manifest file not found: {full}")
                    if not full.resolve().is_relative_to(root):
                        raise ValueError(f"Manifest path {path} resolves outside the dataset root")
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
                    "x": record.x,
                    "y": record.y,
                    "width": record.width,
                    "height": record.height,
                }
                for name, value in record.input_paths.items():
                    row[f"input__{name}"] = value.as_posix()
                for name, value in record.target_paths.items():
                    row[f"target__{name}"] = value.as_posix()
                for name, mask in record.foreground_mask_paths.items():
                    row[f"foreground_mask__{name}"] = mask.as_posix() if mask else ""
                writer.writerow(row)

    @classmethod
    def from_csv(
        cls, path: Path, dataset_root: Path, metadata: ManifestMetadata | None = None
    ) -> DatasetManifest:
        if metadata is None:
            raise ValueError("ManifestMetadata is required to parse a v4 manifest")
        expected = manifest_fieldnames(metadata)
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            header = tuple(next(reader, ()))
            duplicated = sorted({name for name in header if header.count(name) > 1})
            if duplicated:
                raise ValueError(f"Manifest CSV at {path} has duplicate columns: {duplicated}")
            if header != expected:
                raise ValueError(
                    f"Manifest CSV at {path} must match exact v4 columns: {list(expected)}; "
                    f"got {list(header)}"
                )
            records: list[ManifestRecord] = []
            for row_num, cells in enumerate(reader, start=2):
                if len(cells) != len(header):
                    raise ValueError(
                        f"Manifest CSV {path}, row {row_num}: expected {len(header)} cells, "
                        f"got {len(cells)}"
                    )
                row = dict(zip(header, cells, strict=True))
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
                        target_paths={
                            name: _parse_path(
                                row[f"target__{name}"], f"target__{name}", row_num, path
                            )
                            for name in metadata.target_modalities
                        },
                        foreground_mask_paths={
                            name: _optional_path(row, f"foreground_mask__{name}", row_num, path)
                            for name in metadata.target_modalities
                        },
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
        raise ValueError(f"Invalid manifest metadata at {metadata_path}: {exc}") from exc
    return DatasetManifest.from_csv(
        manifest_path, dataset_root=project.dataset_root, metadata=metadata
    )


def require_model_modalities(
    manifest: DatasetManifest, inputs: Sequence[str], outputs: Sequence[str]
) -> None:
    """Fail unless the manifest provides every selected model input and output."""
    metadata = manifest.metadata
    missing = sorted(set(inputs) - set(metadata.input_modalities))
    if missing:
        raise ValueError(
            f"model inputs {missing} are not manifest input modalities "
            f"{list(metadata.input_modalities)}"
        )
    missing = sorted(set(outputs) - set(metadata.target_modalities))
    if missing:
        raise ValueError(
            f"model outputs {missing} are not manifest target modalities "
            f"{list(metadata.target_modalities)}"
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


def prepared_split_unit(project: ProjectConfig) -> str | None:
    """The ``split.unit`` recorded by preparation in ``split_assignment.csv``, if any."""
    path = DatasetLayout.from_project(project).split_assignment_path
    if not path.is_file():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        units = {row["unit"] for row in csv.DictReader(handle)}
    if len(units) > 1:
        raise ValueError(f"Split assignment {path} mixes units {sorted(units)}")
    return units.pop() if units else None


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
    target_names: Sequence[str],
    include_masks: bool,
    groups: Mapping[str, tuple[str, str]],
) -> list[AssetRow]:
    """Rows for the files a paired consumer actually reads from each manifest record.

    Targets and their foreground masks carry their own target name as ``domain``.
    """
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
        for name in target_names:
            rows.append(
                AssetRow(
                    locator=record.target_paths[name].as_posix(),
                    role="target",
                    domain=name,
                    **common,
                )
            )
            if include_masks:
                mask = record.foreground_mask_paths[name]
                if mask is None:
                    raise FileNotFoundError(
                        f"Foreground mask of target {name!r} is missing for {record.sample_id!r}"
                    )
                rows.append(AssetRow(locator=mask.as_posix(), role="mask", domain=name, **common))
    return rows
