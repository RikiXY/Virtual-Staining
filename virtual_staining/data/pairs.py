from __future__ import annotations

import csv
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from virtual_staining.data.config import PreprocessingConfig

PAIR_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
PAIR_INVENTORY_FIELDS = (
    "pair_id",
    "source_path",
    "target_path",
    "already_aligned",
    "shared_mask_path",
    "source_mask_path",
    "target_mask_path",
    "patient_id",
    "specimen_id",
    "source_slide_id",
    "target_slide_id",
)
PAIR_MANIFEST_FIELDS = (
    "pair_id",
    "split",
    "source_path",
    "target_path",
    "patient_id",
    "specimen_id",
    "source_slide_id",
    "target_slide_id",
    "already_aligned",
    "shared_mask_path",
    "source_mask_path",
    "target_mask_path",
    "status",
    "alignment_method",
    "alignment_metadata_path",
)
_REQUIRED_FIELDS = ("pair_id", "source_path", "target_path")


@dataclass(frozen=True)
class SlidePair:
    pair_id: str
    source_path: Path
    target_path: Path
    already_aligned: bool | None = None
    shared_mask_path: Path | None = None
    source_mask_path: Path | None = None
    target_mask_path: Path | None = None
    patient_id: str | None = None
    specimen_id: str | None = None
    source_slide_id: str | None = None
    target_slide_id: str | None = None


def _optional_text(value: str | None) -> str | None:
    stripped = (value or "").strip()
    return stripped or None


def _parse_bool(value: str | None, *, row: int) -> bool | None:
    value = (value or "").strip().lower()
    if not value:
        return None
    if value == "true":
        return True
    if value == "false":
        return False
    raise ValueError(f"Pair inventory row {row}: already_aligned must be true, false, or blank")


def _resolve_relative_path(
    value: str | None,
    *,
    field: str,
    row: int,
    dataset_root: Path,
    required: bool = False,
) -> Path | None:
    text = (value or "").strip()
    if not text:
        if required:
            raise ValueError(f"Pair inventory row {row}: {field} must not be empty")
        return None
    path = Path(text)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Pair inventory row {row}: {field} must stay inside dataset_root")
    root = dataset_root.resolve()
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"Pair inventory row {row}: {field} resolves outside dataset_root")
    if not resolved.is_file():
        raise FileNotFoundError(f"Pair inventory row {row}: {field} not found: {resolved}")
    return resolved.relative_to(root)


def load_pair_inventory(path: Path, dataset_root: Path) -> tuple[SlidePair, ...]:
    """Parse and normalize a strict source/target pair inventory."""
    inventory_path = path if path.is_absolute() else dataset_root / path
    if not inventory_path.is_file():
        raise FileNotFoundError(f"Pair inventory not found: {inventory_path}")

    pairs: list[SlidePair] = []
    with inventory_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = tuple(reader.fieldnames or ())
        missing = [field for field in _REQUIRED_FIELDS if field not in fields]
        unexpected = [field for field in fields if field not in PAIR_INVENTORY_FIELDS]
        if missing:
            raise ValueError(f"Pair inventory is missing required columns: {missing}")
        if unexpected:
            raise ValueError(f"Pair inventory has unexpected columns: {unexpected}")

        for row_number, row in enumerate(reader, start=2):
            pair_id = (row.get("pair_id") or "").strip()
            if not PAIR_ID_PATTERN.fullmatch(pair_id):
                raise ValueError(f"Pair inventory row {row_number}: unsafe pair_id {pair_id!r}")
            source_path = _resolve_relative_path(
                row.get("source_path"),
                field="source_path",
                row=row_number,
                dataset_root=dataset_root,
                required=True,
            )
            target_path = _resolve_relative_path(
                row.get("target_path"),
                field="target_path",
                row=row_number,
                dataset_root=dataset_root,
                required=True,
            )
            assert source_path is not None and target_path is not None
            if source_path == target_path:
                raise ValueError(
                    f"Pair inventory row {row_number}: source_path and target_path must differ"
                )
            pairs.append(
                SlidePair(
                    pair_id=pair_id,
                    source_path=source_path,
                    target_path=target_path,
                    already_aligned=_parse_bool(row.get("already_aligned"), row=row_number),
                    shared_mask_path=_resolve_relative_path(
                        row.get("shared_mask_path"),
                        field="shared_mask_path",
                        row=row_number,
                        dataset_root=dataset_root,
                    ),
                    source_mask_path=_resolve_relative_path(
                        row.get("source_mask_path"),
                        field="source_mask_path",
                        row=row_number,
                        dataset_root=dataset_root,
                    ),
                    target_mask_path=_resolve_relative_path(
                        row.get("target_mask_path"),
                        field="target_mask_path",
                        row=row_number,
                        dataset_root=dataset_root,
                    ),
                    patient_id=_optional_text(row.get("patient_id")),
                    specimen_id=_optional_text(row.get("specimen_id")),
                    source_slide_id=_optional_text(row.get("source_slide_id")),
                    target_slide_id=_optional_text(row.get("target_slide_id")),
                )
            )

    ids = [pair.pair_id for pair in pairs]
    duplicates = sorted({pair_id for pair_id in ids if ids.count(pair_id) > 1})
    if duplicates:
        raise ValueError(f"Duplicate pair_id values: {duplicates}")
    if not pairs:
        raise ValueError("Pair inventory must contain at least one pair")
    return tuple(sorted(pairs, key=lambda pair: pair.pair_id))


def resolve_slide_pairs(config: PreprocessingConfig) -> tuple[SlidePair, ...]:
    """Resolve inventory or deprecated single-pair fields from PreprocessingConfig."""
    dataset_root = config.dataset_root
    inputs = config.inputs
    if inputs is not None and inputs.inventory is not None:
        return load_pair_inventory(inputs.inventory, dataset_root)

    warnings.warn(
        "preprocessing.source_name/target_name are deprecated; use inputs.inventory",
        FutureWarning,
        stacklevel=2,
    )
    source = _resolve_relative_path(
        config.source_name,
        field="source_name",
        row=1,
        dataset_root=dataset_root,
        required=True,
    )
    target = _resolve_relative_path(
        config.target_name,
        field="target_name",
        row=1,
        dataset_root=dataset_root,
        required=True,
    )
    assert source is not None and target is not None
    return (SlidePair("pair_0000", source, target),)


def load_pair_manifest(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != PAIR_MANIFEST_FIELDS:
            raise ValueError(f"Pair manifest must have exact columns: {list(PAIR_MANIFEST_FIELDS)}")
        rows: dict[str, dict[str, str]] = {}
        for row in reader:
            pair_id = row["pair_id"]
            if pair_id in rows:
                raise ValueError(f"Pair manifest duplicates pair_id {pair_id!r}")
            rows[pair_id] = row
    return rows
