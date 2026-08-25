from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from virtual_staining.config.data import PreprocessingConfig

MODALITY_NAME_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_-]*\Z")
SET_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")


@dataclass(frozen=True)
class SlideAsset:
    modality: str
    path: Path
    already_aligned: bool | None = None
    mask_path: Path | None = None
    slide_id: str | None = None


@dataclass(frozen=True)
class SlideSet:
    set_id: str
    inputs: tuple[SlideAsset, ...]
    target: SlideAsset
    reference_modality: str
    patient_id: str | None = None
    specimen_id: str | None = None


def _optional_text(value: str | None) -> str | None:
    text = (value or "").strip()
    return text or None


def _parse_bool(value: str | None, *, row: int, field: str) -> bool | None:
    text = (value or "").strip().lower()
    if not text:
        return None
    if text == "true":
        return True
    if text == "false":
        return False
    raise ValueError(f"Inventory row {row}: {field} must be true, false, or blank")


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
            raise ValueError(f"Inventory row {row}: {field} must not be empty")
        return None
    path = Path(text)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Inventory row {row}: {field} must be relative and non-traversing")
    root = dataset_root.resolve()
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"Inventory row {row}: {field} resolves outside dataset_root")
    if not resolved.is_file():
        raise FileNotFoundError(f"Inventory row {row}: {field} not found: {resolved}")
    return resolved.relative_to(root)


def load_slide_set_inventory(
    path: Path,
    dataset_root: Path,
    *,
    modalities: tuple[str, ...],
    reference_modality: str,
    target_modality: str,
) -> tuple[SlideSet, ...]:
    inventory_path = path if path.is_absolute() else dataset_root / path
    if not inventory_path.is_file():
        raise FileNotFoundError(f"Slide-set inventory not found: {inventory_path}")
    required = [
        "set_id",
        *(f"input__{m}_path" for m in modalities),
        *(f"input__{m}_aligned" for m in modalities),
        "target_path",
        "target_aligned",
    ]
    allowed = set(required) | {"patient_id", "specimen_id", "target_mask", "target_slide_id"}
    for modality in modalities:
        allowed.update({f"input__{modality}_mask", f"input__{modality}_slide_id"})

    sets: list[SlideSet] = []
    with inventory_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = tuple(reader.fieldnames or ())
        missing = [field for field in required if field not in fields]
        unexpected = [field for field in fields if field not in allowed]
        if missing:
            raise ValueError(f"Slide-set inventory is missing required columns: {missing}")
        if unexpected:
            raise ValueError(f"Slide-set inventory has unexpected columns: {unexpected}")
        for row_number, row in enumerate(reader, start=2):
            set_id = (row.get("set_id") or "").strip()
            if not SET_ID_PATTERN.fullmatch(set_id):
                raise ValueError(f"Inventory row {row_number}: unsafe set_id {set_id!r}")
            inputs: list[SlideAsset] = []
            for modality in modalities:
                path_value = _resolve_relative_path(
                    row.get(f"input__{modality}_path"),
                    field=f"input__{modality}_path",
                    row=row_number,
                    dataset_root=dataset_root,
                    required=True,
                )
                aligned = _parse_bool(
                    row.get(f"input__{modality}_aligned"),
                    row=row_number,
                    field=f"input__{modality}_aligned",
                )
                if modality == reference_modality and aligned is False:
                    raise ValueError(
                        f"Inventory row {row_number}: reference modality must be aligned"
                    )
                assert path_value is not None
                mask_path = _resolve_relative_path(
                    row.get(f"input__{modality}_mask"),
                    field=f"input__{modality}_mask",
                    row=row_number,
                    dataset_root=dataset_root,
                )
                inputs.append(
                    SlideAsset(
                        modality=modality,
                        path=path_value,
                        already_aligned=aligned,
                        mask_path=mask_path,
                        slide_id=_optional_text(row.get(f"input__{modality}_slide_id")),
                    )
                )
            target_path = _resolve_relative_path(
                row.get("target_path"),
                field="target_path",
                row=row_number,
                dataset_root=dataset_root,
                required=True,
            )
            target_aligned = _parse_bool(
                row.get("target_aligned"), row=row_number, field="target_aligned"
            )
            target_mask = _resolve_relative_path(
                row.get("target_mask"),
                field="target_mask",
                row=row_number,
                dataset_root=dataset_root,
            )
            assert target_path is not None
            if target_path in {asset.path for asset in inputs}:
                raise ValueError(
                    f"Inventory row {row_number}: target_path must differ from every input"
                )
            sets.append(
                SlideSet(
                    set_id=set_id,
                    inputs=tuple(inputs),
                    target=SlideAsset(
                        modality=target_modality,
                        path=target_path,
                        already_aligned=target_aligned,
                        mask_path=target_mask,
                        slide_id=_optional_text(row.get("target_slide_id")),
                    ),
                    reference_modality=reference_modality,
                    patient_id=_optional_text(row.get("patient_id")),
                    specimen_id=_optional_text(row.get("specimen_id")),
                )
            )
    ids = [item.set_id for item in sets]
    duplicates = sorted({set_id for set_id in ids if ids.count(set_id) > 1})
    if duplicates:
        raise ValueError(f"Duplicate set_id values: {duplicates}")
    if not sets:
        raise ValueError("Slide-set inventory must contain at least one set")
    return tuple(sorted(sets, key=lambda item: item.set_id))


def resolve_slide_sets(config: PreprocessingConfig) -> tuple[SlideSet, ...]:
    return load_slide_set_inventory(
        config.inputs.inventory,
        config.dataset_root,
        modalities=config.inputs.modalities,
        reference_modality=config.inputs.reference,
        target_modality=config.inputs.target_modality,
    )
