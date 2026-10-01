from __future__ import annotations

import csv
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from virtual_staining.config.validation import MODALITY_NAME_PATTERN, check_modality_names
from virtual_staining.data.alignment import AlignmentError, SpatialEvidence

if TYPE_CHECKING:
    from virtual_staining.config.data import PreprocessingConfig

__all__ = [
    "ASSET_FIELDS",
    "MODALITY_NAME_PATTERN",
    "SET_ID_PATTERN",
    "SlideAsset",
    "SlideSet",
    "RegistrationEvidence",
    "asset_column",
    "load_slide_set_inventory",
    "resolve_slide_sets",
    "resolve_registration_evidence",
]

SET_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
AssetRole = Literal["input", "target"]
#: Per-asset inventory fields: ``<role>__<name>_<field>``; path/aligned columns are required.
ASSET_FIELDS = ("path", "aligned", "mask", "slide_id")
_SUPERSEDED_COLUMNS = frozenset({"target_path", "target_aligned", "target_mask", "target_slide_id"})


def asset_column(role: str, name: str, field: str) -> str:
    """The canonical inventory column of one asset field, e.g. ``target__HE_path``."""
    return f"{role}__{name}_{field}"


@dataclass(frozen=True)
class SlideAsset:
    modality: str
    path: Path
    already_aligned: bool | None = None
    mask_path: Path | None = None
    slide_id: str | None = None


@dataclass(frozen=True)
class SlideSet:
    """One raw set: ordered named inputs and every configured named target, all required."""

    set_id: str
    inputs: tuple[SlideAsset, ...]
    targets: tuple[SlideAsset, ...]
    reference_modality: str
    patient_id: str | None = None
    specimen_id: str | None = None

    @property
    def assets(self) -> tuple[SlideAsset, ...]:
        return (*self.inputs, *self.targets)


RegistrationEvidence = Mapping[tuple[str, str], tuple[SpatialEvidence, ...]]


def resolve_registration_evidence(
    slide_sets: tuple[SlideSet, ...], evidence: RegistrationEvidence | None
) -> dict[tuple[str, str], tuple[SpatialEvidence, ...]]:
    """Bind supplied maps to (set_id, modality); actual geometry is checked by AlignmentImage."""
    assets = {(item.set_id, asset.modality) for item in slide_sets for asset in item.assets}
    resolved = {}
    for key, maps in (evidence or {}).items():
        if key not in assets:
            raise AlignmentError("Registration evidence names an unknown set/asset")
        kinds = set()
        for item in maps:
            if not isinstance(item, SpatialEvidence) or item.asset.name != key[1]:
                raise AlignmentError("Registration evidence asset mismatch")
            if item.kind in kinds:
                raise AlignmentError("Duplicate registration evidence kind for an asset")
            kinds.add(item.kind)
        if maps:
            resolved[key] = tuple(sorted(maps, key=lambda item: item.kind))
    return resolved


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


def _read_asset(
    row: dict[str, str],
    role: AssetRole,
    modality: str,
    *,
    row_number: int,
    dataset_root: Path,
    reference_modality: str,
) -> SlideAsset:
    def column(field: str) -> str:
        return asset_column(role, modality, field)

    path_value = _resolve_relative_path(
        row.get(column("path")),
        field=column("path"),
        row=row_number,
        dataset_root=dataset_root,
        required=True,
    )
    aligned = _parse_bool(row.get(column("aligned")), row=row_number, field=column("aligned"))
    if role == "input" and modality == reference_modality and aligned is False:
        raise ValueError(f"Inventory row {row_number}: reference modality must be aligned")
    assert path_value is not None
    return SlideAsset(
        modality=modality,
        path=path_value,
        already_aligned=aligned,
        mask_path=_resolve_relative_path(
            row.get(column("mask")),
            field=column("mask"),
            row=row_number,
            dataset_root=dataset_root,
        ),
        slide_id=_optional_text(row.get(column("slide_id"))),
    )


def _reject_file_reuse(assets: tuple[SlideAsset, ...], dataset_root: Path, row: int) -> None:
    """No two assets of one set may be the same physical file (paths, symlinks, hard links)."""
    seen: dict[tuple[int, int], str] = {}
    for asset in assets:
        stat = os.stat(dataset_root / asset.path)
        identity = (stat.st_dev, stat.st_ino)
        if identity in seen:
            raise ValueError(
                f"Inventory row {row}: {asset.modality} and {seen[identity]} are the same "
                f"physical file {asset.path.as_posix()}"
            )
        seen[identity] = asset.modality


def load_slide_set_inventory(
    path: Path,
    dataset_root: Path,
    *,
    modalities: tuple[str, ...],
    reference_modality: str,
    target_modalities: tuple[str, ...],
) -> tuple[SlideSet, ...]:
    """Read and validate the canonical wide slide-set inventory, sorted by ``set_id``.

    Every input and every configured target is required for every set; each has
    ``<role>__<name>_path``/``_aligned`` and optional ``_mask``/``_slide_id`` columns. A
    relative ``path`` is relative to ``dataset_root``, as are the paths inside it.
    """
    check_modality_names(tuple(modalities), "inventory input modalities")
    check_modality_names(tuple(target_modalities), "inventory target modalities")
    shared = sorted(set(modalities) & set(target_modalities))
    if shared:
        raise ValueError(f"Inventory inputs and targets must be disjoint; both name {shared}")
    if reference_modality not in modalities:
        raise ValueError(f"Reference modality {reference_modality!r} is not an input")
    inventory_path = path if path.is_absolute() else dataset_root / path
    if not inventory_path.is_file():
        raise FileNotFoundError(f"Slide-set inventory not found: {inventory_path}")
    assets: tuple[tuple[AssetRole, str], ...] = (
        *(("input", name) for name in modalities),
        *(("target", name) for name in target_modalities),
    )
    required = [
        "set_id",
        *(
            asset_column(role, name, field)
            for role, name in assets
            for field in ("path", "aligned")
        ),
    ]
    allowed = set(required) | {"patient_id", "specimen_id"}
    allowed.update(
        asset_column(role, name, field) for role, name in assets for field in ("mask", "slide_id")
    )

    sets: list[SlideSet] = []
    with inventory_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = tuple(reader.fieldnames or ())
        duplicated = sorted({field for field in fields if fields.count(field) > 1})
        if duplicated:
            raise ValueError(f"Slide-set inventory has duplicate columns: {duplicated}")
        superseded = sorted(_SUPERSEDED_COLUMNS & set(fields))
        if superseded:
            raise ValueError(
                f"Slide-set inventory columns {superseded} are not part of the current schema; "
                "targets are named: target__<name>_path, target__<name>_aligned, "
                "target__<name>_mask, target__<name>_slide_id"
            )
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
            read = {
                "row_number": row_number,
                "dataset_root": dataset_root,
                "reference_modality": reference_modality,
            }
            slide_set = SlideSet(
                set_id=set_id,
                inputs=tuple(_read_asset(row, "input", name, **read) for name in modalities),
                targets=tuple(
                    _read_asset(row, "target", name, **read) for name in target_modalities
                ),
                reference_modality=reference_modality,
                patient_id=_optional_text(row.get("patient_id")),
                specimen_id=_optional_text(row.get("specimen_id")),
            )
            _reject_file_reuse(slide_set.assets, dataset_root, row_number)
            sets.append(slide_set)
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
        target_modalities=config.inputs.target_modalities,
    )
