"""Canonical long-form raw inventory; no roles, pairing, or inferred group identities."""

from __future__ import annotations

import csv
from dataclasses import dataclass

from virtual_staining.config.data import PreprocessingConfig
from virtual_staining.data.consumption import AssetRow, validate_groups, validate_locator
from virtual_staining.data.slide_sets import SET_ID_PATTERN
from virtual_staining.data.splitting import assign_identity_splits
from virtual_staining.split_contract import DatasetSplit
from virtual_staining.utils.image_io import VALID_IMAGE_EXTENSIONS, detect_openslide_format


@dataclass(frozen=True)
class RawDomainImage:
    domain: str
    path: str
    set_id: str = ""
    specimen_id: str = ""
    patient_id: str = ""
    mask_path: str = ""

    def row(self, *, split: str = "", locator: str | None = None) -> AssetRow:
        return AssetRow(
            root="dataset",
            locator=self.path if locator is None else locator,
            role="input",
            domain=self.domain,
            split=split,
            set_id=self.set_id,
            specimen_id=self.specimen_id,
            patient_id=self.patient_id,
        )


def load_unpaired_inventory(config: PreprocessingConfig) -> tuple[RawDomainImage, ...]:
    """Validate CSV and file membership without decoding images or hashing their bytes."""
    root = config.dataset_root.resolve()
    inventory = config.inputs.inventory
    path = inventory if inventory.is_absolute() else root / inventory
    if not path.resolve().is_relative_to(root):
        raise ValueError("Unpaired inventory must be contained in dataset_root")
    fields = {"domain", "path", "set_id", "specimen_id", "patient_id", "mask_path"}
    items = []
    physical: set[tuple[int, int]] = set()
    parents: dict[tuple[str, str, str], str] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        if (
            len(columns) != len(set(columns))
            or not {"domain", "path"} <= set(columns)
            or set(columns) - fields
        ):
            raise ValueError(
                "Unpaired inventory requires domain,path and only optional "
                "set_id,specimen_id,patient_id,mask_path columns"
            )
        for number, raw in enumerate(reader, 2):
            if None in raw or any(value is None for value in raw.values()):
                raise ValueError(f"Inventory row {number}: malformed CSV row")
            if any(value != value.strip() for value in raw.values()):
                raise ValueError(f"Inventory row {number}: surrounding whitespace is not allowed")
            item = RawDomainImage(**raw)
            if item.domain not in config.inputs.domains:
                raise ValueError(f"Inventory row {number}: unknown domain {item.domain!r}")
            for field in ("set_id", "specimen_id", "patient_id"):
                value = getattr(item, field)
                if value and not SET_ID_PATTERN.fullmatch(value):
                    raise ValueError(f"Inventory row {number}: invalid {field} {value!r}")
            for child, parent in (
                ("set_id", "specimen_id"),
                ("set_id", "patient_id"),
                ("specimen_id", "patient_id"),
            ):
                identity, owner = getattr(item, child), getattr(item, parent)
                if identity:
                    key = child, identity, parent
                    if key in parents and parents[key] != owner:
                        raise ValueError(
                            f"Inventory row {number}: conflicting or incomplete {parent} "
                            f"for {child} {identity!r}"
                        )
                    parents[key] = owner
            for locator in (item.path, *((item.mask_path,) if item.mask_path else ())):
                validate_locator(locator)
                candidate = root / locator
                resolved = candidate.resolve(strict=True)
                if not resolved.is_relative_to(root):
                    raise ValueError(f"Inventory row {number}: path escapes dataset_root")
                if not resolved.is_file() or (
                    candidate.suffix.lower() not in VALID_IMAGE_EXTENSIONS
                    and detect_openslide_format(candidate) is None
                ):
                    raise ValueError(
                        f"Inventory row {number}: not a supported image file: {locator}"
                    )
                stat = resolved.stat()
                key = stat.st_dev, stat.st_ino
                if key in physical:
                    raise ValueError(
                        f"Inventory row {number}: duplicated/reused physical file: {locator}"
                    )
                physical.add(key)
            items.append(item)
    if {item.domain for item in items} != set(config.inputs.domains):
        raise ValueError(
            "Unpaired inventory requires nonempty collections for exactly both configured domains"
        )
    return tuple(
        sorted(items, key=lambda item: (config.inputs.domains.index(item.domain), item.path))
    )


def unpaired_assignments(
    config: PreprocessingConfig,
    items: tuple[RawDomainImage, ...],
    group_validation: str,
) -> dict[str, DatasetSplit]:
    split = config.split
    if split.unit == "patch":
        if split.assignment_file is not None:
            raise ValueError("split.assignment_file is not supported for split.unit='patch'")
        return {}
    groups = {getattr(item, f"{split.unit}_id") for item in items}
    if "" in groups:
        raise ValueError(
            f"split.unit={split.unit!r} requires explicit {split.unit}_id for every image"
        )
    if group_validation in {"patient", "specimen", "set"} and any(
        not getattr(item, f"{group_validation}_id") for item in items
    ):
        raise ValueError(
            f"data.group_validation={group_validation!r} requires explicit "
            "identifiers for every image"
        )
    # Explicit shared identities connect split units; these are not inferred image pairs.
    components = {group: group for group in groups}

    def representative(group: str) -> str:
        while components[group] != group:
            components[group] = components[components[group]]
            group = components[group]
        return group

    if split.assignment_file is None:
        owners: dict[tuple[str, str], str] = {}
        for item in items:
            group = getattr(item, f"{split.unit}_id")
            for field in ("set_id", "specimen_id", "patient_id"):
                value = getattr(item, field)
                if not value:
                    continue
                key = field, value
                previous = owners.setdefault(key, group)
                old, new = representative(group), representative(previous)
                components[max(old, new)] = min(old, new)
    components = {group: representative(group) for group in groups}
    assigned = assign_identity_splits(
        set(components.values()),
        unit=split.unit,
        ratios=(split.train, split.val, split.test),
        seed=split.seed,
        assignment_file=split.assignment_file,
        dataset_root=config.dataset_root,
    )
    assignments: dict[str, DatasetSplit] = {
        group: assigned[component] for group, component in components.items()
    }
    validate_groups(
        [item.row(split=assignments[getattr(item, f"{split.unit}_id")]) for item in items],
        group_validation,
    )
    return assignments
