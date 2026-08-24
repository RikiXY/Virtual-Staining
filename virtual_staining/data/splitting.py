from __future__ import annotations

import csv
import hashlib
import math
from pathlib import Path

from virtual_staining.data.manifest import Split
from virtual_staining.data.slide_sets import SlideSet

SPLITS: tuple[Split, Split, Split] = ("train", "val", "test")


def group_id_for_set(slide_set: SlideSet, unit: str) -> str:
    if unit == "set":
        return slide_set.set_id
    value = getattr(slide_set, f"{unit}_id", None)
    if not value:
        raise ValueError(f"split.unit={unit!r} requires {unit}_id for set {slide_set.set_id!r}")
    return str(value)


def _group_counts(count: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    positive = [index for index, ratio in enumerate(ratios) if ratio > 0]
    if count < len(positive):
        raise ValueError(
            f"Cannot place {count} independent groups into {len(positive)} nonzero splits"
        )
    counts = [1 if index in positive else 0 for index in range(3)]
    remaining = count - len(positive)
    if remaining == 0:
        return tuple(counts)  # type: ignore[return-value]
    weights = [max(count * ratio - counts[index], 0.0) for index, ratio in enumerate(ratios)]
    total = sum(weights) or sum(ratios)
    quotas = [remaining * weight / total for weight in weights]
    additions = [math.floor(quota) for quota in quotas]
    for index in sorted(range(3), key=lambda item: (-(quotas[item] - additions[item]), item))[
        : remaining - sum(additions)
    ]:
        additions[index] += 1
    return tuple(counts[index] + additions[index] for index in range(3))  # type: ignore[return-value]


def _load_frozen_assignment(path: Path, *, unit: str, groups: set[str]) -> dict[str, Split]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        expected = ["group_id", "unit", "split"]
        if reader.fieldnames != expected:
            raise ValueError(f"Frozen split assignment must have exact columns: {expected}")
        assignments: dict[str, Split] = {}
        for row_number, row in enumerate(reader, start=2):
            group_id = row["group_id"].strip()
            if group_id in assignments:
                raise ValueError(f"Frozen split assignment duplicates group {group_id!r}")
            if row["unit"] != unit:
                raise ValueError(
                    f"Frozen split assignment row {row_number} uses unit "
                    f"{row['unit']!r}, expected {unit!r}"
                )
            if row["split"] not in SPLITS:
                raise ValueError(f"Frozen split assignment row {row_number} has invalid split")
            assignments[group_id] = row["split"]  # type: ignore[assignment]
    if set(assignments) != groups:
        missing = sorted(groups - set(assignments))
        unknown = sorted(set(assignments) - groups)
        raise ValueError(
            "Frozen split assignment group set does not match inventory: "
            f"missing={missing}, unknown={unknown}"
        )
    return assignments


def assign_group_splits(
    slide_sets: tuple[SlideSet, ...],
    *,
    unit: str,
    ratios: tuple[float, float, float],
    seed: int,
    assignment_file: Path | None = None,
    dataset_root: Path | None = None,
) -> dict[str, Split]:
    if unit == "patch":
        if assignment_file is not None:
            raise ValueError("split.assignment_file is not supported for split.unit='patch'")
        return {}
    groups_by_set = {item.set_id: group_id_for_set(item, unit) for item in slide_sets}
    groups = set(groups_by_set.values())
    if assignment_file is not None:
        path = (
            assignment_file
            if assignment_file.is_absolute()
            else (dataset_root or Path()) / assignment_file
        )
        group_assignments = _load_frozen_assignment(path, unit=unit, groups=groups)
    else:
        ordered = sorted(
            groups, key=lambda group: (hashlib.sha256(f"{seed}:{group}".encode()).digest(), group)
        )
        counts = _group_counts(len(ordered), ratios)
        group_assignments: dict[str, Split] = {}
        offset = 0
        for split, count in zip(SPLITS, counts, strict=True):
            group_assignments.update({group: split for group in ordered[offset : offset + count]})
            offset += count
    return {set_id: group_assignments[group] for set_id, group in groups_by_set.items()}


def write_split_assignment(path: Path, *, unit: str, assignments: dict[str, Split]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["group_id", "unit", "split"])
        writer.writeheader()
        for group_id, split in sorted(assignments.items()):
            writer.writerow({"group_id": group_id, "unit": unit, "split": split})
