from __future__ import annotations

import csv
import hashlib
import math
from pathlib import Path

from virtual_staining.data.manifest import Split
from virtual_staining.data.pairs import SlidePair

SPLITS: tuple[Split, Split, Split] = ("train", "val", "test")


def group_id_for_pair(pair: SlidePair, unit: str) -> str:
    if unit == "pair":
        return pair.pair_id
    value = getattr(pair, f"{unit}_id", None)
    if not value:
        raise ValueError(f"split.unit={unit!r} requires {unit}_id for pair {pair.pair_id!r}")
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

    residual_weights = [
        max(count * ratio - counts[index], 0.0) for index, ratio in enumerate(ratios)
    ]
    total = sum(residual_weights)
    if total == 0:
        residual_weights = list(ratios)
        total = sum(residual_weights)
    quotas = [remaining * weight / total for weight in residual_weights]
    additions = [math.floor(quota) for quota in quotas]
    for index in sorted(range(3), key=lambda item: (-(quotas[item] - additions[item]), item))[
        : remaining - sum(additions)
    ]:
        additions[index] += 1
    return tuple(counts[index] + additions[index] for index in range(3))  # type: ignore[return-value]


def _load_frozen_assignment(
    path: Path,
    *,
    unit: str,
    groups: set[str],
) -> dict[str, Split]:
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
        raise ValueError(
            "Frozen split assignment group set does not match inventory: "
            f"missing={sorted(groups - set(assignments))}, "
            f"unknown={sorted(set(assignments) - groups)}"
        )
    return assignments


def assign_group_splits(
    pairs: tuple[SlidePair, ...],
    *,
    unit: str,
    ratios: tuple[float, float, float],
    seed: int,
    assignment_file: Path | None = None,
    dataset_root: Path | None = None,
) -> dict[str, Split]:
    """Return pair_id -> split for a deterministic independent-group partition."""
    if unit == "patch":
        if assignment_file is not None:
            raise ValueError("split.assignment_file is not supported for split.unit='patch'")
        return {}
    groups_by_pair = {pair.pair_id: group_id_for_pair(pair, unit) for pair in pairs}
    groups = set(groups_by_pair.values())
    if assignment_file is not None:
        path = assignment_file
        if not path.is_absolute():
            if dataset_root is None:
                raise ValueError("dataset_root is required for a relative assignment_file")
            path = dataset_root / path
        group_assignments = _load_frozen_assignment(path, unit=unit, groups=groups)
    else:
        ordered = sorted(
            groups,
            key=lambda group: (hashlib.sha256(f"{seed}:{group}".encode()).digest(), group),
        )
        counts = _group_counts(len(ordered), ratios)
        group_assignments = {}
        offset = 0
        for split, count in zip(SPLITS, counts, strict=True):
            group_assignments.update({group: split for group in ordered[offset : offset + count]})
            offset += count
    return {pair_id: group_assignments[group] for pair_id, group in groups_by_pair.items()}


def write_split_assignment(
    path: Path,
    *,
    unit: str,
    assignments: dict[str, Split],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["group_id", "unit", "split"])
        writer.writeheader()
        for group_id, split in sorted(assignments.items()):
            writer.writerow({"group_id": group_id, "unit": unit, "split": split})
