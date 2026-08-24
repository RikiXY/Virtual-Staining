from __future__ import annotations

from pathlib import Path

from virtual_staining.data.slide_sets import SlideAsset, SlideSet
from virtual_staining.data.splitting import assign_group_splits


def _sets() -> tuple[SlideSet, ...]:
    return tuple(
        SlideSet(
            f"S{i}",
            (SlideAsset("LF", Path(f"lf{i}.png")),),
            SlideAsset("target", Path(f"target{i}.png")),
            "LF",
            patient_id=f"P{i // 2}",
            specimen_id=f"SP{i // 2}",
        )
        for i in range(6)
    )


def test_patient_and_specimen_units_keep_groups_together() -> None:
    sets = _sets()
    for unit in ("patient", "specimen"):
        assignments = assign_group_splits(sets, unit=unit, ratios=(0.5, 0.25, 0.25), seed=3)
        for left, right in zip(sets[::2], sets[1::2], strict=True):
            assert assignments[left.set_id] == assignments[right.set_id]
