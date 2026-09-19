from __future__ import annotations

from pathlib import Path

from virtual_staining.data.slide_sets import SlideAsset, SlideSet
from virtual_staining.data.splitting import assign_group_splits, assign_split_by_hash


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


def test_assign_split_by_hash_is_deterministic() -> None:
    first = assign_split_by_hash(seed=17, sample_id="sample", ratios=(0.7, 0.2, 0.1))
    second = assign_split_by_hash(seed=17, sample_id="sample", ratios=(0.7, 0.2, 0.1))

    assert first == second


def test_assign_split_by_hash_validates_ratios() -> None:
    import pytest

    with pytest.raises(ValueError, match="Expected 3 split ratios"):
        assign_split_by_hash(seed=1, sample_id="sample", ratios=(0.5, 0.5))
    with pytest.raises(ValueError, match="non-negative"):
        assign_split_by_hash(seed=1, sample_id="sample", ratios=(0.8, 0.3, -0.1))
    with pytest.raises(ValueError, match="sum to 1.0"):
        assign_split_by_hash(seed=1, sample_id="sample", ratios=(0.8, 0.1, 0.05))
