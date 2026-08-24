from __future__ import annotations

from pathlib import Path

import pytest

from virtual_staining.data.slide_sets import SlideAsset, SlideSet, load_slide_set_inventory
from virtual_staining.data.splitting import assign_group_splits, group_id_for_set


def _inventory(root: Path) -> Path:
    for name in ("lf1.png", "af1.png", "target1.png", "lf2.png", "af2.png", "target2.png"):
        (root / name).write_bytes(b"image")
    path = root / "slides.csv"
    path.write_text(
        "set_id,input__LF_path,input__LF_aligned,input__AF_path,input__AF_aligned,target_path,target_aligned,patient_id,specimen_id\n"
        "S2,lf2.png,true,af2.png,false,target2.png,false,P2,SP2\n"
        "S1,lf1.png,true,af1.png,,target1.png,true,P1,SP1\n",
        encoding="utf-8",
    )
    return path


def test_wide_inventory_is_order_independent_and_named(tmp_path: Path) -> None:
    path = _inventory(tmp_path)
    first = load_slide_set_inventory(
        path, tmp_path, modalities=("LF", "AF"), reference_modality="LF", target_modality="target"
    )
    path.write_text(
        path.read_text(encoding="utf-8").replace("S2,", "S1,", 1).replace("S1,", "S2,", 1),
        encoding="utf-8",
    )
    # The parser sorts by set_id, regardless of CSV row order.
    assert [item.set_id for item in first] == ["S1", "S2"]
    assert first[0].inputs[0].modality == "LF"
    assert first[0].inputs[1].already_aligned is None
    assert first[1].target.already_aligned is False


def test_inventory_rejects_unsafe_paths(tmp_path: Path) -> None:
    path = tmp_path / "slides.csv"
    path.write_text(
        "set_id,input__LF_path,input__LF_aligned,target_path,target_aligned\nS1,../x.png,true,target.png,true\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="relative and non-traversing"):
        load_slide_set_inventory(
            path, tmp_path, modalities=("LF",), reference_modality="LF", target_modality="target"
        )


def test_group_split_keeps_patient_and_specimen_sets_together() -> None:
    sets = tuple(
        SlideSet(
            f"S{i}",
            (SlideAsset("LF", Path(f"s{i}.png")),),
            SlideAsset("target", Path(f"t{i}.png")),
            "LF",
            patient_id=f"P{i // 2}",
            specimen_id=f"SP{i // 2}",
        )
        for i in range(4)
    )
    assignments = assign_group_splits(sets, unit="patient", ratios=(0.5, 0.5, 0.0), seed=0)
    assert assignments["S0"] == assignments["S1"]
    assert assignments["S2"] == assignments["S3"]
    assert group_id_for_set(sets[0], "patient") == "P0"


def test_group_split_rejects_pair_unit() -> None:
    sets = (
        SlideSet(
            "S1", (SlideAsset("LF", Path("s.png")),), SlideAsset("target", Path("t.png")), "LF"
        ),
    )
    with pytest.raises(ValueError, match="unit"):
        assign_group_splits(sets, unit="pair", ratios=(0.8, 0.1, 0.1), seed=0)
