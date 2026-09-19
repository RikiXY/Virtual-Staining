from __future__ import annotations

from pathlib import Path

import pytest

from virtual_staining.config.data import InputConfig, PreprocessingConfig
from virtual_staining.data.slide_sets import SlideSet, resolve_slide_sets


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


def _resolve_inventory(root: Path, path: Path, modalities: tuple[str, ...]) -> tuple[SlideSet, ...]:
    config = PreprocessingConfig(
        dataset_root=root,
        inputs=InputConfig(path, modalities, modalities[0], "target"),
    )
    return resolve_slide_sets(config)


def test_wide_inventory_is_order_independent_and_named(tmp_path: Path) -> None:
    path = _inventory(tmp_path)
    first = _resolve_inventory(tmp_path, path, ("LF", "AF"))
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
        _resolve_inventory(tmp_path, path, ("LF",))
