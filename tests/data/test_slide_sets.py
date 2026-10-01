from __future__ import annotations

import os
from pathlib import Path

import pytest

from virtual_staining.config.data import InputConfig, PreprocessingConfig
from virtual_staining.data.slide_sets import (
    SlideSet,
    load_slide_set_inventory,
    resolve_slide_sets,
)


def _write(root: Path, text: str) -> Path:
    path = root / "slides.csv"
    path.write_text(text, encoding="utf-8")
    return path


def _files(root: Path, *names: str) -> None:
    for name in names:
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_bytes(f"image {name}".encode())


def _inventory(root: Path) -> Path:
    _files(root, "lf1.png", "af1.png", "he1.png", "lf2.png", "af2.png", "he2.png")
    return _write(
        root,
        "set_id,input__LF_path,input__LF_aligned,input__AF_path,input__AF_aligned,"
        "target__HE_path,target__HE_aligned,patient_id,specimen_id\n"
        "S2,lf2.png,true,af2.png,false,he2.png,false,P2,SP2\n"
        "S1,lf1.png,true,af1.png,,he1.png,true,P1,SP1\n",
    )


def _load(root: Path, path: Path, targets: tuple[str, ...] = ("HE",)) -> tuple[SlideSet, ...]:
    return load_slide_set_inventory(
        path, root, modalities=("LF", "AF"), reference_modality="LF", target_modalities=targets
    )


def _resolve_inventory(root: Path, path: Path, modalities: tuple[str, ...]) -> tuple[SlideSet, ...]:
    config = PreprocessingConfig(
        dataset_root=root,
        inputs=InputConfig(path, modalities, modalities[0], ("HE",)),
    )
    return resolve_slide_sets(config)


def test_wide_inventory_is_order_independent_and_named(tmp_path: Path) -> None:
    path = _inventory(tmp_path)
    first = _resolve_inventory(tmp_path, path, ("LF", "AF"))

    # The parser sorts by set_id, regardless of CSV row order.
    assert [item.set_id for item in first] == ["S1", "S2"]
    assert first[0].inputs[0].modality == "LF"
    assert first[0].inputs[1].already_aligned is None
    assert [asset.modality for asset in first[1].targets] == ["HE"]
    assert first[1].targets[0].already_aligned is False
    assert first[0].patient_id == "P1"


def test_two_targets_keep_order_masks_and_metadata(tmp_path: Path) -> None:
    _files(tmp_path, "lf.png", "af.png", "he.png", "pas.png", "m/he.png", "m/pas.png")
    path = _write(
        tmp_path,
        "set_id,input__LF_path,input__LF_aligned,input__AF_path,input__AF_aligned,"
        "target__PAS_path,target__PAS_aligned,target__HE_path,target__HE_aligned,"
        "target__HE_mask,target__PAS_mask,target__HE_slide_id,target__PAS_slide_id\n"
        "S1,lf.png,true,af.png,,pas.png,false,he.png,true,m/he.png,m/pas.png,slide-he,slide-pas\n",
    )

    (slide_set,) = _load(tmp_path, path, ("PAS", "HE"))

    assert [asset.modality for asset in slide_set.targets] == ["PAS", "HE"]
    pas, he = slide_set.targets
    assert (pas.path, pas.mask_path, pas.slide_id, pas.already_aligned) == (
        Path("pas.png"),
        Path("m/pas.png"),
        "slide-pas",
        False,
    )
    assert (he.path, he.mask_path, he.slide_id, he.already_aligned) == (
        Path("he.png"),
        Path("m/he.png"),
        "slide-he",
        True,
    )
    assert slide_set.assets == (*slide_set.inputs, *slide_set.targets)


def test_one_target_inventory_uses_the_same_plural_columns(tmp_path: Path) -> None:
    (slide_set, _) = _load(tmp_path, _inventory(tmp_path))

    assert len(slide_set.targets) == 1
    assert not hasattr(slide_set, "target")


def test_every_configured_target_is_required(tmp_path: Path) -> None:
    path = _inventory(tmp_path)
    with pytest.raises(ValueError, match=r"missing required columns: \['target__PAS_path'"):
        _load(tmp_path, path, ("HE", "PAS"))

    _files(tmp_path, "pas.png")
    path = _write(
        tmp_path,
        "set_id,input__LF_path,input__LF_aligned,input__AF_path,input__AF_aligned,"
        "target__HE_path,target__HE_aligned,target__PAS_path,target__PAS_aligned\n"
        "S1,lf1.png,true,af1.png,,he1.png,,,\n",
    )
    with pytest.raises(ValueError, match="target__PAS_path must not be empty"):
        _load(tmp_path, path, ("HE", "PAS"))


def test_singular_target_header_is_rejected_after_the_cutover(tmp_path: Path) -> None:
    _files(tmp_path, "lf.png", "af.png", "t.png")
    path = _write(
        tmp_path,
        "set_id,input__LF_path,input__LF_aligned,input__AF_path,input__AF_aligned,"
        "target_path,target_aligned\nS1,lf.png,true,af.png,,t.png,\n",
    )
    with pytest.raises(ValueError, match="not part of the current schema"):
        _load(tmp_path, path)


@pytest.mark.parametrize(
    ("row", "match"),
    [
        ("S1,lf.png,true,af.png,,lf.png,,pas.png,", "LF are the same physical file"),
        ("S1,lf.png,true,af.png,,he.png,,he.png,", "HE are the same physical file"),
        ("S1,lf.png,true,af.png,,he.png,,link.png,", "HE are the same physical file"),
        ("S1,lf.png,true,af.png,,he.png,,../x.png,", "relative and non-traversing"),
        ("S1,lf.png,true,af.png,,he.png,,escape.png,", "resolves outside dataset_root"),
        ("../S1,lf.png,true,af.png,,he.png,,pas.png,", "unsafe set_id"),
        ("S1,lf.png,false,af.png,,he.png,,pas.png,", "reference modality must be aligned"),
        ("S1,lf.png,true,af.png,,he.png,maybe,pas.png,", "true, false, or blank"),
    ],
)
def test_inventory_rows_are_validated(tmp_path: Path, row: str, match: str) -> None:
    root = tmp_path / "root"
    _files(root, "lf.png", "af.png", "he.png", "pas.png")
    os.symlink(root / "he.png", root / "link.png")
    (tmp_path / "outside.png").write_bytes(b"x")
    os.symlink(tmp_path / "outside.png", root / "escape.png")
    path = _write(
        root,
        "set_id,input__LF_path,input__LF_aligned,input__AF_path,input__AF_aligned,"
        f"target__HE_path,target__HE_aligned,target__PAS_path,target__PAS_aligned\n{row}\n",
    )
    with pytest.raises(ValueError, match=match):
        _load(root, path, ("HE", "PAS"))


@pytest.mark.parametrize(
    ("targets", "match"),
    [(("HE", "HE"), "duplicate names"), (("LF",), "disjoint"), (("H&E",), "invalid identifiers")],
)
def test_target_names_are_validated(tmp_path: Path, targets: tuple[str, ...], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _load(tmp_path, _inventory(tmp_path), targets)


def test_inventory_rejects_duplicate_columns(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "set_id,input__LF_path,input__LF_aligned,input__AF_path,input__AF_aligned,"
        "target__HE_path,target__HE_aligned,target__HE_path\n",
    )
    with pytest.raises(ValueError, match="duplicate columns"):
        _load(tmp_path, path)


def test_inventory_rejects_unsafe_paths(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "set_id,input__LF_path,input__LF_aligned,target__HE_path,target__HE_aligned\n"
        "S1,../x.png,true,target.png,true\n",
    )
    with pytest.raises(ValueError, match="relative and non-traversing"):
        _resolve_inventory(tmp_path, path, ("LF",))


def test_public_loader_is_the_resolver(tmp_path: Path) -> None:
    path = _inventory(tmp_path)

    assert _load(tmp_path, Path("slides.csv")) == _resolve_inventory(tmp_path, path, ("LF", "AF"))
