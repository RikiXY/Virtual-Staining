from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from tests.config_helpers import cyclegan_config_data
from virtual_staining.applications import inventory_authoring
from virtual_staining.applications.config_authoring import inspect_run_mapping, preflight
from virtual_staining.applications.inventory_authoring import (
    InventoryPreview,
    InventoryRequest,
    preview_inventory,
    render_inventory_csv,
    write_inventory,
)
from virtual_staining.data.slide_sets import load_slide_set_inventory


def _touch(root: Path, *paths: str) -> None:
    for path in paths:
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_bytes(b"placeholder, not an image")


def _files(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _request(root: Path, **overrides: Any) -> InventoryRequest:
    values: dict[str, Any] = {
        "dataset_root": root,
        "inputs": (("LF", "raw/LF"), ("AF", "raw/AF")),
        "targets": (("HE", "raw/HE"),),
        "reference": "LF",
    }
    return InventoryRequest(**{**values, **overrides})


def _paired(root: Path, *names: str) -> None:
    _touch(root, *(f"raw/{m}/{name}" for m in ("LF", "AF", "HE") for name in names))


def _kinds(preview: InventoryPreview) -> list[str]:
    return [issue.kind for issue in preview.issues]


def _load(
    root: Path, path: Path = Path("inputs/slide_sets.csv"), targets: tuple[str, ...] = ("HE",)
) -> Any:
    return load_slide_set_inventory(
        path, root, modalities=("LF", "AF"), reference_modality="LF", target_modalities=targets
    )


def test_flat_exact_matching_round_trips_through_canonical_loader(tmp_path: Path) -> None:
    _paired(tmp_path, "S002.svs", "S001.svs")
    before = _files(tmp_path)

    preview = preview_inventory(_request(tmp_path))

    assert preview.valid and preview.matched_count == 2 and preview.issues == ()
    assert _files(tmp_path) == before  # preview writes nothing
    written = write_inventory(preview)
    assert written == tmp_path.resolve() / "inputs" / "slide_sets.csv"
    assert written.read_text(encoding="utf-8") == (
        "set_id,input__LF_path,input__LF_aligned,input__AF_path,input__AF_aligned,"
        "target__HE_path,target__HE_aligned\n"
        "S001,raw/LF/S001.svs,true,raw/AF/S001.svs,,raw/HE/S001.svs,\n"
        "S002,raw/LF/S002.svs,true,raw/AF/S002.svs,,raw/HE/S002.svs,\n"
    )
    loaded = _load(tmp_path)
    assert loaded == tuple(sorted((m.slide_set for m in preview.matches), key=lambda s: s.set_id))
    assert [item.already_aligned for item in loaded[0].inputs] == [True, None]
    assert loaded[0].targets[0].already_aligned is None
    assert sorted(p.name for p in written.parent.iterdir()) == ["slide_sets.csv"]


def test_rendering_is_deterministic(tmp_path: Path) -> None:
    _paired(tmp_path, "b/S2.tif", "a/S1.tif")
    first = render_inventory_csv(preview_inventory(_request(tmp_path)))
    assert first == render_inventory_csv(preview_inventory(_request(tmp_path)))


def test_nested_directories_and_glob_anchor(tmp_path: Path) -> None:
    _paired(tmp_path, "case1/S001.svs", "case2/S002.svs")
    _touch(tmp_path, "raw/AF/case1/notes.txt")
    request = _request(tmp_path, inputs=(("LF", "raw/LF"), ("AF", "raw/AF/**/*.svs")))

    preview = preview_inventory(request)

    assert preview.valid
    assert [m.key for m in preview.matches] == ["case1/S001.svs", "case2/S002.svs"]
    assert [m.slide_set.set_id for m in preview.matches] == ["S001", "S002"]
    assert preview.matches[0].slide_set.inputs[1].path == Path("raw/AF/case1/S001.svs")


def test_relative_stem_matches_across_extensions(tmp_path: Path) -> None:
    _touch(tmp_path, "raw/LF/c/S001.svs", "raw/AF/c/S001.tif", "raw/HE/c/S001.tiff")
    strict = preview_inventory(_request(tmp_path))
    assert not strict.valid and strict.matched_count == 0
    assert _kinds(strict) == ["incomplete"] * 3

    stem = preview_inventory(_request(tmp_path, key_rule="relative-stem"))
    assert stem.valid and [m.key for m in stem.matches] == ["c/S001"]
    assert stem.matches[0].slide_set.set_id == "S001"

    # Only the final extension is removed.
    (tmp_path / "raw/HE/c/S001.tiff").rename(tmp_path / "raw/HE/c/S001.ome.tiff")
    assert not preview_inventory(_request(tmp_path, key_rule="relative-stem")).valid


def test_duplicate_stem_keys_within_one_mapping_are_reported(tmp_path: Path) -> None:
    _paired(tmp_path, "S001.svs")
    _touch(tmp_path, "raw/AF/S001.tif")

    preview = preview_inventory(_request(tmp_path, key_rule="relative-stem"))

    assert not preview.valid and _kinds(preview) == ["duplicate"]
    assert "raw/AF/S001.svs" in preview.issues[0].message
    assert "raw/AF/S001.tif" in preview.issues[0].message


def test_incomplete_keys_are_all_reported_without_positional_pairing(tmp_path: Path) -> None:
    # Every mapping holds two files, but the names never line up.
    _touch(tmp_path, "raw/LF/A.svs", "raw/LF/B.svs")
    _touch(tmp_path, "raw/AF/A.svs", "raw/AF/C.svs")
    _touch(tmp_path, "raw/HE/D.svs", "raw/HE/E.svs")

    preview = preview_inventory(_request(tmp_path))

    assert not preview.valid and preview.matched_count == 0
    assert [issue.key for issue in preview.issues] == ["A.svs", "B.svs", "C.svs", "D.svs", "E.svs"]
    assert "no HE asset (found in LF, AF)" in preview.issues[0].message


def test_empty_and_unsafe_specs(tmp_path: Path) -> None:
    _paired(tmp_path, "S001.svs")
    (tmp_path / "raw/empty").mkdir()
    specs = ("raw/empty", "/abs/LF", "raw/../raw/LF", "missing/dir")
    for spec in specs:
        preview = preview_inventory(_request(tmp_path, inputs=(("LF", "raw/LF"), ("AF", spec))))
        assert not preview.valid and "spec" in _kinds(preview), spec


def test_glob_matching_a_directory_is_rejected(tmp_path: Path) -> None:
    _paired(tmp_path, "c/S001.svs")
    preview = preview_inventory(_request(tmp_path, targets=(("HE", "raw/HE/*"),)))
    assert "matches a directory" in preview.issues[0].message


def test_symlinked_assets_and_directories_are_rejected(tmp_path: Path) -> None:
    root = tmp_path / "ds"
    _paired(root, "S001.svs")
    outside = tmp_path / "outside"
    _touch(outside, "S002.svs")
    os.symlink(outside / "S002.svs", root / "raw/LF/S002.svs")
    os.symlink(outside, root / "raw/AF/escape")

    preview = preview_inventory(_request(root))

    messages = " ".join(issue.message for issue in preview.issues)
    assert "raw/LF/S002.svs: symlinks are not followed" in messages
    assert "raw/AF/escape: symlinks are not followed" in messages
    assert "S002.svs" not in {m.key for m in preview.matches}

    os.symlink(root / "raw/HE", root / "linked")
    linked = preview_inventory(_request(root, targets=(("HE", "linked/*.svs"),)))
    assert any("traverses a symlink" in issue.message for issue in linked.issues)


def test_same_file_as_target_and_input_is_rejected(tmp_path: Path) -> None:
    _paired(tmp_path, "S001.svs")
    preview = preview_inventory(_request(tmp_path, targets=(("HE", "raw/AF"),)))
    assert _kinds(preview) == ["conflict"]


def test_same_file_for_two_targets_is_rejected(tmp_path: Path) -> None:
    _paired(tmp_path, "S001.svs")
    preview = preview_inventory(_request(tmp_path, targets=(("HE", "raw/HE"), ("PAS", "raw/HE"))))
    assert _kinds(preview) == ["conflict"]


def test_derived_set_id_collision_and_metadata_resolution(tmp_path: Path) -> None:
    _paired(tmp_path, "patient1/S001.tif", "patient2/S001.tif")
    collided = preview_inventory(_request(tmp_path))
    assert _kinds(collided) == ["set_id"] and "S001" in collided.issues[0].message

    (tmp_path / "meta.csv").write_text(
        "key,set_id,patient_id,specimen_id,input__AF_aligned,input__AF_slide_id,"
        "target__HE_aligned,target__HE_slide_id\n"
        "patient1/S001.tif,P1-S001,P1,SP1,true,af-1,false,he-1\n"
        "patient2/S001.tif,P2-S001,P2,,,,,\n",
        encoding="utf-8",
    )
    preview = preview_inventory(_request(tmp_path, metadata=Path("meta.csv")))
    assert preview.valid, preview.issues
    first = preview.matches[0].slide_set
    assert (first.set_id, first.patient_id, first.specimen_id) == ("P1-S001", "P1", "SP1")
    assert (first.inputs[1].already_aligned, first.inputs[1].slide_id) == (True, "af-1")
    assert (first.targets[0].already_aligned, first.targets[0].slide_id) == (False, "he-1")
    assert preview.matches[1].slide_set.specimen_id is None

    write_inventory(preview)
    loaded = _load(tmp_path)
    assert [s.set_id for s in loaded] == ["P1-S001", "P2-S001"]
    assert loaded[0] == first
    header = (tmp_path / "inputs/slide_sets.csv").read_text(encoding="utf-8").splitlines()[0]
    assert header.endswith(
        "target__HE_aligned,input__AF_slide_id,target__HE_slide_id,patient_id,specimen_id"
    )


@pytest.mark.parametrize(
    ("rows", "kind"),
    [
        ("a/S001.svs,bad id\nb/S001.svs,\n", "set_id"),
        ("a/S001.svs,X\nb/S001.svs,X\n", "set_id"),
    ],
)
def test_unsafe_or_duplicate_supplied_set_ids(tmp_path: Path, rows: str, kind: str) -> None:
    _paired(tmp_path, "a/S001.svs", "b/S001.svs")
    (tmp_path / "meta.csv").write_text("key,set_id\n" + rows, encoding="utf-8")
    preview = preview_inventory(_request(tmp_path, metadata=Path("meta.csv")))
    assert not preview.valid and set(_kinds(preview)) == {kind}


@pytest.mark.parametrize(
    ("csv_text", "expected"),
    [
        ("key,set_id\nS001.svs,A\nS001.svs,B\n", "duplicate key"),
        ("key\nS009.svs\n", "matches no discovered asset"),
        ("key,colour\nS001.svs,red\n", "'colour' is unknown"),
        ("key,input__XR_aligned\nS001.svs,true\n", "names no input modality"),
        ("key,input__AF_mask\nS001.svs,m.png\n", "masks come from mask mappings"),
        ("key,target__HE_aligned\nS001.svs,maybe\n", "must be true, false, or blank"),
        ("key,target_aligned\nS001.svs,true\n", "'target_aligned' is unknown"),
        ("key,target__PAS_slide_id\nS001.svs,x\n", "names no target modality"),
        ("key,input__LF_aligned\nS001.svs,false\n", "reference input LF must be aligned"),
        ("set_id\nS001\n", "requires a 'key' column"),
    ],
)
def test_metadata_rejections(tmp_path: Path, csv_text: str, expected: str) -> None:
    _paired(tmp_path, "S001.svs")
    (tmp_path / "meta.csv").write_text(csv_text, encoding="utf-8")
    preview = preview_inventory(_request(tmp_path, metadata=Path("meta.csv")))
    assert not preview.valid and "metadata" in _kinds(preview)
    assert expected in " ".join(issue.message for issue in preview.issues)


def test_input_and_target_masks(tmp_path: Path) -> None:
    _paired(tmp_path, "S001.svs", "S002.svs")
    _touch(tmp_path, "masks/AF/S001.svs", "masks/HE/S002.svs")
    request = _request(
        tmp_path, input_masks=(("AF", "masks/AF"),), target_masks=(("HE", "masks/HE"),)
    )

    preview = preview_inventory(request)

    assert preview.valid
    first, second = (m.slide_set for m in preview.matches)
    assert first.inputs[1].mask_path == Path("masks/AF/S001.svs")
    assert first.targets[0].mask_path is None and second.inputs[1].mask_path is None
    assert second.targets[0].mask_path == Path("masks/HE/S002.svs")
    write_inventory(preview)
    assert _load(tmp_path) == (first, second)


def test_duplicate_and_unmatched_masks(tmp_path: Path) -> None:
    _paired(tmp_path, "S001.svs")
    _touch(tmp_path, "masks/S001.png", "masks/S001.tif", "masks/S404.png")
    request = _request(tmp_path, target_masks=(("HE", "masks"),), key_rule="relative-stem")
    preview = preview_inventory(request)
    assert sorted(_kinds(preview)) == ["duplicate", "mask"]


def test_request_is_rejected_before_scanning(tmp_path: Path) -> None:
    missing = tmp_path / "never-scanned"
    bad: list[dict[str, Any]] = [
        {"inputs": (("LF", "a"), ("LF", "b"))},
        {"reference": "XR"},
        {"targets": (("AF", "raw/AF"),)},
        {"targets": ()},
        {"targets": (("HE", "a"), ("HE", "b"))},
        {"targets": (("H&E", "a"),)},
        {"input_masks": (("XR", "m"),)},
        {"target_masks": (("PAS", "m"),)},
        {"target_masks": (("HE", "m"), ("HE", "n"))},
        {"key_rule": "fuzzy"},
    ]
    for overrides in bad:
        with pytest.raises(ValueError):
            _request(missing, **overrides)


def test_preview_never_decodes_images(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import PIL.Image

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("an image was opened")

    monkeypatch.setattr(PIL.Image, "open", forbidden)
    _paired(tmp_path, "S001.png")
    write_inventory(preview_inventory(_request(tmp_path)))


@pytest.mark.parametrize("change", ["delete", "add"])
def test_changes_between_preview_and_write_are_detected(tmp_path: Path, change: str) -> None:
    _paired(tmp_path, "S001.svs", "S002.svs")
    preview = preview_inventory(_request(tmp_path))
    if change == "delete":
        (tmp_path / "raw/HE/S002.svs").unlink()
    else:
        _touch(tmp_path, "raw/LF/S003.svs")

    with pytest.raises(ValueError, match="changed since the preview"):
        write_inventory(preview)
    assert not (tmp_path / "inputs/slide_sets.csv").exists()


def test_canonical_validation_failure_publishes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _paired(tmp_path, "S001.svs")
    preview = preview_inventory(_request(tmp_path))
    monkeypatch.setattr(inventory_authoring, "load_slide_set_inventory", lambda *a, **k: ())

    with pytest.raises(ValueError, match="did not reproduce"):
        write_inventory(preview)
    assert list((tmp_path / "inputs").iterdir()) == []


def test_existing_destination_is_never_replaced(tmp_path: Path) -> None:
    _paired(tmp_path, "S001.svs")
    _touch(tmp_path, "inputs/slide_sets.csv")
    preview = preview_inventory(_request(tmp_path))

    with pytest.raises(FileExistsError):
        write_inventory(preview)
    assert (tmp_path / "inputs/slide_sets.csv").read_bytes() == b"placeholder, not an image"
    assert [p.name for p in (tmp_path / "inputs").iterdir()] == ["slide_sets.csv"]


def test_output_must_stay_inside_dataset_root(tmp_path: Path) -> None:
    root = tmp_path / "ds"
    _paired(root, "S001.svs")
    preview = preview_inventory(_request(root))
    for output in (tmp_path / "elsewhere.csv", Path("../elsewhere.csv")):
        with pytest.raises(ValueError, match="inside dataset_root"):
            write_inventory(preview, output)

    written = write_inventory(preview, root / "custom" / "sets.csv")
    assert written == root.resolve() / "custom" / "sets.csv"
    assert _load(root, Path("custom/sets.csv"))


def test_invalid_preview_is_never_written(tmp_path: Path) -> None:
    _touch(tmp_path, "raw/LF/S001.svs", "raw/AF/S001.svs")
    (tmp_path / "raw/HE").mkdir()
    with pytest.raises(ValueError, match="invalid"):
        write_inventory(preview_inventory(_request(tmp_path)))
    assert not (tmp_path / "inputs").exists()


def test_unpaired_domains_need_no_paired_inventory(tmp_path: Path) -> None:
    config = inspect_run_mapping(cyclegan_config_data(tmp_path)).config
    root = config.project.dataset_root
    for domain in ("label_free", "stained"):
        for split in ("train", "val", "test"):
            _touch(root, f"domains/{domain}/{split}/{domain}_{split}.png")

    assert preflight(config, ["train"], depth="assets").valid
    assert not (root / "inputs").exists()

    # Independent domain collections never become A/B pairs.
    preview = preview_inventory(
        InventoryRequest(
            dataset_root=root,
            inputs=(("label_free", "domains/label_free"),),
            targets=(("stained", "domains/stained"),),
            reference="label_free",
        )
    )
    assert preview.matched_count == 0 and set(_kinds(preview)) == {"incomplete"}


def _two_targets(root: Path, *names: str) -> None:
    _touch(root, *(f"raw/{m}/{name}" for m in ("LF", "AF", "HE", "PAS") for name in names))


def test_two_targets_round_trip_with_target_specific_masks_and_metadata(tmp_path: Path) -> None:
    _two_targets(tmp_path, "S001.svs", "S002.svs")
    _touch(tmp_path, "masks/HE/S001.svs", "masks/PAS/S002.svs")
    (tmp_path / "meta.csv").write_text(
        "key,target__HE_aligned,target__HE_slide_id,target__PAS_aligned,target__PAS_slide_id\n"
        "S001.svs,true,he-1,false,pas-1\n",
        encoding="utf-8",
    )
    request = _request(
        tmp_path,
        targets=(("PAS", "raw/PAS"), ("HE", "raw/HE")),
        target_masks=(("HE", "masks/HE"), ("PAS", "masks/PAS")),
        metadata=Path("meta.csv"),
    )

    preview = preview_inventory(request)

    assert preview.valid, preview.issues
    first, second = (m.slide_set for m in preview.matches)
    assert [asset.modality for asset in first.targets] == ["PAS", "HE"]
    pas, he = first.targets
    assert (pas.slide_id, pas.already_aligned, pas.mask_path) == ("pas-1", False, None)
    assert (he.slide_id, he.already_aligned, he.mask_path) == (
        "he-1",
        True,
        Path("masks/HE/S001.svs"),
    )
    assert second.targets[0].mask_path == Path("masks/PAS/S002.svs")
    assert second.targets[1].mask_path is None
    written = write_inventory(preview)
    header = written.read_text(encoding="utf-8").splitlines()[0].split(",")
    assert header[:9] == [
        "set_id",
        "input__LF_path",
        "input__LF_aligned",
        "input__AF_path",
        "input__AF_aligned",
        "target__PAS_path",
        "target__PAS_aligned",
        "target__HE_path",
        "target__HE_aligned",
    ]
    assert {"target__PAS_mask", "target__HE_mask", "target__HE_slide_id"} <= set(header)
    assert _load(tmp_path, targets=("PAS", "HE")) == (first, second)


def test_a_key_needs_exactly_one_asset_for_every_target(tmp_path: Path) -> None:
    _paired(tmp_path, "S001.svs", "S002.svs")
    _touch(tmp_path, "raw/PAS/S001.svs")

    preview = preview_inventory(_request(tmp_path, targets=(("HE", "raw/HE"), ("PAS", "raw/PAS"))))

    assert [m.key for m in preview.matches] == ["S001.svs"]
    assert _kinds(preview) == ["incomplete"]
    assert "no PAS asset" in preview.issues[0].message
