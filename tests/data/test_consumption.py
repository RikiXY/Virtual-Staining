from __future__ import annotations

import os
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from virtual_staining.data.consumption import (
    AssetRow,
    DataLeakageError,
    SnapshotPaths,
    build_snapshot,
    enrich_with_groups,
    load_group_metadata,
    load_snapshot,
    relative_locator,
    validate_groups,
    write_snapshot,
)


def _write(root: Path, locator: str, content: bytes) -> None:
    path = root / locator
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _row(locator: str, split: str = "train", **fields: Any) -> AssetRow:
    return AssetRow(
        root="dataset", locator=locator, role="input", domain="A", split=split, **fields
    )


def _snapshot(root: Path, rows: list[AssetRow], **kwargs: Any) -> Any:
    options: dict[str, Any] = {"hash_policy": "content", "group_validation": "unavailable"}
    options.update(kwargs)
    return build_snapshot(
        rows, kind="consumed", adapter="test/1", roots={"dataset": root}, **options
    )


def _dataset(root: Path) -> list[AssetRow]:
    _write(root, "train/a.png", b"a")
    _write(root, "val/b.png", b"b")
    return [
        _row("train/a.png", patient_id="p1", specimen_id="s1", set_id="x1"),
        _row("val/b.png", "val", patient_id="p2", specimen_id="s2", set_id="x2"),
    ]


# Portable identity


def test_relocated_root_keeps_identity_and_records_binding_separately(tmp_path: Path) -> None:
    first_root, second_root = tmp_path / "mount_a", tmp_path / "mount_b"
    rows = _dataset(first_root)
    shutil.copytree(first_root, second_root)

    first = _snapshot(first_root, rows)
    second = _snapshot(second_root, list(reversed(rows)))

    assert first.snapshot_id == second.snapshot_id
    assert first.membership_sha256 == second.membership_sha256
    assert first.roots != second.roots
    assert first.roots["dataset"] == str(first_root.resolve())


@pytest.mark.parametrize(
    "change",
    [
        {"locator": "train/c.png"},
        {"role": "target"},
        {"domain": "B"},
        {"split": "test"},
        {"patient_id": "p9"},
        {"specimen_id": "s9"},
        {"set_id": "x9"},
    ],
)
def test_semantic_membership_changes_identity(tmp_path: Path, change: dict[str, str]) -> None:
    rows = _dataset(tmp_path)
    _write(tmp_path, "train/c.png", b"c")
    baseline = _snapshot(tmp_path, rows)
    changed = _snapshot(tmp_path, [replace(rows[0], **change), rows[1]])
    assert changed.snapshot_id != baseline.snapshot_id


def test_rows_are_canonically_ordered(tmp_path: Path) -> None:
    rows = _dataset(tmp_path)
    snapshot = _snapshot(tmp_path, list(reversed(rows)))
    assert [row.locator for row in snapshot.rows] == ["train/a.png", "val/b.png"]


# Content hashing


def test_changed_bytes_change_verified_identity(tmp_path: Path) -> None:
    rows = _dataset(tmp_path)
    baseline = _snapshot(tmp_path, rows)
    _write(tmp_path, "train/a.png", b"z")
    changed = _snapshot(tmp_path, rows)
    assert changed.snapshot_id != baseline.snapshot_id
    assert all(row.sha256 and row.sha256.startswith("sha256:") for row in changed.rows)


def test_membership_policy_is_explicitly_unverified(tmp_path: Path) -> None:
    rows = _dataset(tmp_path)
    content = _snapshot(tmp_path, rows)
    membership = _snapshot(tmp_path, rows, hash_policy="membership")
    paths = SnapshotPaths.in_dir(tmp_path / "meta")
    reference = write_snapshot(membership, paths)

    assert membership.snapshot_id != content.snapshot_id
    assert all(row.sha256 is None for row in membership.rows)
    assert reference["content_verified"] is False
    assert any("not a content-identical freeze" in text for text in membership.limitations)
    # Same-size byte edits are invisible to membership identity, by design.
    _write(tmp_path, "train/a.png", b"q")
    assert _snapshot(tmp_path, rows, hash_policy="membership").snapshot_id == (
        membership.snapshot_id
    )


def test_hash_failure_does_not_fall_back_to_membership(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = _dataset(tmp_path)

    def fail(_path: Path) -> tuple[str, int]:
        raise RuntimeError("File changed while it was being hashed")

    monkeypatch.setattr("virtual_staining.data.consumption.sha256_file_verified", fail)
    with pytest.raises(RuntimeError, match="changed while it was being hashed"):
        _snapshot(tmp_path, rows)


# Path safety


def test_path_safety(tmp_path: Path) -> None:
    root = tmp_path / "root"
    _write(root, "ok.png", b"ok")
    _write(tmp_path, "outside.png", b"outside")
    (root / "escape.png").symlink_to(tmp_path / "outside.png")

    for locator in ("/abs.png", "../outside.png", "a/../ok.png", "./ok.png", ""):
        with pytest.raises(ValueError, match="normalized root-relative"):
            _snapshot(root, [_row(locator)])
    with pytest.raises(ValueError, match="resolves outside its root"):
        _snapshot(root, [_row("escape.png")])
    with pytest.raises(ValueError, match="outside the snapshot root"):
        relative_locator(root, tmp_path / "outside.png")
    assert relative_locator(root, root / "ok.png") == "ok.png"


def test_aliases_of_one_file_across_splits_fail(tmp_path: Path) -> None:
    _write(tmp_path, "train/a.png", b"a")
    (tmp_path / "val").mkdir()
    (tmp_path / "val" / "link.png").symlink_to(tmp_path / "train" / "a.png")
    os.link(tmp_path / "train" / "a.png", tmp_path / "val" / "hard.png")

    for alias in ("val/link.png", "val/hard.png"):
        with pytest.raises(DataLeakageError, match="same file appears in disjoint splits"):
            _snapshot(
                tmp_path,
                [_row("train/a.png"), _row(alias, "val")],
                hash_policy="membership",
            )
    with pytest.raises(DataLeakageError, match="same file"):
        _snapshot(tmp_path, [_row("train/a.png"), _row("train/a.png", "val")])


# Duplicate content


def test_identical_bytes_across_splits_fail_but_are_recorded_within_a_split(
    tmp_path: Path,
) -> None:
    _write(tmp_path, "train/a.png", b"same")
    _write(tmp_path, "train/b.png", b"same")
    _write(tmp_path, "val/c.png", b"same")

    with pytest.raises(DataLeakageError, match="same content appears in disjoint splits"):
        _snapshot(tmp_path, [_row("train/a.png"), _row("val/c.png", "val")])

    within = _snapshot(tmp_path, [_row("train/a.png"), _row("train/b.png")])
    assert len(within.rows) == 2
    assert within.duplicates == (
        {
            "kind": "same_content",
            "split": "train",
            "sha256": within.rows[0].sha256,
            "locations": ["dataset:train/a.png", "dataset:train/b.png"],
        },
    )


def test_different_bytes_make_no_biological_independence_claim(tmp_path: Path) -> None:
    snapshot = _snapshot(tmp_path, _dataset(tmp_path))
    assert snapshot.group_validation["status"] == "unavailable"
    assert snapshot.group_validation["unit"] is None
    assert any("do not prove" in text for text in snapshot.limitations)


# Biological groups


def test_group_validation_selects_strongest_complete_unit(tmp_path: Path) -> None:
    rows = _dataset(tmp_path)
    assert validate_groups(rows, "auto")["unit"] == "patient"
    partial = [replace(rows[0], patient_id=""), rows[1]]
    result = validate_groups(partial, "auto")
    assert result["unit"] == "specimen"
    assert result["incomplete_stronger_units"] == ["patient"]
    with pytest.raises(ValueError, match="requires a patient_id for every asset"):
        validate_groups(partial, "patient")


@pytest.mark.parametrize(
    ("unit", "splits"),
    [("patient", ("train", "test")), ("specimen", ("train", "val")), ("set", ("val", "test"))],
)
def test_group_leakage_across_splits_fails(unit: str, splits: tuple[str, str]) -> None:
    rows = [
        _row("a.png", splits[0], patient_id="p1", specimen_id="s1", set_id="x1"),
        _row("b.png", splits[1], patient_id="p2", specimen_id="s2", set_id="x2"),
    ]
    rows[1] = replace(rows[1], **{f"{unit}_id": getattr(rows[0], f"{unit}_id")})
    with pytest.raises(DataLeakageError, match=f"{unit}_id values appear in more than one split"):
        validate_groups(rows, unit)


def test_same_group_across_domains_within_a_split_is_allowed() -> None:
    rows = [
        _row("a.png", patient_id="p1", specimen_id="s1", set_id="x1"),
        replace(_row("b.png", patient_id="p1", specimen_id="s1", set_id="x1"), domain="B"),
        _row("c.png", "val", patient_id="p2", specimen_id="s2", set_id="x2"),
    ]
    assert validate_groups(rows, "patient")["status"] == "validated"


def test_same_group_across_domains_and_splits_fails() -> None:
    rows = [
        _row("a.png", patient_id="p1", specimen_id="s1", set_id="x1"),
        replace(_row("b.png", "val", patient_id="p1", specimen_id="s2", set_id="x2"), domain="B"),
    ]
    with pytest.raises(DataLeakageError, match="patient_id"):
        validate_groups(rows, "auto")


def test_missing_group_metadata_requires_explicit_unavailable(tmp_path: Path) -> None:
    rows = [_row("train/a.png"), _row("val/b.png", "val")]
    _dataset(tmp_path)
    with pytest.raises(ValueError, match="set data.group_validation: unavailable"):
        _snapshot(tmp_path, rows, group_validation="auto")
    snapshot = _snapshot(tmp_path, rows, group_validation="unavailable")
    reference = write_snapshot(snapshot, SnapshotPaths.in_dir(tmp_path / "meta"))
    stored = load_snapshot(SnapshotPaths.in_dir(tmp_path / "meta"))
    assert reference["snapshot_id"] == stored.snapshot_id
    assert stored.group_validation == {
        "requested": "unavailable",
        "splits": ["train", "val"],
        "unit": None,
        "status": "unavailable",
    }
    assert any("group_validation=unavailable" in text for text in stored.limitations)
    assert all(not row.patient_id for row in stored.rows)


def test_sidecar_enriches_only_listed_paths_and_rejects_conflicts(tmp_path: Path) -> None:
    sidecar = tmp_path / "groups.csv"
    sidecar.write_text(
        "path,domain,split,set_id,specimen_id,patient_id\n"
        "train/a.png,A,train,x1,s1,p1\n"
        "test/z.png,A,test,x9,s9,p1\n",
        encoding="utf-8",
    )
    groups = load_group_metadata(sidecar)
    rows = enrich_with_groups([_row("train/a.png"), _row("val/b.png", "val")], groups)
    assert rows[0].patient_id == "p1"
    assert rows[1].patient_id == ""
    with pytest.raises(DataLeakageError, match="patient_id"):
        validate_groups(rows, "auto", groups)
    with pytest.raises(ValueError, match="declares 'train/a.png' as A/train"):
        enrich_with_groups([_row("train/a.png", "val")], groups)
    sidecar.write_text("path,domain\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exactly the columns"):
        load_group_metadata(sidecar)


# Persistence


def test_snapshot_round_trip_and_torn_pairs_are_rejected(tmp_path: Path) -> None:
    snapshot = _snapshot(tmp_path, _dataset(tmp_path))
    paths = SnapshotPaths.in_dir(tmp_path / "meta")
    reference = write_snapshot(snapshot, paths)

    assert reference["rows_path"] == str(paths.rows)
    assert load_snapshot(paths).rows == snapshot.rows
    assert not list(paths.metadata.parent.glob(".*.tmp"))
    paths.rows.write_text(paths.rows.read_text() + "extra\n", encoding="utf-8")
    with pytest.raises(ValueError, match="do not match"):
        load_snapshot(paths)


def test_missing_files_are_recorded_only_when_allowed(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        _snapshot(tmp_path, [_row("absent.png")])
    snapshot = _snapshot(tmp_path, [_row("absent.png")], allow_missing=True)
    assert snapshot.rows[0].status == "missing"
    assert snapshot.rows[0].sha256 is None
