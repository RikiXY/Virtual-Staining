from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from virtual_staining.data import consumption
from virtual_staining.data.consumption import (
    SNAPSHOT_SCHEMA_VERSION,
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
from virtual_staining.utils.hashing import sha256_bytes, sha256_json


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
    first_paths = SnapshotPaths.in_dir(tmp_path / "first")
    second_paths = SnapshotPaths.in_dir(tmp_path / "second")
    first_reference = write_snapshot(first, first_paths)
    second_reference = write_snapshot(second, second_paths)
    assert first_reference["snapshot_id"] == second_reference["snapshot_id"]
    assert first_paths.rows.read_bytes() == second_paths.rows.read_bytes()
    assert load_snapshot(second_paths).roots == second.roots


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


@pytest.mark.parametrize(
    "rows",
    [
        # Partial metadata: only the colliding assets carry a patient ID.
        [
            _row("a.png", patient_id="p1"),
            _row("b.png", "val"),
            _row("c.png", "test", patient_id="p1"),
        ],
        # Complete metadata.
        [
            _row("a.png", patient_id="p1"),
            _row("b.png", "val", patient_id="p2"),
            _row("c.png", "test", patient_id="p1"),
        ],
    ],
)
def test_unavailable_does_not_ignore_supplied_group_contradictions(rows: list[AssetRow]) -> None:
    with pytest.raises(DataLeakageError, match="patient_id values appear in more than one split"):
        validate_groups(rows, "unavailable")


def test_unavailable_with_partial_consistent_metadata_makes_no_claim() -> None:
    rows = [_row("a.png", patient_id="p1"), _row("b.png", "val"), _row("c.png", "test")]
    assert validate_groups(rows, "unavailable") == {
        "requested": "unavailable",
        "splits": ["test", "train", "val"],
        "unit": None,
        "status": "unavailable",
    }


def test_declared_patch_split_records_shared_groups_only_under_unavailable() -> None:
    rows = [
        _row("a.png", patient_id="p1", set_id="x1"),
        _row("b.png", "test", patient_id="p1", set_id="x1"),
    ]
    result = validate_groups(rows, "unavailable", patch_split=True)
    assert result["status"] == "unavailable"
    assert result["groups_shared_across_splits"] == {"patient": 1, "set": 1}
    for requested in ("auto", "set"):
        with pytest.raises(DataLeakageError, match="patient_id"):
            validate_groups(rows, requested, patch_split=True)


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
        validate_groups([*rows, *groups], "auto")
    with pytest.raises(ValueError, match="declares 'train/a.png' as A/train"):
        enrich_with_groups([_row("train/a.png", "val")], groups)
    sidecar.write_text("path,domain\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exactly the columns"):
        load_group_metadata(sidecar)


# Persistence


@pytest.mark.parametrize("hash_policy", ["content", "membership"])
@pytest.mark.parametrize("empty", [False, True])
def test_publication_preserves_canonical_identity_and_artifacts(
    tmp_path: Path, hash_policy: str, empty: bool
) -> None:
    rows = [] if empty else _dataset(tmp_path)
    if rows:
        rows[0] = replace(rows[0], sample_id="é,1")
    snapshot = _snapshot(
        tmp_path,
        list(reversed(rows)),
        hash_policy=hash_policy,
        selection={"domains": ["A"], "example": {"z": None, "a": "é"}},
        context={"protocol": "paired"},
        sources={"inventory": "inventory.csv", "sha256": sha256_bytes(b"inventory")},
    )
    membership = sha256_json([asdict(row) for row in snapshot.rows])
    identity_payload = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "kind": snapshot.kind,
        "adapter": snapshot.adapter,
        "hash_policy": snapshot.hash_policy,
        "selection": snapshot.selection,
        "context": snapshot.context,
        "membership_sha256": membership,
    }
    snapshot_id = sha256_json(identity_payload)
    paths = SnapshotPaths.in_dir(tmp_path / "meta")
    created_at = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
    with (
        patch.object(consumption, "sha256_json", wraps=sha256_json) as hashing,
        patch.object(consumption, "datetime") as clock,
    ):
        clock.now.return_value = created_at
        reference = write_snapshot(snapshot, paths)
        # Count the actual canonical hashes, not unrelated rows-file or content hashes.
        assert hashing.call_count == 2
        assert hashing.call_args_list[0].args == ([asdict(row) for row in snapshot.rows],)
        assert hashing.call_args_list[1].args == (identity_payload,)

    expected_csv = (
        "root,locator,role,domain,split,sample_id,set_id,specimen_id,patient_id,"
        "status,size,sha256\n"
    )
    if not empty:
        first_hash = sha256_bytes(b"a") if hash_policy == "content" else ""
        second_hash = sha256_bytes(b"b") if hash_policy == "content" else ""
        expected_csv += (
            f'dataset,train/a.png,input,A,train,"é,1",x1,s1,p1,present,1,{first_hash}\n'
            f"dataset,val/b.png,input,A,val,,x2,s2,p2,present,1,{second_hash}\n"
        )
    assert paths.rows.read_bytes() == expected_csv.encode("utf-8")
    expected_metadata = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "kind": snapshot.kind,
        "adapter": snapshot.adapter,
        "snapshot_id": snapshot_id,
        "membership_sha256": membership,
        "hash_policy": hash_policy,
        "content_verified": hash_policy == "content",
        "selection": snapshot.selection,
        "context": snapshot.context,
        "sources": snapshot.sources,
        "group_validation": snapshot.group_validation,
        "duplicates": list(snapshot.duplicates),
        "limitations": list(snapshot.limitations),
        "validation_context": snapshot.validation_context,
        "root_binding": snapshot.roots,
        "row_count": len(rows),
        "rows_file": paths.rows.name,
        "rows_sha256": sha256_bytes(expected_csv.encode("utf-8")),
        "created_at": created_at.isoformat(),
    }
    assert paths.metadata.read_bytes() == (json.dumps(expected_metadata, indent=2) + "\n").encode(
        "utf-8"
    )
    assert reference == {
        "snapshot_id": snapshot_id,
        "membership_sha256": membership,
        "hash_policy": hash_policy,
        "content_verified": hash_policy == "content",
        "row_count": len(rows),
        "metadata_path": str(paths.metadata),
        "rows_path": str(paths.rows),
    }
    assert snapshot.membership_sha256 == membership
    assert snapshot.snapshot_id == snapshot_id
    assert snapshot.reference(paths) == reference
    assert load_snapshot(paths) == snapshot


@pytest.mark.parametrize(
    "field", ["selection", "context", "sources", "group_validation", "validation_context"]
)
def test_publications_observe_nested_mutation_without_caching(tmp_path: Path, field: str) -> None:
    rows = _dataset(tmp_path)
    _write(tmp_path, "test/c.png", b"c")
    snapshot = _snapshot(
        tmp_path,
        rows,
        selection={"split": ["train", "val"]},
        context={"protocol": "paired"},
        sources={"inventory": {"path": "inventory.csv"}},
        validation_context=[_row("test/c.png", "test")],
    )
    nested = getattr(snapshot, field)
    nested["example"] = {"value": "original"}
    first_paths = SnapshotPaths.in_dir(tmp_path / "first")
    first = write_snapshot(snapshot, first_paths)
    repeated_paths = SnapshotPaths.in_dir(tmp_path / "repeated")
    repeated = write_snapshot(snapshot, repeated_paths)
    assert repeated == {
        **first,
        "metadata_path": str(repeated_paths.metadata),
        "rows_path": str(repeated_paths.rows),
    }
    first_metadata = json.loads(first_paths.metadata.read_bytes())
    repeated_metadata = json.loads(repeated_paths.metadata.read_bytes())
    first_metadata.pop("created_at")
    repeated_metadata.pop("created_at")
    assert repeated_metadata == first_metadata
    assert repeated_paths.rows.read_bytes() == first_paths.rows.read_bytes()

    nested["example"]["value"] = "changed"
    second_paths = SnapshotPaths.in_dir(tmp_path / "second")
    second = write_snapshot(snapshot, second_paths)
    assert (second["snapshot_id"] != first["snapshot_id"]) == (field in {"selection", "context"})
    assert second["membership_sha256"] == first["membership_sha256"]
    assert second["snapshot_id"] == snapshot.snapshot_id
    assert second == snapshot.reference(second_paths)
    assert json.loads(second_paths.metadata.read_bytes())[field] == nested
    assert load_snapshot(second_paths) == snapshot
    assert first_metadata[field]["example"]["value"] == "original"


@pytest.mark.parametrize("missing", ["metadata", "rows"])
def test_missing_snapshot_artifact_is_rejected(tmp_path: Path, missing: str) -> None:
    snapshot = _snapshot(tmp_path, _dataset(tmp_path))
    paths = SnapshotPaths.in_dir(tmp_path / "meta")
    write_snapshot(snapshot, paths)
    getattr(paths, missing).unlink()
    with pytest.raises(FileNotFoundError):
        load_snapshot(paths)


def test_mismatched_snapshot_identity_is_rejected(tmp_path: Path) -> None:
    snapshot = _snapshot(tmp_path, _dataset(tmp_path))
    paths = SnapshotPaths.in_dir(tmp_path / "meta")
    write_snapshot(snapshot, paths)
    metadata = json.loads(paths.metadata.read_bytes())
    metadata["context"]["protocol"] = "changed"
    paths.metadata.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="Snapshot identity mismatch"):
        load_snapshot(paths)


@pytest.mark.parametrize("artifact", ["rows", "metadata"])
def test_publication_failure_preserves_error_and_cleans_temporary_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, artifact: str
) -> None:
    rows = _dataset(tmp_path)
    paths = SnapshotPaths.in_dir(tmp_path / "meta")
    write_snapshot(_snapshot(tmp_path, rows), paths)
    old_rows, old_metadata = paths.rows.read_bytes(), paths.metadata.read_bytes()
    _write(tmp_path, "train/a.png", b"changed")
    snapshot = _snapshot(tmp_path, rows)
    replace_path = Path.replace
    failure = OSError("injected publication failure")

    def fail(source: Path, target: Path) -> Path:
        if target == getattr(paths, artifact):
            raise failure
        return replace_path(source, target)

    monkeypatch.setattr(Path, "replace", fail)
    with pytest.raises(OSError, match="injected publication failure") as raised:
        write_snapshot(snapshot, paths)
    assert raised.value is failure
    assert not list(paths.metadata.parent.glob(".*.tmp"))
    assert paths.metadata.read_bytes() == old_metadata
    if artifact == "rows":
        assert paths.rows.read_bytes() == old_rows
        load_snapshot(paths)
    else:
        assert paths.rows.read_bytes() != old_rows
        with pytest.raises(ValueError, match="do not match"):
            load_snapshot(paths)


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


# Held-out validation context


def test_validation_context_takes_part_in_leakage_checks_without_becoming_rows(
    tmp_path: Path,
) -> None:
    rows = _dataset(tmp_path)
    _write(tmp_path, "test/x.png", b"x")
    _write(tmp_path, "test/y.png", b"x")  # within-context duplicate: not listed
    context = [_row("test/x.png", "test"), _row("test/y.png", "test")]
    snapshot = _snapshot(tmp_path, rows, validation_context=context)

    assert [row.locator for row in snapshot.rows] == ["train/a.png", "val/b.png"]
    assert snapshot.duplicates == ()
    assert snapshot.group_validation["splits"] == ["test", "train", "val"]
    assert snapshot.validation_context["row_count"] == 2
    assert snapshot.validation_context["splits"] == ["test"]
    # The held-out context is checked, not consumed: it never enters the identity.
    assert snapshot.snapshot_id == _snapshot(tmp_path, rows).snapshot_id
    paths = SnapshotPaths.in_dir(tmp_path / "meta")
    write_snapshot(snapshot, paths)
    assert load_snapshot(paths).validation_context == snapshot.validation_context

    _write(tmp_path, "test/x.png", b"a")  # bytes of train/a.png under another name
    with pytest.raises(DataLeakageError, match="same content appears in disjoint splits"):
        _snapshot(tmp_path, rows, validation_context=context)
    # Membership mode cannot assert byte identity.
    _snapshot(tmp_path, rows, hash_policy="membership", validation_context=context)


def test_validation_context_aliases_and_groups_fail_in_membership_mode(tmp_path: Path) -> None:
    rows = _dataset(tmp_path)
    (tmp_path / "test").mkdir()
    os.link(tmp_path / "val" / "b.png", tmp_path / "test" / "hard.png")
    (tmp_path / "test" / "link.png").symlink_to(tmp_path / "train" / "a.png")
    for alias in ("test/hard.png", "test/link.png"):
        with pytest.raises(DataLeakageError, match="same file appears in disjoint splits"):
            _snapshot(
                tmp_path, rows, hash_policy="membership", validation_context=[_row(alias, "test")]
            )

    _write(tmp_path, "test/z.png", b"z")
    with pytest.raises(DataLeakageError, match="patient_id"):
        _snapshot(
            tmp_path,
            rows,
            hash_policy="membership",
            validation_context=[_row("test/z.png", "test", patient_id="p2")],
        )
