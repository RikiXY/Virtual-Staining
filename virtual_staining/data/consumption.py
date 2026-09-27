"""Versioned snapshots of the exact files a tracked stage consumed or produced.

A snapshot is a deterministic list of semantic asset rows plus compact metadata. Rows carry
root-relative locators, so the portable identity (``membership_sha256`` / ``snapshot_id``)
never depends on where a root is mounted, on timestamps, or on traversal order; absolute
roots are recorded only as a machine-local binding.

Adapters (the stage applications) resolve their inputs once, build rows from those exact
objects, and hand the same objects to the consumer. This module only observes, validates,
hashes, and persists; it never re-discovers files.
"""

from __future__ import annotations

import csv
import io
import json
import os
import uuid
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from virtual_staining.utils.hashing import sha256_bytes, sha256_file_verified, sha256_json

SNAPSHOT_SCHEMA_VERSION = 1
HASH_POLICIES = ("content", "membership")
GROUP_UNITS = ("patient", "specimen", "set")
ROW_FIELDS = (
    "root",
    "locator",
    "role",
    "domain",
    "split",
    "sample_id",
    "set_id",
    "specimen_id",
    "patient_id",
    "status",
    "size",
    "sha256",
)
ROLES = frozenset({"input", "target", "mask", "generated", "reference"})
GROUP_METADATA_FIELDS = ("path", "domain", "split", "set_id", "specimen_id", "patient_id")
PROVENANCE_LIMITATIONS = (
    "File provenance does not establish biological independence: distinct SHA-256 digests "
    "do not prove that two images come from different patients, specimens, or slides.",
    "Re-encoded or visually near-duplicate images are not detected.",
)
_UNAVAILABLE_LIMITATION = (
    "group_validation=unavailable: no patient, specimen, or set independence between splits "
    "is claimed or checked for this snapshot."
)
_MEMBERSHIP_LIMITATION = (
    "hash_policy=membership: file bytes were not hashed; the identity covers locators, sizes, "
    "and semantic metadata only and is not a content-identical freeze."
)


class DataLeakageError(ValueError):
    """The same file, bytes, or biological group appears in disjoint splits."""


@dataclass(frozen=True)
class SnapshotPaths:
    metadata: Path
    rows: Path

    @classmethod
    def in_dir(cls, directory: Path) -> SnapshotPaths:
        return cls(directory / "snapshot.json", directory / "rows.csv")

    def remove(self) -> None:
        self.metadata.unlink(missing_ok=True)
        self.rows.unlink(missing_ok=True)


@dataclass(frozen=True)
class AssetRow:
    root: str
    locator: str
    role: str
    domain: str = ""
    split: str = ""
    sample_id: str = ""
    set_id: str = ""
    specimen_id: str = ""
    patient_id: str = ""
    status: str = "present"
    size: int | None = None
    sha256: str | None = None

    def sort_key(self) -> tuple[str, ...]:
        return tuple("" if value is None else str(value) for value in asdict(self).values())

    def group_id(self, unit: str) -> str:
        return str(getattr(self, f"{unit}_id"))


@dataclass(frozen=True)
class DataSnapshot:
    kind: str
    adapter: str
    hash_policy: str
    rows: tuple[AssetRow, ...]
    roots: dict[str, str]
    selection: dict[str, Any] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)
    sources: dict[str, Any] = field(default_factory=dict)
    group_validation: dict[str, Any] = field(default_factory=dict)
    duplicates: tuple[dict[str, Any], ...] = ()
    limitations: tuple[str, ...] = ()

    @property
    def membership_sha256(self) -> str:
        return sha256_json([asdict(row) for row in self.rows])

    @property
    def snapshot_id(self) -> str:
        return sha256_json(
            {
                "schema_version": SNAPSHOT_SCHEMA_VERSION,
                "kind": self.kind,
                "adapter": self.adapter,
                "hash_policy": self.hash_policy,
                "selection": self.selection,
                "context": self.context,
                "membership_sha256": self.membership_sha256,
            }
        )

    def reference(self, paths: SnapshotPaths) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "membership_sha256": self.membership_sha256,
            "hash_policy": self.hash_policy,
            "content_verified": self.hash_policy == "content",
            "row_count": len(self.rows),
            "metadata_path": str(paths.metadata),
            "rows_path": str(paths.rows),
        }


def validate_locator(locator: str) -> str:
    path = PurePosixPath(locator)
    if (
        not locator
        or "\\" in locator
        or path.is_absolute()
        or ".." in path.parts
        or path.as_posix() != locator
        or locator in {".", ""}
    ):
        raise ValueError(f"Snapshot locator must be a normalized root-relative path: {locator!r}")
    return locator


def relative_locator(root: Path, path: Path) -> str:
    """Return the lexical root-relative locator of ``path``; reject paths outside ``root``."""
    try:
        relative = Path(os.path.abspath(path)).relative_to(os.path.abspath(root))
    except ValueError:
        raise ValueError(f"{path} is outside the snapshot root {root}") from None
    return validate_locator(relative.as_posix())


def _observe(
    row: AssetRow, root: Path, hash_policy: str, *, allow_missing: bool
) -> tuple[AssetRow, tuple[int, int] | None]:
    validate_locator(row.locator)
    candidate = root / row.locator
    if not os.path.lexists(candidate):
        if allow_missing:
            return replace(row, status="missing", size=None, sha256=None), None
        raise FileNotFoundError(f"Consumed file not found: {candidate}")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"Consumed file {candidate} resolves outside its root {root}")
    if not resolved.is_file():
        raise ValueError(f"Consumed path is not a regular file: {candidate}")
    stat = resolved.stat()
    if hash_policy == "content":
        digest, size = sha256_file_verified(resolved)
    else:
        digest, size = None, stat.st_size
    return replace(row, status="present", size=size, sha256=digest), (stat.st_dev, stat.st_ino)


def _check_duplicates(
    observed: Sequence[tuple[AssetRow, tuple[int, int] | None]],
) -> tuple[dict[str, Any], ...]:
    by_file: dict[tuple[int, int], list[AssetRow]] = defaultdict(list)
    by_content: dict[str, list[AssetRow]] = defaultdict(list)
    for row, key in observed:
        if key is not None:
            by_file[key].append(row)
        if row.sha256 is not None:
            by_content[row.sha256].append(row)
    duplicates: list[dict[str, Any]] = []
    for kind, groups in (("same_file", by_file), ("same_content", by_content)):
        for rows in groups.values():
            locations = sorted({(row.root, row.locator) for row in rows})
            if len(locations) < 2 and kind == "same_content":
                continue
            splits = sorted({row.split for row in rows if row.split})
            if len(splits) > 1:
                described = ", ".join(f"{root}:{locator}" for root, locator in locations)
                raise DataLeakageError(
                    f"{kind.replace('_', ' ')} appears in disjoint splits {splits}: {described}"
                )
            if len(locations) > 1:
                duplicates.append(
                    {
                        "kind": kind,
                        "split": splits[0] if splits else "",
                        "sha256": rows[0].sha256,
                        "locations": [f"{root}:{locator}" for root, locator in locations],
                    }
                )
    return tuple(sorted(duplicates, key=lambda item: (item["kind"], item["locations"])))


def validate_groups(
    rows: Iterable[AssetRow], requested: str, context: Iterable[AssetRow] = ()
) -> dict[str, Any]:
    """Check biological-group independence across splits at the strongest declared unit.

    ``context`` rows are not consumed but share the split partition (for example the held-out
    test split during training) and take part in the leakage check. Any observed group that
    spans splits fails, even at a unit too incomplete to be claimed.
    """
    items = [*rows, *context]
    splits = sorted({row.split for row in items if row.split})
    result: dict[str, Any] = {"requested": requested, "splits": splits}
    if len(splits) < 2:
        return {**result, "unit": None, "status": "not_applicable"}
    if requested == "unavailable":
        return {**result, "unit": None, "status": "unavailable"}
    for unit in GROUP_UNITS:
        group_splits: dict[str, set[str]] = defaultdict(set)
        for row in items:
            if row.group_id(unit):
                group_splits[row.group_id(unit)].add(row.split)
        leaked = sorted(group for group, values in group_splits.items() if len(values) > 1)
        if leaked:
            raise DataLeakageError(
                f"{unit}_id values appear in more than one split: {leaked[:10]}"
                + (f" (+{len(leaked) - 10} more)" if len(leaked) > 10 else "")
            )
    complete = [unit for unit in GROUP_UNITS if all(row.group_id(unit) for row in items)]
    candidates = GROUP_UNITS if requested == "auto" else (requested,)
    for unit in candidates:
        if unit in complete:
            return {
                **result,
                "unit": unit,
                "status": "validated",
                "incomplete_stronger_units": list(GROUP_UNITS[: GROUP_UNITS.index(unit)]),
            }
        if requested != "auto":
            missing = sum(not row.group_id(unit) for row in items)
            raise ValueError(
                f"data.group_validation={unit!r} requires a {unit}_id for every asset; "
                f"{missing} of {len(items)} assets have none"
            )
    raise ValueError(
        "No biological group metadata (patient_id, specimen_id, or set_id) is available for "
        "every asset, so split independence cannot be validated. Supply data.group_metadata "
        "or set data.group_validation: unavailable explicitly."
    )


def build_snapshot(
    rows: Iterable[AssetRow],
    *,
    kind: str,
    adapter: str,
    roots: Mapping[str, Path],
    hash_policy: str,
    group_validation: str = "auto",
    group_context: Iterable[AssetRow] = (),
    selection: Mapping[str, Any] | None = None,
    context: Mapping[str, Any] | None = None,
    sources: Mapping[str, Any] | None = None,
    allow_missing: bool = False,
) -> DataSnapshot:
    """Observe ``rows`` under ``hash_policy``, validate splits/groups, and canonicalize them."""
    if hash_policy not in HASH_POLICIES:
        raise ValueError(f"Unsupported hash policy {hash_policy!r}")
    if kind not in {"consumed", "produced"}:
        raise ValueError(f"Unsupported snapshot kind {kind!r}")
    observed = []
    for row in rows:
        if row.role not in ROLES:
            raise ValueError(f"Unsupported snapshot role {row.role!r}")
        if row.root not in roots:
            raise ValueError(f"Snapshot row references unbound root {row.root!r}")
        observed.append(_observe(row, roots[row.root], hash_policy, allow_missing=allow_missing))
    duplicates = _check_duplicates(observed)
    canonical = tuple(sorted((row for row, _ in observed), key=AssetRow.sort_key))
    groups = validate_groups(canonical, group_validation, group_context)
    limitations: list[str] = list(PROVENANCE_LIMITATIONS)
    if hash_policy == "membership":
        limitations.append(_MEMBERSHIP_LIMITATION)
    if groups["status"] == "unavailable":
        limitations.append(_UNAVAILABLE_LIMITATION)
    return DataSnapshot(
        kind=kind,
        adapter=adapter,
        hash_policy=hash_policy,
        rows=canonical,
        roots={name: str(Path(path).resolve()) for name, path in sorted(roots.items())},
        selection=_plain(selection or {}),
        context=_plain(context or {}),
        sources=_plain(sources or {}),
        group_validation=groups,
        duplicates=duplicates,
        limitations=tuple(limitations),
    )


def _plain(value: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(dict(value), default=str))


def _rows_bytes(rows: Sequence[AssetRow]) -> bytes:
    buffer = io.StringIO()
    writer: csv.DictWriter[str] = csv.DictWriter(
        buffer, fieldnames=list(ROW_FIELDS), lineterminator="\n"
    )
    writer.writeheader()
    for row in rows:
        writer.writerow({key: "" if value is None else value for key, value in asdict(row).items()})
    return buffer.getvalue().encode("utf-8")


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_snapshot(snapshot: DataSnapshot, paths: SnapshotPaths) -> dict[str, Any]:
    """Atomically publish rows then metadata; the metadata binds the rows file digest.

    Returns the compact reference recorded in stage records. Local single-writer store: a
    concurrent writer is not coordinated, but a torn pair is detected by :func:`load_snapshot`.
    """
    rows = _rows_bytes(snapshot.rows)
    _atomic_write(paths.rows, rows)
    metadata = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "kind": snapshot.kind,
        "adapter": snapshot.adapter,
        "snapshot_id": snapshot.snapshot_id,
        "membership_sha256": snapshot.membership_sha256,
        "hash_policy": snapshot.hash_policy,
        "content_verified": snapshot.hash_policy == "content",
        "selection": snapshot.selection,
        "context": snapshot.context,
        "sources": snapshot.sources,
        "group_validation": snapshot.group_validation,
        "duplicates": list(snapshot.duplicates),
        "limitations": list(snapshot.limitations),
        "root_binding": snapshot.roots,
        "row_count": len(snapshot.rows),
        "rows_file": paths.rows.name,
        "rows_sha256": sha256_bytes(rows),
        "created_at": datetime.now(UTC).isoformat(),
    }
    _atomic_write(paths.metadata, (json.dumps(metadata, indent=2) + "\n").encode("utf-8"))
    return snapshot.reference(paths)


def _parse_row(raw: Mapping[str, str]) -> AssetRow:
    values: dict[str, Any] = dict(raw)
    values["size"] = int(raw["size"]) if raw["size"] else None
    values["sha256"] = raw["sha256"] or None
    return AssetRow(**values)


def load_snapshot(paths: SnapshotPaths) -> DataSnapshot:
    """Read a snapshot, rejecting unsupported schemas and mismatched rows/metadata pairs."""
    metadata = json.loads(paths.metadata.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict) or metadata.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported data snapshot schema at {paths.metadata}")
    payload = paths.rows.read_bytes()
    if sha256_bytes(payload) != metadata.get("rows_sha256"):
        raise ValueError(f"Snapshot rows {paths.rows} do not match {paths.metadata}")
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8")))
    if tuple(reader.fieldnames or ()) != ROW_FIELDS:
        raise ValueError(f"Snapshot rows {paths.rows} have unsupported columns")
    snapshot = DataSnapshot(
        kind=metadata["kind"],
        adapter=metadata["adapter"],
        hash_policy=metadata["hash_policy"],
        rows=tuple(_parse_row(row) for row in reader),
        roots=dict(metadata["root_binding"]),
        selection=metadata["selection"],
        context=metadata["context"],
        sources=metadata["sources"],
        group_validation=metadata["group_validation"],
        duplicates=tuple(metadata["duplicates"]),
        limitations=tuple(metadata["limitations"]),
    )
    if snapshot.snapshot_id != metadata.get("snapshot_id"):
        raise ValueError(f"Snapshot identity mismatch at {paths.metadata}")
    return snapshot


def load_group_metadata(path: Path) -> tuple[AssetRow, ...]:
    """Parse a raw-collection sidecar of ``path,domain,split,set_id,specimen_id,patient_id``.

    Paths are root-relative locators. Group IDs are taken only from this explicit file; none
    are derived from filenames or directories.
    """
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != GROUP_METADATA_FIELDS:
            raise ValueError(
                f"Group metadata {path} must have exactly the columns {list(GROUP_METADATA_FIELDS)}"
            )
        entries: list[AssetRow] = []
        seen: set[str] = set()
        for number, raw in enumerate(reader, start=2):
            values = {key: (raw.get(key) or "").strip() for key in GROUP_METADATA_FIELDS}
            try:
                locator = validate_locator(values["path"])
            except ValueError as exc:
                raise ValueError(f"Group metadata {path}, row {number}: {exc}") from None
            if not values["domain"] or not values["split"]:
                raise ValueError(f"Group metadata {path}, row {number}: domain and split required")
            if locator in seen:
                raise ValueError(f"Group metadata {path}, row {number}: duplicate path {locator!r}")
            seen.add(locator)
            entries.append(
                AssetRow(
                    root="dataset",
                    locator=locator,
                    role="input",
                    domain=values["domain"],
                    split=values["split"],
                    set_id=values["set_id"],
                    specimen_id=values["specimen_id"],
                    patient_id=values["patient_id"],
                )
            )
    return tuple(entries)


def enrich_with_groups(rows: Iterable[AssetRow], groups: Sequence[AssetRow]) -> list[AssetRow]:
    """Attach sidecar group IDs to rows by locator; a conflicting domain/split fails."""
    by_locator = {entry.locator: entry for entry in groups}
    enriched: list[AssetRow] = []
    for row in rows:
        entry = by_locator.get(row.locator)
        if entry is None:
            enriched.append(row)
            continue
        if entry.domain != row.domain or entry.split != row.split:
            raise ValueError(
                f"Group metadata declares {row.locator!r} as {entry.domain}/{entry.split}, "
                f"but it was selected as {row.domain}/{row.split}"
            )
        enriched.append(
            replace(
                row,
                set_id=entry.set_id,
                specimen_id=entry.specimen_id,
                patient_id=entry.patient_id,
            )
        )
    return enriched


def compare_generated(observed: Iterable[AssetRow], produced: DataSnapshot) -> dict[str, Any]:
    """Diff consumed generated files against a producer's recorded output snapshot."""
    seen = {
        row.locator: row for row in observed if row.role == "generated" and row.status == "present"
    }
    recorded = {row.locator: row for row in produced.rows if row.role == "generated"}
    changed = sorted(
        locator
        for locator in seen.keys() & recorded.keys()
        if seen[locator].size != recorded[locator].size
        or (
            seen[locator].sha256 is not None
            and recorded[locator].sha256 is not None
            and seen[locator].sha256 != recorded[locator].sha256
        )
    )
    verified = all(
        seen[locator].sha256 is not None and recorded[locator].sha256 is not None
        for locator in seen.keys() & recorded.keys()
    )
    return {
        "missing": sorted(recorded.keys() - seen.keys()),
        "extra": sorted(seen.keys() - recorded.keys()),
        "changed": changed,
        "content_verified": verified,
    }
