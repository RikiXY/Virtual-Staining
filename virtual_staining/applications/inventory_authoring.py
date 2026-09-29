"""Author the raw paired slide-set inventory from explicit asset mappings.

Every input, target, and mask mapping is named by the caller; nothing is inferred from
folder names, positions, dimensions, or image content, and no asset is ever opened. The
inventory schema stays owned by ``virtual_staining.data.slide_sets``: a written
inventory is published only after that canonical loader has read it back.
"""

from __future__ import annotations

import csv
import fnmatch
import io
import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

from virtual_staining.data.slide_sets import (
    MODALITY_NAME_PATTERN,
    SET_ID_PATTERN,
    SlideAsset,
    SlideSet,
    load_slide_set_inventory,
)
from virtual_staining.utils.files import publish_file_no_replace

KeyRule = Literal["relative-path", "relative-stem"]
KEY_RULES: tuple[KeyRule, ...] = ("relative-path", "relative-stem")
IssueKind = Literal["spec", "duplicate", "incomplete", "conflict", "set_id", "metadata", "mask"]
DEFAULT_OUTPUT = Path("inputs/slide_sets.csv")

NAME_LIMITATION = (
    "Sets were matched by file membership and names only; no image or slide was opened. "
    "Matching establishes no biological independence, patient identity, specimen "
    "identity, spatial correspondence, or registration validity."
)
REFERENCE_LIMITATION = (
    "Reference alignment=true means identity to the declared reference coordinate "
    "system only; it certifies no anatomical correspondence."
)
_GLOB_CHARS = frozenset("*?[")


@dataclass(frozen=True)
class InventoryRequest:
    """Explicit asset mappings for one paired inventory.

    ``inputs`` and ``input_masks`` are ordered ``(modality, spec)`` pairs. A spec is a
    ``dataset_root``-relative directory or glob; a relative ``metadata`` path is
    relative to ``dataset_root`` too.
    """

    dataset_root: Path
    inputs: tuple[tuple[str, str], ...]
    target_modality: str
    target: str
    reference: str
    input_masks: tuple[tuple[str, str], ...] = ()
    target_mask: str | None = None
    metadata: Path | None = None
    key_rule: KeyRule = "relative-path"

    def __post_init__(self) -> None:
        names = self.modalities
        if not names:
            raise ValueError("at least one input mapping is required")
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"duplicate input names: {duplicates}")
        invalid = [name for name in names if not MODALITY_NAME_PATTERN.fullmatch(name)]
        if invalid:
            raise ValueError(f"invalid input names: {invalid}")
        if self.reference not in names:
            raise ValueError(f"reference {self.reference!r} is not an input name")
        if not self.target_modality.strip():
            raise ValueError("target modality must not be blank")
        if self.target_modality in names:
            raise ValueError("target modality must differ from every input name")
        mask_names = [name for name, _ in self.input_masks]
        unknown = sorted(set(mask_names) - set(names))
        if unknown:
            raise ValueError(f"input masks name unknown inputs: {unknown}")
        if len(set(mask_names)) != len(mask_names):
            raise ValueError("at most one mask mapping per input")
        if self.key_rule not in KEY_RULES:
            raise ValueError(f"key rule must be one of: {', '.join(KEY_RULES)}")

    @property
    def modalities(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.inputs)


@dataclass(frozen=True)
class InventoryIssue:
    kind: IssueKind
    message: str
    key: str | None = None


@dataclass(frozen=True)
class InventoryMatch:
    key: str
    slide_set: SlideSet


@dataclass(frozen=True)
class InventoryPreview:
    """What ``write_inventory`` would publish; ``sources`` is the discovered membership."""

    request: InventoryRequest
    dataset_root: Path
    matches: tuple[InventoryMatch, ...]
    issues: tuple[InventoryIssue, ...]
    sources: tuple[str, ...]
    limitations: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return not self.issues and bool(self.matches)

    @property
    def matched_count(self) -> int:
        return len(self.matches)


def _key(relative: str, rule: KeyRule) -> str:
    path = PurePosixPath(relative)
    return relative if rule == "relative-path" else path.with_suffix("").as_posix()


def _derived_set_id(key: str, rule: KeyRule) -> str:
    """The key's file name without its final extension (already removed by relative-stem)."""
    name = PurePosixPath(key).name
    return name if rule == "relative-stem" else PurePosixPath(name).stem


def _glob_match(pattern: tuple[str, ...], parts: tuple[str, ...]) -> bool:
    if not pattern:
        return not parts
    if pattern[0] == "**":
        return any(_glob_match(pattern[1:], parts[index:]) for index in range(len(parts) + 1))
    return (
        bool(parts)
        and fnmatch.fnmatchcase(parts[0], pattern[0])
        and _glob_match(pattern[1:], parts[1:])
    )


def _walk(anchor: Path) -> Iterator[Path]:
    """Every entry below ``anchor`` in sorted order; symlinked directories are not entered."""
    for directory, dirnames, filenames in os.walk(anchor):
        dirnames.sort()
        for name in sorted([*dirnames, *filenames]):
            yield Path(directory) / name


def _discover(
    root: Path, label: str, spec: str, rule: KeyRule, *, required: bool
) -> tuple[dict[str, list[str]], list[InventoryIssue]]:
    """Map each key of one spec to its ``dataset_root``-relative POSIX paths."""
    issues: list[InventoryIssue] = []

    def issue(message: str) -> None:
        issues.append(InventoryIssue("spec", f"{label} {spec!r}: {message}"))

    pure = PurePosixPath(spec)
    if not spec.strip() or pure.is_absolute() or ".." in pure.parts:
        issue("must be a non-empty dataset_root-relative path without '..'")
        return {}, issues
    split = next(
        (index for index, part in enumerate(pure.parts) if _GLOB_CHARS & set(part)),
        len(pure.parts),
    )
    anchor_parts, pattern = pure.parts[:split], pure.parts[split:]
    anchor = root.joinpath(*anchor_parts)
    if any(root.joinpath(*anchor_parts[:end]).is_symlink() for end in range(1, split + 1)):
        issue("traverses a symlink")
        return {}, issues
    if not anchor.is_dir():
        issue(f"directory not found: {PurePosixPath(*anchor_parts or ('.',))}")
        return {}, issues

    found: dict[str, list[str]] = {}
    for entry in _walk(anchor):
        relative = entry.relative_to(anchor)
        matched = not pattern or _glob_match(pattern, relative.parts)
        shown = entry.relative_to(root).as_posix()
        if entry.is_symlink():
            if matched or entry.is_dir():
                issue(f"{shown}: symlinks are not followed")
        elif not matched:
            continue
        elif entry.is_dir():
            if pattern:
                issue(f"{shown}: matches a directory, not a file")
        elif not entry.is_file():
            issue(f"{shown}: not a regular file")
        else:
            found.setdefault(_key(relative.as_posix(), rule), []).append(shown)
    if required and not found and not issues:
        issue("matched no files")
    return found, issues


def _scan(
    root: Path,
    label: str,
    spec: str,
    rule: KeyRule,
    issues: list[InventoryIssue],
    sources: set[str],
    *,
    required: bool,
) -> dict[str, list[str]]:
    found, found_issues = _discover(root, label, spec, rule, required=required)
    issues.extend(found_issues)
    for key, paths in found.items():
        sources.update(paths)
        if len(paths) > 1:
            issues.append(InventoryIssue("duplicate", f"{label}: key {key!r} matches {paths}", key))
    return found


def _read_metadata(
    request: InventoryRequest, root: Path, keys: set[str]
) -> tuple[dict[str, dict[str, str]], list[InventoryIssue]]:
    """Metadata rows by key; only existing inventory semantics are accepted."""
    issues: list[InventoryIssue] = []
    if request.metadata is None:
        return {}, issues
    path = root / request.metadata  # an absolute metadata path stays absolute

    def issue(message: str, key: str | None = None) -> None:
        issues.append(InventoryIssue("metadata", f"metadata {message}", key))

    if not path.is_file():
        issue(f"file not found: {path}")
        return {}, issues
    names = request.modalities
    allowed = {"key", "set_id", "patient_id", "specimen_id", "target_aligned", "target_slide_id"}
    allowed.update(f"input__{name}_{field}" for name in names for field in ("aligned", "slide_id"))
    rows: dict[str, dict[str, str]] = {}
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fields = list(reader.fieldnames or ())
            if "key" not in fields:
                issue("requires a 'key' column")
                return {}, issues
            for field in fields:
                if field in allowed:
                    continue
                if field.endswith("_mask"):
                    issue(f"column {field!r} is not allowed; masks come from mask mappings")
                elif field.startswith("input__"):
                    issue(f"column {field!r} names no input modality of this request")
                else:
                    issue(f"column {field!r} is unknown")
            for number, row in enumerate(reader, start=2):
                clean = {
                    field: (row.get(field) or "").strip() for field in fields if field in allowed
                }
                key = clean.pop("key")
                if key in rows:
                    issue(f"row {number}: duplicate key {key!r}", key)
                    continue
                if key not in keys:
                    issue(f"row {number}: key {key!r} matches no discovered asset", key)
                for field, value in clean.items():
                    if field.endswith("_aligned"):
                        clean[field] = value = value.lower()
                        if value not in {"", "true", "false"}:
                            issue(f"row {number}: {field} must be true, false, or blank", key)
                if clean.get(f"input__{request.reference}_aligned") == "false":
                    issue(f"row {number}: reference input {request.reference} must be aligned", key)
                rows[key] = clean
    except (UnicodeDecodeError, csv.Error) as exc:
        issue(f"is not a readable CSV: {exc}")
    return rows, issues


def _aligned(value: str | None) -> bool | None:
    return None if not value else value == "true"


def preview_inventory(request: InventoryRequest) -> InventoryPreview:
    """Discover and match every mapping without writing or opening any file content.

    Every discrepancy is collected; the preview is ``valid`` only when there are none.
    """
    root = request.dataset_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"dataset_root is not a directory: {root}")
    rule = request.key_rule
    issues: list[InventoryIssue] = []
    sources: set[str] = set()
    target = request.target_modality
    found = {
        name: _scan(root, f"input {name}", spec, rule, issues, sources, required=True)
        for name, spec in request.inputs
    }
    found[target] = _scan(
        root, f"target {target}", request.target, rule, issues, sources, required=True
    )

    all_keys = sorted(set().union(*found.values()))
    complete: list[str] = []
    for key in all_keys:
        missing = [label for label, assets in found.items() if key not in assets]
        if missing:
            present = [label for label in found if label not in missing]
            issues.append(
                InventoryIssue(
                    "incomplete",
                    f"key {key!r}: no {', '.join(missing)} asset (found in {', '.join(present)})",
                    key,
                )
            )
        elif all(len(assets[key]) == 1 for assets in found.values()):
            if found[target][key][0] in {found[name][key][0] for name in request.modalities}:
                issues.append(
                    InventoryIssue(
                        "conflict", f"key {key!r}: target is the same file as an input", key
                    )
                )
            else:
                complete.append(key)

    metadata, metadata_issues = _read_metadata(request, root, set(all_keys))
    issues.extend(metadata_issues)

    masks: dict[str, dict[str, list[str]]] = {}
    mask_specs = [
        (f"input__{name}", f"input mask {name}", spec) for name, spec in request.input_masks
    ]
    if request.target_mask is not None:
        mask_specs.append(("target", "target mask", request.target_mask))
    for prefix, label, spec in mask_specs:
        masks[prefix] = _scan(root, label, spec, rule, issues, sources, required=False)
        issues.extend(
            InventoryIssue("mask", f"{label}: key {key!r} corresponds to no matched set", key)
            for key in sorted(set(masks[prefix]) - set(complete))
        )

    def asset(prefix: str, modality: str, key: str, row: dict[str, str]) -> SlideAsset:
        mask = masks.get(prefix, {}).get(key)
        aligned = _aligned(row.get(f"{prefix}_aligned"))
        return SlideAsset(
            modality=modality,
            path=Path(found[modality][key][0]),
            already_aligned=True if modality == request.reference and aligned is None else aligned,
            mask_path=Path(mask[0]) if mask else None,
            slide_id=row.get(f"{prefix}_slide_id") or None,
        )

    matches: list[InventoryMatch] = []
    for key in complete:
        row = metadata.get(key, {})
        set_id = row.get("set_id") or _derived_set_id(key, rule)
        if not SET_ID_PATTERN.fullmatch(set_id):
            issues.append(InventoryIssue("set_id", f"key {key!r}: unsafe set_id {set_id!r}", key))
        slide_set = SlideSet(
            set_id=set_id,
            inputs=tuple(asset(f"input__{name}", name, key, row) for name in request.modalities),
            target=asset("target", target, key, row),
            reference_modality=request.reference,
            patient_id=row.get("patient_id") or None,
            specimen_id=row.get("specimen_id") or None,
        )
        matches.append(InventoryMatch(key, slide_set))
    by_id: dict[str, list[str]] = {}
    for match in matches:
        by_id.setdefault(match.slide_set.set_id, []).append(match.key)
    issues.extend(
        InventoryIssue("set_id", f"set_id {set_id!r} is shared by keys {keys}")
        for set_id, keys in sorted(by_id.items())
        if len(keys) > 1
    )
    return InventoryPreview(
        request=request,
        dataset_root=root,
        matches=tuple(matches),
        issues=tuple(issues),
        sources=tuple(sorted(sources)),
        limitations=_limitations(request, matches),
    )


def _limitations(request: InventoryRequest, matches: list[InventoryMatch]) -> tuple[str, ...]:
    sets = [match.slide_set for match in matches]
    limitations = [NAME_LIMITATION, REFERENCE_LIMITATION]
    unknown = sum(
        any(item.already_aligned is None for item in (*slide_set.inputs, slide_set.target))
        for slide_set in sets
    )
    if unknown:
        limitations.append(
            f"{unknown} set(s) leave non-reference or target alignment blank (unknown); "
            "nothing was inferred from names, keys, directories, or dimensions."
        )
    absent = [
        field for field in ("patient_id", "specimen_id") if not any(getattr(s, field) for s in sets)
    ]
    if absent:
        limitations.append(
            f"No {' or '.join(absent)} supplied; group identity is unknown and was not inferred."
        )
    if request.metadata is None:
        limitations.append("No metadata CSV supplied.")
    return tuple(limitations)


_OPTIONAL_TAIL = ("target_mask", "target_slide_id", "patient_id", "specimen_id")


def render_inventory_csv(preview: InventoryPreview) -> str:
    """The canonical wide CSV for ``preview``: rows by ``set_id``, deterministic columns."""
    names = preview.request.modalities
    required = [
        "set_id",
        *(f"input__{name}_{field}" for name in names for field in ("path", "aligned")),
        "target_path",
        "target_aligned",
    ]
    optional = [f"input__{name}_{field}" for name in names for field in ("mask", "slide_id")]
    rows = [
        _csv_row(match.slide_set)
        for match in sorted(preview.matches, key=lambda item: item.slide_set.set_id)
    ]
    columns = required + [
        column for column in (*optional, *_OPTIONAL_TAIL) if any(row[column] for row in rows)
    ]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, columns, lineterminator="\n", extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _csv_row(slide_set: SlideSet) -> dict[str, str]:
    row = {
        "set_id": slide_set.set_id,
        "patient_id": slide_set.patient_id or "",
        "specimen_id": slide_set.specimen_id or "",
    }
    assets = [(f"input__{item.modality}", item) for item in slide_set.inputs]
    for prefix, item in [*assets, ("target", slide_set.target)]:
        aligned = item.already_aligned
        row[f"{prefix}_path"] = item.path.as_posix()
        row[f"{prefix}_aligned"] = "" if aligned is None else str(aligned).lower()
        row[f"{prefix}_mask"] = item.mask_path.as_posix() if item.mask_path else ""
        row[f"{prefix}_slide_id"] = item.slide_id or ""
    return row


def _destination(preview: InventoryPreview, output: Path | str | None) -> Path:
    root = preview.dataset_root
    path = Path(output) if output is not None else DEFAULT_OUTPUT
    if path.is_absolute():
        path = Path(os.path.abspath(path))
        for base in (root, preview.request.dataset_root.absolute()):
            if path.is_relative_to(base):
                path = path.relative_to(base)
                break
        else:
            raise ValueError(f"output must be inside dataset_root {root}: {output}")
    if not path.parts or ".." in path.parts:
        raise ValueError(f"output must be a file path inside dataset_root: {output}")
    if any(root.joinpath(*path.parts[:end]).is_symlink() for end in range(1, len(path.parts) + 1)):
        raise ValueError(f"output must not traverse a symlink: {output}")
    return root / path


def write_inventory(preview: InventoryPreview, output: Path | str | None = None) -> Path:
    """Publish ``preview`` as the canonical inventory; never replaces an existing path.

    Discovery is rerun from the request and must reproduce the preview exactly; the
    rendered CSV is written to a sibling temporary file and published only after the
    canonical loader resolves it to the previewed slide sets. ``output`` defaults to
    ``inputs/slide_sets.csv``; a relative ``output`` is relative to ``dataset_root``.
    """
    if not preview.valid:
        raise ValueError("refusing to write an invalid inventory preview")
    destination = _destination(preview, output)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing {destination}")
    request = preview.request
    current = preview_inventory(request)
    if (current.matches, current.sources, current.issues) != (
        preview.matches,
        preview.sources,
        preview.issues,
    ):
        raise ValueError("source assets or metadata changed since the preview; preview again")
    expected = tuple(sorted((m.slide_set for m in current.matches), key=lambda s: s.set_id))

    def verify(temporary: Path) -> None:
        loaded = load_slide_set_inventory(
            temporary,
            current.dataset_root,
            modalities=request.modalities,
            reference_modality=request.reference,
            target_modality=request.target_modality,
        )
        if loaded != expected:
            raise ValueError("canonical loader did not reproduce the previewed slide sets")

    destination.parent.mkdir(parents=True, exist_ok=True)
    return publish_file_no_replace(
        render_inventory_csv(current).encode("utf-8"), destination, verify=verify
    )
