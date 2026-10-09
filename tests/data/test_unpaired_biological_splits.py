"""Audit published partitions against raw identities and pixels, without the split validator.

All identities and pixels are test fixtures based on the example template. The independent oracle
uses CSV joins, explicit patch coordinates and a graph walk of declared ancestry.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from PIL import Image

from tests.config_helpers import cyclegan_config_data, write_config_data
from virtual_staining import cli
from virtual_staining.applications.prepare import prepare
from virtual_staining.applications.train import _unpaired_datasets
from virtual_staining.config.run import RunConfig
from virtual_staining.data.builder import DatasetBuildResult
from virtual_staining.data.unpaired import UnpairedImageDataset, resolve_domain_collections

SPLITS = ("train", "val", "test")
IDENTITIES = ("patient_id", "specimen_id", "set_id")
UNITS = ("patient", "specimen", "set")
EXAMPLES = Path(__file__).resolve().parents[2] / "examples/unpaired"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def example(tmp_path: Path) -> Path:
    root = tmp_path / "synthetic"
    root.mkdir()
    rows = read_csv(EXAMPLES / "paths_grouped.csv")
    rng = np.random.default_rng(2026)
    for row in rows:
        width = 32 if row["domain"] == "LF" else 48
        path = root / row["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rng.integers(20, 210, (32, width, 3), dtype=np.uint8)).save(path)
    write_csv(root / "paths.csv", rows)
    for unit in (*UNITS, "patch"):
        write_config_data(root / f"prepare_{unit}.yaml", config_data(root, unit))
    return root


def config_data(root: Path, unit: str) -> dict[str, Any]:
    raw = yaml.safe_load((EXAMPLES / "prepare_grouped.yaml").read_text())
    raw["dataset_root"] = str(root)
    raw["data"]["group_validation"] = "unavailable" if unit == "patch" else "patient"
    raw["preprocessing"]["inputs"]["inventory"] = "paths.csv"
    raw["preprocessing"]["patching"].update(patch_size=[16, 16], grid_movement=[16, 16])
    raw["preprocessing"]["io"] = {"tiled": False, "backend": "pillow"}
    raw["preprocessing"]["split"]["unit"] = unit
    return raw


def build(root: Path, raw: dict[str, Any]) -> DatasetBuildResult:
    path = write_config_data(root / "selected.yaml", raw)
    return prepare(RunConfig.from_yaml(path, stages=("prepare",)), path)


def assignment_rows(root: Path, unit: str) -> list[dict[str, str]]:
    # Freeze an explicit test-side patient partition, not the algorithm's own result.
    patient_split = {
        f"EXAMPLE_P{i:03}": "test" if i == 1 else "val" if i == 2 else "train" for i in range(1, 7)
    }
    groups = {
        row[f"{unit}_id"]: patient_split[row["patient_id"]] for row in read_csv(root / "paths.csv")
    }
    return [
        {"group_id": group, "unit": unit, "split": split} for group, split in sorted(groups.items())
    ]


def audit_published(
    raw_root: Path,
    result: DatasetBuildResult,
    unit: str,
    validation: str = "patient",
) -> dict[tuple[str, int, int], str]:
    """Join every physical output to source identity, location, pixels and assignment."""
    root = result.output_root
    inventory = read_csv(raw_root / "paths.csv")
    sources = {row["path"]: row for row in inventory}
    assert len(sources) == len(inventory)
    evidence = json.loads((root / "metadata/images.json").read_text())
    assert len(evidence) == len(sources)
    assert {item["source"]["path"] for item in evidence} == set(sources)
    expected_origins = set()
    for path in sources:
        with Image.open(raw_root / path) as image:
            expected_origins.update(
                (path, x, y)
                for x in range(0, image.width - 15, 16)
                for y in range(0, image.height - 15, 16)
            )
    membership = {}
    expected_sidecar = {}
    sample_splits = {}
    for item in evidence:
        source = sources[item["source"]["path"]]
        assert not item["excluded"]
        for field in ("domain", *IDENTITIES):
            assert item["source"][field] == source.get(field, "")
        with Image.open(raw_root / source["path"]) as image:
            pixels = np.array(image.convert("RGB"))
        assert (item["geometry"]["height"], item["geometry"]["width"]) == pixels.shape[:2]
        for patch in item["accepted"]:
            key = source["path"], patch["x"], patch["y"]
            assert key not in membership
            membership[key] = patch["split"]
            locator = patch["path"]
            assert locator not in expected_sidecar
            assert Path(locator).parts[:3] == ("splits", patch["split"], source["domain"])
            assert patch["source"] == source["path"] and patch["domain"] == source["domain"]
            assert patch["width"] == patch["height"] == 16
            with Image.open(root / locator) as image:
                np.testing.assert_array_equal(
                    np.array(image),
                    pixels[patch["y"] : patch["y"] + 16, patch["x"] : patch["x"] + 16],
                )
            for field in IDENTITIES:
                assert patch[field] == source.get(field, "")
            expected_sidecar[locator] = {
                "path": locator,
                "domain": source["domain"],
                "split": patch["split"],
                **{field: source.get(field, "") for field in IDENTITIES},
            }
            assert patch["sample_id"] not in sample_splits
            sample_splits[patch["sample_id"]] = patch["split"]
    assert set(membership) == expected_origins
    assert len(expected_origins) == result.train_count + result.val_count + result.test_count
    assert result.skipped_count == 0
    actual_files = {path.relative_to(root).as_posix() for path in (root / "splits").rglob("*.png")}
    assert actual_files == set(expected_sidecar)
    sidecar = read_csv(root / "metadata/groups.csv")
    assert len(sidecar) == len(expected_sidecar)
    assert {row["path"]: row for row in sidecar} == expected_sidecar
    published = read_csv(root / "metadata/prepared_data/rows.csv")
    assert len(published) == len(sidecar)
    assert {
        row["locator"]: {
            "path": row["locator"],
            **{field: row[field] for field in ("domain", "split", *IDENTITIES)},
        }
        for row in published
    } == expected_sidecar
    assert {row["role"] for row in published} == {"input"}
    metadata = json.loads((root / "metadata/prepared_data/snapshot.json").read_text())
    group_result = metadata["group_validation"]
    assert group_result["requested"] == validation
    assert group_result["unit"] == (None if validation == "unavailable" else validation)
    assert group_result["status"] == ("unavailable" if validation == "unavailable" else "validated")
    assert not (root / "manifests/manifest.csv").exists()
    for split in SPLITS:
        for domain in ("LF", "HE"):
            assert any(row["split"] == split and row["domain"] == domain for row in sidecar)
    assignments = read_csv(root / "metadata/split_assignment.csv")
    mapping = {row["group_id"]: row["split"] for row in assignments}
    assert len(mapping) == len(assignments)
    assert {row["unit"] for row in assignments} == {unit}
    if unit == "patch":
        assert mapping == sample_splits
    else:
        assert set(mapping) == {row[f"{unit}_id"] for row in inventory}
        for (path, _, _), split in membership.items():
            assert mapping[sources[path][f"{unit}_id"]] == split
        for field in IDENTITIES:
            group_splits: dict[str, set[str]] = defaultdict(set)
            for row in sidecar:
                if row[field]:
                    group_splits[row[field]].add(row["split"])
            assert all(len(splits) == 1 for splits in group_splits.values())
        for path in sources:
            assert (
                len({split for (source, _, _), split in membership.items() if source == path}) == 1
            )
    # Verify the consumer resolves precisely these images with unchanged identity evidence.
    collections, consumed_rows = resolve_domain_collections(
        result.domain_collections,
        root,
        splits=("train", "val", "test"),
        roles={"LF": "input", "HE": "target"},
        group_metadata=Path("metadata/groups.csv"),
    )
    assert {
        row.locator: {
            "path": row.locator,
            "domain": row.domain,
            "split": row.split,
            **{field: getattr(row, field) for field in IDENTITIES},
        }
        for row in consumed_rows
    } == expected_sidecar
    for split in SPLITS:
        left, right = collections[split, "LF"], collections[split, "HE"]
        dataset = UnpairedImageDataset(left, right, transform=np.asarray)
        assert len(dataset) == max(len(left), len(right))
        assert dataset[0]["path_a"] == str(left[0]) and dataset[0]["path_b"] == str(right[0])
        assert dataset[len(dataset) - 1]["path_a"] == str(left[(len(dataset) - 1) % len(left)])
    return membership


def ancestry_components(rows: list[dict[str, str]]) -> list[set[str]]:
    """Independent graph traversal: image -> set -> specimen -> patient, missing IDs omitted."""
    graph: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)
    for row in rows:
        chain = [("image", row["path"])] + [
            (field, row[field]) for field in reversed(IDENTITIES) if row.get(field)
        ]
        for left, right in pairwise(chain):
            graph[left].add(right)
            graph[right].add(left)
    remaining = set(graph)
    components = []
    while remaining:
        pending = [next(iter(remaining))]
        visited = set(pending)
        while pending:
            for neighbor in graph[pending.pop()] - visited:
                visited.add(neighbor)
                pending.append(neighbor)
        remaining -= visited
        components.append({value for field, value in visited if field == "image"})
    return components


@pytest.mark.parametrize("unit", UNITS)
def test_cli_nested_and_transitive_published_membership(example: Path, unit: str) -> None:
    path = example / f"prepare_{unit}.yaml"
    config = RunConfig.from_yaml(path, stages=("prepare",))
    assert config.model is config.method is config.training is None
    assert config.project.results_path is config.project.run_name is None
    cli.main(["config", "check", "--config", str(path), "--stages", "prepare", "--assets"])
    cli.main(["prepare", "--config", str(path)])
    result = prepare(config, path)
    assert result.reused
    membership = audit_published(example, result, unit)
    rows = read_csv(example / "paths.csv")
    assert Counter(row["domain"] for row in rows) == {"LF": 24, "HE": 15}
    assert [len({row[field] for row in rows}) for field in IDENTITIES] == [6, 12, 24]
    components = ancestry_components(rows)
    assert len(components) == 6
    # Ancestry paths cross different specimens and sets, not just duplicate group rows.
    for paths in components:
        members = [row for row in rows if row["path"] in paths]
        assert len({row["specimen_id"] for row in members}) == 2
        assert len({row["set_id"] for row in members}) == 4
        assert {row["domain"] for row in members} == {"LF", "HE"}
        assert len({split for (path, _, _), split in membership.items() if path in paths}) == 1
    assert Counter(
        next(iter({split for (path, _, _), split in membership.items() if path in paths}))
        for paths in components
    ) == {"train": 4, "val": 1, "test": 1}
    assert not (example / "results").exists()


@pytest.mark.parametrize("unit", (*UNITS, "patch"))
def test_actual_cyclegan_adapter_retains_domains_and_group_claim(example: Path, unit: str) -> None:
    result = build(example, config_data(example, unit))
    validation = "unavailable" if unit == "patch" else "patient"
    audit_published(example, result, unit, validation)
    raw = cyclegan_config_data(example)
    raw["dataset_root"] = str(result.output_root)
    raw["model"].update(inputs=["LF"], outputs=["HE"])
    raw["data"] = {
        "pairing": "unpaired",
        "domains": result.domain_collections,
        "group_metadata": "metadata/groups.csv",
        "group_validation": validation,
    }
    train, val, snapshot = _unpaired_datasets(RunConfig.from_mapping(raw), np.asarray, 42)
    expected = read_csv(result.output_root / "metadata/groups.csv")
    assert set(train.paths_a) == {
        result.output_root / row["path"]
        for row in expected
        if row["split"] == "train" and row["domain"] == "LF"
    }
    assert set(train.paths_b) == {
        result.output_root / row["path"]
        for row in expected
        if row["split"] == "train" and row["domain"] == "HE"
    }
    assert len(val) and snapshot.group_validation["requested"] == validation
    assert snapshot.group_validation["unit"] == (None if unit == "patch" else "patient")
    assert {(row.domain, row.role) for row in snapshot.rows} == {("LF", "input"), ("HE", "target")}
    before = [train.domain_b_index(i) for i in range(len(train))]
    train.set_epoch(1)
    assert [train.domain_b_index(i) for i in range(len(train))] != before
    if unit == "patch":
        assert snapshot.group_validation["status"] == "unavailable"
        assert snapshot.group_validation["split_unit"] == "patch"
        groups: dict[str, set[str]] = defaultdict(set)
        for row in expected:
            groups[row["patient_id"]].add(row["split"])
        assert any(len(splits) > 1 for splits in groups.values())
        assert snapshot.group_validation["groups_shared_across_splits"]["patient"] == sum(
            len(splits) > 1 for splits in groups.values()
        )
        assert any("not biologically independent" in text for text in snapshot.limitations)
        produced = json.loads(
            (result.output_root / "metadata/prepared_data/snapshot.json").read_text()
        )
        assert produced["group_validation"]["unit"] is None
        assert any("not biologically independent" in text for text in produced["limitations"])
        evidence = json.loads((result.output_root / "metadata/images.json").read_text())
        assert any(len({patch["split"] for patch in item["accepted"]}) > 1 for item in evidence)


@pytest.mark.parametrize("unit", (*UNITS, "patch"))
def test_determinism_csv_and_file_order_seed_and_reuse(example: Path, unit: str) -> None:
    raw = config_data(example, unit)
    validation = "unavailable" if unit == "patch" else "patient"
    first = build(example, raw)
    expected = audit_published(example, first, unit, validation)
    assert build(example, raw).reused
    assignment_bytes = (first.output_root / "metadata/split_assignment.csv").read_bytes()
    rows = read_csv(example / "paths.csv")
    write_csv(example / "paths.csv", list(reversed(rows)))
    # Recreate only these test-owned images in reverse order; bytes and locators are identical.
    payloads = [(example / row["path"], (example / row["path"]).read_bytes()) for row in rows]
    for path, _ in payloads:
        path.unlink()
    for path, payload in reversed(payloads):
        path.write_bytes(payload)
    reordered = build(example, raw)
    assert audit_published(example, reordered, unit, validation) == expected
    assert (
        reordered.output_root / "metadata/split_assignment.csv"
    ).read_bytes() == assignment_bytes
    alternatives = []
    for seed in (0, 17):
        raw["preprocessing"]["split"]["seed"] = seed
        alternatives.append(audit_published(example, build(example, raw), unit, validation))
    assert any(other != expected for other in alternatives)


@pytest.mark.parametrize("unit", UNITS)
def test_frozen_mapping_is_independent_of_seed_and_changes_invalidate_reuse(
    example: Path, unit: str
) -> None:
    frozen = assignment_rows(example, unit)
    write_csv(example / "frozen.csv", list(reversed(frozen)))
    raw = config_data(example, unit)
    raw["preprocessing"]["split"]["assignment_file"] = "frozen.csv"
    original = None
    for seed in (42, 999):
        raw["preprocessing"]["split"]["seed"] = seed
        result = build(example, raw)
        observed = audit_published(example, result, unit)
        assert read_csv(result.output_root / "metadata/split_assignment.csv") == frozen
        if original is not None:
            assert observed == original
        original = observed
    for row in frozen:
        if row["split"] in {"test", "val"}:
            row["split"] = "test" if row["split"] == "val" else "val"
    write_csv(example / "frozen.csv", frozen)
    changed = build(example, raw)
    assert not changed.reused
    assert audit_published(example, changed, unit) != original
    assert read_csv(changed.output_root / "metadata/split_assignment.csv") == frozen


@pytest.mark.parametrize("unit", UNITS)
@pytest.mark.parametrize(
    "issue",
    [
        "missing",
        "unexpected",
        "duplicate",
        "wrong_unit",
        "bad_split",
        "renamed",
        "extra_cell",
        "short_row",
    ],
)
def test_invalid_frozen_assignment_never_publishes(example: Path, unit: str, issue: str) -> None:
    rows = assignment_rows(example, unit)
    if issue == "missing":
        rows.pop()
    elif issue == "unexpected":
        rows.append({"group_id": "UNKNOWN", "unit": unit, "split": "test"})
    elif issue == "duplicate":
        rows.append(dict(rows[0]))
    elif issue == "wrong_unit":
        rows[0]["unit"] = "patient" if unit != "patient" else "specimen"
    elif issue == "bad_split":
        rows[0]["split"] = "validation"
    elif issue == "renamed":
        rows[0]["group_id"] = "UNKNOWN"
    write_csv(example / "frozen.csv", rows)
    if issue in {"extra_cell", "short_row"}:
        lines = (example / "frozen.csv").read_text().splitlines()
        lines[1] = lines[1] + ",train" if issue == "extra_cell" else lines[1].rsplit(",", 1)[0]
        (example / "frozen.csv").write_text("\n".join(lines) + "\n")
    raw = config_data(example, unit)
    raw["preprocessing"]["split"]["assignment_file"] = "frozen.csv"
    with pytest.raises(ValueError, match="Frozen split assignment"):
        build(example, raw)
    assert not (example / "prepared_unpaired").exists()


@pytest.mark.parametrize("unit", ["specimen", "set"])
@pytest.mark.parametrize("validation", ["patient", "unavailable"])
def test_frozen_shared_parent_leakage_cannot_be_waived(
    example: Path, unit: str, validation: str
) -> None:
    frozen = assignment_rows(example, unit)
    frozen[0]["split"] = "train"  # One child of test patient P001; siblings remain in test.
    write_csv(example / "frozen.csv", frozen)
    raw = config_data(example, unit)
    raw["data"]["group_validation"] = validation
    raw["preprocessing"]["split"]["assignment_file"] = "frozen.csv"
    with pytest.raises(ValueError, match="patient_id values appear in more than one split"):
        build(example, raw)
    assert not (example / "prepared_unpaired").exists()


def test_frozen_specimen_leakage_without_patient_evidence(example: Path) -> None:
    frozen = assignment_rows(example, "set")
    frozen[0]["split"] = "train"  # A1 and A2 still share specimen A, even without patient IDs.
    write_csv(example / "frozen.csv", frozen)
    rows = read_csv(example / "paths.csv")
    for row in rows:
        row.pop("patient_id")
    write_csv(example / "paths.csv", rows)
    raw = config_data(example, "set")
    raw["data"]["group_validation"] = "specimen"
    raw["preprocessing"]["split"]["assignment_file"] = "frozen.csv"
    with pytest.raises(ValueError, match="specimen_id values appear in more than one split"):
        build(example, raw)
    assert not (example / "prepared_unpaired").exists()


@pytest.mark.parametrize("unit", UNITS)
def test_partial_patient_evidence_cannot_claim_patient_validation(example: Path, unit: str) -> None:
    rows = read_csv(example / "paths.csv")
    for row in rows:
        if row["specimen_id"] == "EXAMPLE_SP001_B":
            row["patient_id"] = ""
    write_csv(example / "paths.csv", rows)
    with pytest.raises(ValueError, match="patient_id|requires explicit identifiers"):
        build(example, config_data(example, unit))
    assert not (example / "prepared_unpaired").exists()


@pytest.mark.parametrize(
    "missing,unit,validation,valid",
    [
        ("patient_id", "patient", "unavailable", False),
        ("patient_id", "specimen", "patient", False),
        ("patient_id", "set", "patient", False),
        ("patient_id", "specimen", "specimen", True),
        ("patient_id", "set", "set", True),
        ("specimen_id", "specimen", "patient", False),
        ("specimen_id", "patient", "patient", True),
        ("set_id", "set", "patient", False),
        ("set_id", "patient", "patient", True),
    ],
)
def test_missing_metadata_conditional_requirements(
    example: Path, missing: str, unit: str, validation: str, valid: bool
) -> None:
    rows = read_csv(example / "paths.csv")
    for row in rows:
        row.pop(missing)
    write_csv(example / "paths.csv", rows)
    raw = config_data(example, unit)
    raw["data"]["group_validation"] = validation
    if valid:
        audit_published(example, build(example, raw), unit, validation)
    else:
        with pytest.raises(ValueError, match=f"{missing}|requires explicit identifiers"):
            build(example, raw)
        assert not (example / "prepared_unpaired").exists()


@pytest.mark.parametrize("field", ["patient_id", "specimen_id"])
@pytest.mark.parametrize("replacement", ["CONTRADICTORY", ""])
def test_shared_child_rejects_conflicting_or_incomplete_parent(
    example: Path, field: str, replacement: str
) -> None:
    rows = read_csv(example / "paths.csv")
    rows[1][field] = replacement  # Same set and specimen as rows[0].
    write_csv(example / "paths.csv", rows)
    with pytest.raises(ValueError, match="conflicting or incomplete"):
        build(example, config_data(example, "set"))
    assert not (example / "prepared_unpaired").exists()


@pytest.mark.parametrize("replacement", ["CONTRADICTORY", ""])
def test_different_sets_cannot_give_one_specimen_incompatible_patients(
    example: Path, replacement: str
) -> None:
    rows = read_csv(example / "paths.csv")
    for row in rows:
        if row["set_id"] == "EXAMPLE_S001_A2":
            row["patient_id"] = replacement
    write_csv(example / "paths.csv", rows)
    # Each set remains internally consistent; only the shared specimen exposes the conflict.
    with pytest.raises(ValueError, match="conflicting or incomplete patient_id for specimen_id"):
        build(example, config_data(example, "set"))
    assert not (example / "prepared_unpaired").exists()


@pytest.mark.parametrize("validation", ["auto", "patient", "specimen", "set"])
def test_patch_exception_is_explicit(example: Path, validation: str) -> None:
    raw = config_data(example, "patch")
    raw["data"]["group_validation"] = validation
    with pytest.raises(ValueError, match="requires data.group_validation='unavailable'"):
        build(example, raw)
    assert not (example / "prepared_unpaired").exists()
