"""Synthetic independent domains: software contracts, no correspondence claims."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from tests.config_helpers import write_config_data
from virtual_staining import cli
from virtual_staining.applications.prepare import prepare
from virtual_staining.config.run import RunConfig
from virtual_staining.data.consumption import build_snapshot
from virtual_staining.data.layout import DatasetLayout
from virtual_staining.data.unpaired import (
    UnpairedImageDataset,
    prepared_unpaired_patch_split,
    resolve_domain_collections,
)
from virtual_staining.data.unpaired_inventory import load_unpaired_inventory, unpaired_assignments


def raw_config(root: Path) -> dict[str, Any]:
    return {
        "dataset_root": str(root),
        "data": {"pairing": "unpaired", "hash_policy": "content", "group_validation": "patient"},
        "preprocessing": {
            "inputs": {"inventory": "paths.csv", "domains": ["LF", "HE"]},
            "split": {"unit": "patient", "train": 0.5, "val": 0.25, "test": 0.25, "seed": 42},
            "patching": {"patch_size": [8, 8], "grid_movement": [8, 8], "margin": 0},
            "masks": {"generation": "never"},
            "filtering": {
                "foreground": {"enabled": False},
                "max_white_ratio": 1,
                "max_largest_white_component_ratio": 1,
            },
        },
    }


def inventory(root: Path) -> list[dict[str, str]]:
    rows = []
    rng = np.random.default_rng(10)
    # One more source in HE; dimensions and patch counts differ by domain/source.
    for domain, count, size in (("LF", 4, (16, 24)), ("HE", 5, (24, 32))):
        for index in range(count):
            suffix = (".png", ".tif", ".bmp", ".jpg")[index % 4]
            path = f"raw/{domain}-{index}{suffix}"
            (root / path).parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rng.integers(10, 180, (*size, 3), dtype=np.uint8)).save(root / path)
            group = index % 4
            rows.append(
                {
                    "domain": domain,
                    "path": path,
                    "set_id": f"set{group}",
                    "specimen_id": f"specimen{group}",
                    "patient_id": f"patient{group}",
                }
            )
    write_inventory(root, rows)
    return rows


def write_inventory(root: Path, rows: list[dict[str, str]]) -> None:
    with (root / "paths.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_prepare(root: Path, raw: dict[str, Any] | None = None):
    raw = raw or raw_config(root)
    path = write_config_data(root / "prepare.yaml", raw)
    return prepare(RunConfig.from_mapping(raw, stages=("prepare",)), path)


def test_minimal_and_strict_config(tmp_path: Path) -> None:
    raw = raw_config(tmp_path)
    for key in ("patching", "masks", "filtering"):
        raw["preprocessing"].pop(key)
    config = RunConfig.from_mapping(raw, stages=("prepare",))
    assert config.model is config.method is config.training is None
    assert config.project.results_path is config.project.run_name is None
    assert RunConfig.from_mapping(config.to_dict(), stages=("prepare",)) == config
    raw["preprocessing"]["io"] = {"unknown": True}
    with pytest.raises(ValueError, match="Unknown"):
        RunConfig.from_mapping(raw)


@pytest.mark.parametrize(
    "domains",
    [["LF"], ["LF", "HE", "IHC"], ["LF", "LF"], ["LF", "../HE"], ["LF", " HE"], ["LF", 1]],
)
def test_invalid_domains(tmp_path: Path, domains: list[Any]) -> None:
    raw = raw_config(tmp_path)
    raw["preprocessing"]["inputs"]["domains"] = domains
    with pytest.raises((ValueError, TypeError)):
        RunConfig.from_mapping(raw, stages=("prepare",))


def test_cli_end_to_end_and_reuse(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rows = inventory(tmp_path)
    # Author raw collections, then consume only the plain CSV through normal prepare.
    for row in rows:
        old = tmp_path / row["path"]
        row["path"] = f"raw/{row['domain']}/{old.name}"
        destination = tmp_path / row["path"]
        destination.parent.mkdir(exist_ok=True)
        old.rename(destination)
    (tmp_path / "paths.csv").unlink()
    with (tmp_path / "meta.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["path", "set_id", "specimen_id", "patient_id"],
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)
    args = [
        "--dataset-root",
        str(tmp_path),
        "--pairing",
        "unpaired",
        "--domain",
        "LF=raw/LF",
        "--domain",
        "HE=raw/HE",
        "--metadata",
        "meta.csv",
    ]
    cli.main(["inventory", "preview", *args])
    assert not (tmp_path / "inputs").exists()
    cli.main(["inventory", "write", *args])
    raw = raw_config(tmp_path)
    raw["preprocessing"]["inputs"]["inventory"] = "inputs/paths.csv"
    path = write_config_data(tmp_path / "prepare.yaml", raw)
    monkeypatch.setattr(
        "virtual_staining.data.slide_set_processor.SlideSetProcessor.process",
        lambda self: pytest.fail("paired processor called"),
    )
    cli.main(["config", "check", "--config", str(path), "--stages", "prepare", "--assets"])
    cli.main(["config", "resolve", "--config", str(path), "--stages", "prepare"])
    assert not (tmp_path / "prepared_unpaired").exists()
    cli.main(["prepare", "--config", str(path)])
    result = run_prepare(tmp_path, raw)
    assert result.reused
    layout = DatasetLayout(result.output_root)
    assert not layout.manifest_path.exists()
    collections, rows = resolve_domain_collections(
        result.domain_collections,
        result.output_root,
        splits=("train", "val", "test"),
        roles={"LF": "input", "HE": "target"},
        group_metadata=Path("metadata/groups.csv"),
    )
    assert sum(len(v) for (split, domain), v in collections.items() if domain == "LF") == 24
    assert sum(len(v) for (split, domain), v in collections.items() if domain == "HE") == 60
    for split in ("train", "val", "test"):
        dataset = UnpairedImageDataset(
            collections[split, "LF"], collections[split, "HE"], transform=np.asarray
        )
        assert dataset[0]["domain_a"].shape == (8, 8, 3)
        assert len(dataset) == max(len(collections[split, domain]) for domain in ("LF", "HE"))
    snapshot = build_snapshot(
        rows,
        kind="consumed",
        adapter="test",
        roots={"dataset": result.output_root},
        hash_policy="content",
        group_validation="patient",
    )
    assert snapshot.group_validation["status"] == "validated"
    assert {row.patient_id for row in snapshot.rows} == {f"patient{i}" for i in range(4)}
    images = json.loads((layout.metadata_dir / "images.json").read_text())
    assert {(image["geometry"]["width"], image["geometry"]["height"]) for image in images} == {
        (24, 16),
        (32, 24),
    }
    assert not (tmp_path / "results").exists()
    # Actual training adapter applies the same collection and group contract, without training.
    from tests.config_helpers import cyclegan_config_data
    from virtual_staining.applications.train import _unpaired_datasets

    raw = cyclegan_config_data(tmp_path)
    raw["dataset_root"] = str(result.output_root)
    raw["model"]["inputs"], raw["model"]["outputs"] = ["LF"], ["HE"]
    raw["data"] = {
        "pairing": "unpaired",
        "domains": result.domain_collections,
        "group_metadata": "metadata/groups.csv",
        "group_validation": "patient",
    }
    train, val, consumed = _unpaired_datasets(RunConfig.from_mapping(raw), np.asarray, 42)
    assert len(train) and len(val) and consumed.group_validation["unit"] == "patient"


@pytest.mark.parametrize(
    "problem",
    [
        "missing",
        "duplicate",
        "escape",
        "symlink",
        "hardlink",
        "domain",
        "empty",
        "unsupported",
        "metadata",
        "incomplete",
        "unsafe_id",
        "whitespace",
    ],
)
def test_inventory_validation(tmp_path: Path, problem: str) -> None:
    rows = inventory(tmp_path)
    if problem == "missing":
        rows[0]["path"] = "missing.png"
    elif problem == "duplicate":
        rows.append(dict(rows[0]))
    elif problem == "escape":
        rows[0]["path"] = "../outside.png"
    elif problem == "symlink":
        outside = tmp_path.parent / "outside.png"
        outside.write_bytes(b"outside")
        (tmp_path / "escape.png").symlink_to(outside)
        rows[0]["path"] = "escape.png"
    elif problem == "hardlink":
        os.link(tmp_path / rows[1]["path"], tmp_path / "linked.png")
        rows[0]["path"] = "linked.png"
    elif problem == "domain":
        rows[0]["domain"] = "lf"
    elif problem == "empty":
        rows = [row for row in rows if row["domain"] == "LF"]
    elif problem == "unsupported":
        (tmp_path / "file.txt").write_text("bad")
        rows[0]["path"] = "file.txt"
    elif problem == "metadata":
        rows[-1]["patient_id"] = "contradiction"
    elif problem == "incomplete":
        rows[-1]["patient_id"] = ""
    elif problem == "unsafe_id":
        rows[0]["patient_id"] = "../patient"
    elif problem == "whitespace":
        rows[0]["domain"] = "LF "
    write_inventory(tmp_path, rows)
    config = RunConfig.from_mapping(raw_config(tmp_path))
    assert config.preprocessing is not None
    with pytest.raises((ValueError, FileNotFoundError)):
        load_unpaired_inventory(config.preprocessing)
    assert not (tmp_path / "prepared_unpaired").exists()


@pytest.mark.parametrize("unit", ["patient", "specimen", "set"])
def test_group_and_frozen_splits(tmp_path: Path, unit: str) -> None:
    inventory(tmp_path)
    raw = raw_config(tmp_path)
    raw["preprocessing"]["split"]["unit"] = unit
    result = run_prepare(tmp_path, raw)
    frozen = result.output_root / "metadata/split_assignment.csv"
    raw["preprocessing"]["split"].update(assignment_file=str(frozen), seed=1234)
    second = run_prepare(tmp_path, raw)
    assert second.output_root != result.output_root
    assert (second.output_root / "metadata/groups.csv").read_bytes() == (
        result.output_root / "metadata/groups.csv"
    ).read_bytes()
    assert (
        second.output_root / "metadata/split_assignment.csv"
    ).read_bytes() == frozen.read_bytes()


@pytest.mark.parametrize("unit", ["patient", "specimen", "set"])
def test_missing_requested_identity(tmp_path: Path, unit: str) -> None:
    rows = inventory(tmp_path)
    for row in rows:
        row[f"{unit}_id"] = ""
    write_inventory(tmp_path, rows)
    raw = raw_config(tmp_path)
    raw["preprocessing"]["split"]["unit"] = unit
    with pytest.raises(ValueError, match=f"requires explicit {unit}_id"):
        run_prepare(tmp_path, raw)
    assert not (tmp_path / "prepared_unpaired").exists()


def test_patch_split_explicit_exception(tmp_path: Path) -> None:
    inventory(tmp_path)
    raw = raw_config(tmp_path)
    raw["preprocessing"]["split"]["unit"] = "patch"
    with pytest.raises(ValueError, match="unavailable"):
        RunConfig.from_mapping(raw)
    raw["data"]["group_validation"] = "unavailable"
    result = run_prepare(tmp_path, raw)
    assert prepared_unpaired_patch_split(result.output_root, Path("metadata/groups.csv"))
    _, rows = resolve_domain_collections(
        result.domain_collections,
        result.output_root,
        splits=("train", "val", "test"),
        roles={"LF": "input", "HE": "target"},
        group_metadata=Path("metadata/groups.csv"),
    )
    observed = build_snapshot(
        rows,
        kind="consumed",
        adapter="test",
        roots={"dataset": result.output_root},
        hash_policy="content",
        group_validation="unavailable",
        patch_split=True,
    )
    assert observed.group_validation["status"] == "unavailable"
    assert observed.group_validation["groups_shared_across_splits"]


@pytest.mark.parametrize("policy", ["reference", "target", "intersection", "union"])
def test_reject_paired_policies(tmp_path: Path, policy: str) -> None:
    raw = raw_config(tmp_path)
    raw["preprocessing"]["filtering"]["foreground"]["policy"] = policy
    with pytest.raises(ValueError, match="domain-local"):
        RunConfig.from_mapping(raw)


def test_bounded_reading_closes_and_no_registration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inventory(tmp_path)
    from virtual_staining.utils.image_io import open_image_reader

    closed = []

    class Reader:
        def __init__(self, path, backend):
            self.reader = open_image_reader(path, backend=backend)

        @property
        def size(self):
            return self.reader.size

        @property
        def metadata(self):
            return self.reader.metadata

        def read_full(self):
            pytest.fail("whole image read")

        def read_preview(self, scale):
            pytest.fail("unnecessary preview")

        def read_region(self, x, y, width, height):
            assert (width, height) == (8, 8)
            return self.reader.read_region(x, y, width, height)

        def close(self):
            closed.append(self.reader.path)
            self.reader.close()

    monkeypatch.setattr("virtual_staining.data.unpaired_processor.open_image_reader", Reader)
    raw = raw_config(tmp_path)
    raw["preprocessing"]["masks"] = {"generation": "if_missing", "strategy": "hsv"}
    run_prepare(tmp_path, raw)
    assert len(closed) == 9


@pytest.mark.parametrize("change", ["source", "inventory", "config", "assignment"])
def test_stale_builds_are_not_reused(tmp_path: Path, change: str) -> None:
    rows = inventory(tmp_path)
    raw = raw_config(tmp_path)
    first = run_prepare(tmp_path, raw)
    if change == "source":
        path = tmp_path / rows[0]["path"]
        with Image.open(path) as image:
            pixels = np.array(image)
        pixels[0, 0, 0] ^= 1
        Image.fromarray(pixels).save(path)
    elif change == "inventory":
        rows.reverse()
        write_inventory(tmp_path, rows)
    elif change == "config":
        raw["preprocessing"]["split"]["seed"] = 24
    else:
        raw["preprocessing"]["split"]["assignment_file"] = str(
            first.output_root / "metadata/split_assignment.csv"
        )
    second = run_prepare(tmp_path, raw)
    assert not second.reused and second.output_root != first.output_root
    assert first.output_root.is_dir()


def test_failed_publication_and_corrupt_reuse_preserve_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = inventory(tmp_path)
    original = (tmp_path / rows[0]["path"]).read_bytes()
    paired = tmp_path / "splits/train/caller.txt"
    paired.parent.mkdir(parents=True)
    paired.write_text("preserve")
    import importlib

    module = importlib.import_module("virtual_staining.applications.prepare")
    with monkeypatch.context() as patch:
        patch.setattr(
            module,
            "publish_directory_no_replace",
            lambda *args: (_ for _ in ()).throw(OSError("publication failed")),
        )
        with pytest.raises(OSError, match="publication failed"):
            run_prepare(tmp_path)
    assert not list((tmp_path / "prepared_unpaired").glob("*/metadata/dataset_build.json"))
    assert not list((tmp_path / "prepared_unpaired").glob(".prepare-*"))
    assert list((tmp_path / "prepared_unpaired").glob("failure-*.json"))
    result = run_prepare(tmp_path)
    output = next((result.output_root / "splits").rglob("*.png"))
    output.write_bytes(b"caller modification")
    with pytest.raises(FileExistsError, match="preserved"):
        run_prepare(tmp_path)
    assert output.read_bytes() == b"caller modification"
    assert paired.read_text() == "preserve"
    assert (tmp_path / rows[0]["path"]).read_bytes() == original


def test_unreadable_image_is_not_accepted(tmp_path: Path) -> None:
    rows = inventory(tmp_path)
    (tmp_path / rows[0]["path"]).write_bytes(b"not an image")
    with pytest.raises(ValueError, match="domain LF, source"):
        run_prepare(tmp_path)
    assert not list((tmp_path / "prepared_unpaired").glob("*/metadata/dataset_build.json"))


def test_shared_parents_connect_specimen_splits(tmp_path: Path) -> None:
    rows = inventory(tmp_path)
    # Each domain has distinct specimens, but explicit patients connect the two collections.
    for row in rows:
        row["specimen_id"] = row["domain"] + row["specimen_id"]
        row["set_id"] = row["domain"] + row["set_id"]
    write_inventory(tmp_path, rows)
    raw = raw_config(tmp_path)
    raw["preprocessing"]["split"]["unit"] = "specimen"
    config = RunConfig.from_mapping(raw)
    assert config.preprocessing
    items = load_unpaired_inventory(config.preprocessing)
    assigned = unpaired_assignments(config.preprocessing, items, "patient")
    for i in range(4):
        assert assigned[f"LFspecimen{i}"] == assigned[f"HEspecimen{i}"]


@pytest.mark.parametrize("bad", ["mask", "white", "unknown", "geometry", "maskless"])
def test_domain_local_masks_and_filters(tmp_path: Path, bad: str) -> None:
    rows = inventory(tmp_path)
    for row in rows:
        with Image.open(tmp_path / row["path"]) as image:
            mask = np.full((image.height, image.width), 255, dtype=np.uint8)
            mask[-1, rows.index(row)] = 0
        if row is rows[0]:
            if bad == "mask":
                mask[:8, :8] = 0
            elif bad == "unknown":
                mask[0, 0] = 127
            elif bad == "geometry":
                mask = mask[:8, :8]
            elif bad == "white":
                with Image.open(tmp_path / row["path"]) as image:
                    pixels = np.array(image)
                pixels[:8, :8] = 255
                Image.fromarray(pixels).save(tmp_path / row["path"])
        row["mask_path"] = row["path"] + ".mask.png"
        Image.fromarray(mask).save(tmp_path / row["mask_path"])
    if bad == "maskless":
        for row in rows:
            row["mask_path"] = ""
    write_inventory(tmp_path, rows)
    raw = raw_config(tmp_path)
    raw["preprocessing"]["filtering"] = {
        "foreground": {"enabled": True, "policy": "all", "min_ratio": 0.5}
    }
    raw["preprocessing"]["masks"].update(save_patch_masks=True, save_resolved_masks=True)
    if bad in {"unknown", "geometry", "maskless"}:
        with pytest.raises(
            ValueError,
            match={"unknown": "binary", "geometry": "geometry", "maskless": "maskless"}[bad],
        ):
            run_prepare(tmp_path, raw)
        return
    result = run_prepare(tmp_path, raw)
    assert result.skipped_count == 1
    assert sum(1 for _ in (result.output_root / "splits").rglob("*.png")) == 83
    assert list((result.output_root / "masks").rglob("*.png"))
    evidence = json.loads((result.output_root / "metadata/images.json").read_text())
    assert evidence[0]["excluded"][0]["reasons"]


@pytest.mark.parametrize("corruption", ["missing", "extra", "symlink", "marker"])
def test_incomplete_or_caller_owned_artifacts_not_reused(tmp_path: Path, corruption: str) -> None:
    inventory(tmp_path)
    result = run_prepare(tmp_path)
    output = next((result.output_root / "splits").rglob("*.png"))
    if corruption == "missing":
        output.unlink()
    elif corruption == "extra":
        (result.output_root / "caller.txt").write_text("keep")
    elif corruption == "symlink":
        payload = output.read_bytes()
        output.unlink()
        raw = tmp_path / "replacement.png"
        raw.write_bytes(payload)
        output.symlink_to(raw)
    else:
        (result.output_root / "metadata/dataset_build.json").unlink()
    with pytest.raises(FileExistsError, match="preserved"):
        run_prepare(tmp_path)
    if corruption == "extra":
        assert (result.output_root / "caller.txt").read_text() == "keep"


def test_source_mutation_during_build_aborts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = inventory(tmp_path)
    from virtual_staining.data import builder

    process = builder.process_domain_image

    def mutate(*args, **kwargs):
        result = process(*args, **kwargs)
        source = tmp_path / rows[-1]["path"]
        source.touch()
        return result

    monkeypatch.setattr(builder, "process_domain_image", mutate)
    with pytest.raises(ValueError, match="changed during"):
        run_prepare(tmp_path)
    assert not list((tmp_path / "prepared_unpaired").glob("*/metadata/dataset_build.json"))


def test_configuration_and_assets_do_not_decode_or_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = inventory(tmp_path)
    # Preflight checks membership/group feasibility; pixel verification belongs to prepare.
    (tmp_path / rows[0]["path"]).write_bytes(b"unreadable")
    path = write_config_data(tmp_path / "prepare.yaml", raw_config(tmp_path))
    monkeypatch.setattr(
        "virtual_staining.data.unpaired_processor.open_image_reader",
        lambda *a, **k: pytest.fail("decoded"),
    )
    monkeypatch.setattr(
        "virtual_staining.applications.prepare.sha256_file", lambda *a: pytest.fail("hashed")
    )
    cli.main(["config", "check", "--config", str(path), "--stages", "prepare", "--assets"])
    assert not (tmp_path / "prepared_unpaired").exists()


def test_maskless_two_column_inventory_and_no_inferred_ids(tmp_path: Path) -> None:
    rows = [
        {key: value for key, value in row.items() if key in {"domain", "path"}}
        for row in inventory(tmp_path)
    ]
    write_inventory(tmp_path, rows)
    raw = raw_config(tmp_path)
    raw["preprocessing"]["split"]["unit"] = "patch"
    raw["data"]["group_validation"] = "unavailable"
    result = run_prepare(tmp_path, raw)
    with (result.output_root / "metadata/groups.csv").open() as handle:
        assert all(
            not row["set_id"] and not row["patient_id"] and not row["specimen_id"]
            for row in csv.DictReader(handle)
        )


def test_output_parent_escape_is_rejected(tmp_path: Path) -> None:
    inventory(tmp_path)
    outside = tmp_path.parent / "caller_outputs"
    outside.mkdir()
    (tmp_path / "prepared_unpaired").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="escapes"):
        run_prepare(tmp_path)
    assert list(outside.iterdir()) == []


@pytest.mark.slow
def test_native_openslide_preparation_is_tiled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from virtual_staining.utils.image_io import (
        OpenSlideRegionImageReader,
        convert_to_pyramidal_tiff,
    )

    rows = inventory(tmp_path)
    for row in rows:
        destination = row["path"] + ".pyramid.tif"
        convert_to_pyramidal_tiff(tmp_path / row["path"], tmp_path / destination)
        row["path"] = destination
    write_inventory(tmp_path, rows)
    calls = []
    real_region = OpenSlideRegionImageReader.read_region

    def region(self, x, y, width, height):
        calls.append((width, height))
        return real_region(self, x, y, width, height)

    monkeypatch.setattr(OpenSlideRegionImageReader, "read_region", region)
    monkeypatch.setattr(
        OpenSlideRegionImageReader, "read_full", lambda self: pytest.fail("full WSI read")
    )
    monkeypatch.setattr(
        OpenSlideRegionImageReader, "read_preview", lambda self, scale: pytest.fail("WSI preview")
    )
    raw = raw_config(tmp_path)
    raw["preprocessing"]["io"] = {"backend": "openslide", "tiled": True, "max_memory_gb": 0.001}
    result = run_prepare(tmp_path, raw)
    assert calls and set(calls) == {(8, 8)}
    assert result.train_count + result.val_count + result.test_count == 84


def test_duplicate_content_across_groups_fails_before_processing(tmp_path: Path) -> None:
    rows = inventory(tmp_path)
    # Explicit frozen split assignments make the content-leakage fixture unambiguous.
    raw = raw_config(tmp_path)
    (tmp_path / "frozen.csv").write_text(
        "group_id,unit,split\npatient0,patient,train\npatient1,patient,val\npatient2,patient,test\npatient3,patient,train\n"
    )
    raw["preprocessing"]["split"]["assignment_file"] = "frozen.csv"
    (tmp_path / rows[1]["path"]).write_bytes((tmp_path / rows[0]["path"]).read_bytes())
    with pytest.raises(ValueError, match="same content"):
        run_prepare(tmp_path, raw)
    assert not (tmp_path / "prepared_unpaired").exists()


@pytest.mark.parametrize("budget", [0.0000001, 0.00001])
def test_memory_limit_fails_and_closes_reader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, budget: float
) -> None:
    inventory(tmp_path)
    from virtual_staining.utils.image_io import PillowRegionImageReader

    closed = []
    monkeypatch.setattr(PillowRegionImageReader, "close", lambda self: closed.append(self.path))
    raw = raw_config(tmp_path)
    raw["preprocessing"]["io"] = {"max_memory_gb": budget}
    with pytest.raises(ValueError, match="max_memory_gb"):
        run_prepare(tmp_path, raw)
    assert len(closed) == 1


@pytest.mark.parametrize(
    "header,body",
    [
        ("domain,path,path", "LF,a.png,a.png"),
        ("domain,path,extra", "LF,a.png,no"),
        ("domain,path", "LF,a.png,extra"),
        ("domain,path,patient_id", "LF,a.png"),
    ],
)
def test_malformed_inventory_schema(tmp_path: Path, header: str, body: str) -> None:
    (tmp_path / "paths.csv").write_text(header + "\n" + body + "\n")
    config = RunConfig.from_mapping(raw_config(tmp_path))
    assert config.preprocessing
    with pytest.raises(ValueError):
        load_unpaired_inventory(config.preprocessing)


@pytest.mark.parametrize("existing", ["empty", "file", "directory", "symlink", "absent"])
def test_exclusive_directory_publication(tmp_path: Path, existing: str) -> None:
    from virtual_staining.utils.files import publish_directory_no_replace

    source, destination = tmp_path / "stage", tmp_path / "published"
    source.mkdir()
    (source / "artifact").write_text("prepared")
    if existing == "absent":
        publish_directory_no_replace(source, destination)
        assert (destination / "artifact").read_text() == "prepared"
        assert not source.exists()
        return
    if existing in {"empty", "directory"}:
        destination.mkdir()
        if existing == "directory":
            (destination / "caller").write_text("keep")
    elif existing == "file":
        destination.write_text("keep")
    else:
        destination.symlink_to(tmp_path / "missing")
    inode = destination.lstat().st_ino
    with pytest.raises(FileExistsError):
        publish_directory_no_replace(source, destination)
    assert destination.lstat().st_ino == inode
    assert (source / "artifact").read_text() == "prepared"


@pytest.mark.slow
def test_committed_unpaired_example_executes(tmp_path: Path) -> None:
    import shutil

    import yaml

    root = Path(__file__).resolve().parents[2]
    config_path = root / "examples/unpaired/prepare.yaml"
    raw = yaml.safe_load(config_path.read_text())
    raw["dataset_root"] = str(tmp_path)
    (tmp_path / "unpaired").mkdir()
    shutil.copyfile(root / "examples/unpaired/paths.csv", tmp_path / "unpaired/paths.csv")
    for path in (root / "examples").glob("*.png"):
        shutil.copyfile(path, tmp_path / path.name)
    path = write_config_data(tmp_path / "prepare.yaml", raw)
    cli.main(["prepare", "--config", str(path)])
    result = run_prepare(tmp_path, raw)
    assert result.reused
    assert result.train_count and result.val_count and result.test_count
