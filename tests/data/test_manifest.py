from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from tests.manifest_helpers import make_manifest_record
from virtual_staining.config.project import ProjectConfig
from virtual_staining.data.manifest import (
    MANIFEST_SCHEMA_VERSION,
    DatasetManifest,
    ManifestMetadata,
    ManifestRecord,
    load_manifest_or_raise,
    manifest_fieldnames,
    require_model_modalities,
)


def _metadata(
    inputs: tuple[str, ...] = ("LF", "AF"), targets: tuple[str, ...] = ("HE", "PAS")
) -> ManifestMetadata:
    return ManifestMetadata(
        schema_version=MANIFEST_SCHEMA_VERSION,
        input_modalities=inputs,
        target_modalities=targets,
        reference_modality=inputs[0],
    )


def _record(
    sample_id: str = "a",
    split: str = "train",
    *,
    inputs: tuple[str, ...] = ("LF", "AF"),
    targets: tuple[str, ...] = ("HE", "PAS"),
    masks: dict[str, Path | None] | None = None,
) -> ManifestRecord:
    base = Path(f"splits/{split}/s")
    return make_manifest_record(
        sample_id,
        split,
        input_paths={name: base / f"{sample_id}__input__{name}.png" for name in inputs},
        target_paths={name: base / f"{sample_id}__target__{name}.png" for name in targets},
        foreground_mask_paths=masks,
        set_id="s",
        x=3,
        y=5,
        width=16,
        height=8,
    )


def _manifest(tmp_path: Path, *records: ManifestRecord, **metadata: Any) -> DatasetManifest:
    return DatasetManifest(records or (_record(),), tmp_path, _metadata(**metadata))


def test_schema_is_v4_with_exact_canonical_metadata() -> None:
    assert MANIFEST_SCHEMA_VERSION == "4.0"
    assert DatasetManifest.SCHEMA_VERSION == "4.0"
    assert _metadata().to_dict() == {
        "schema_version": "4.0",
        "input_modalities": ["LF", "AF"],
        "target_modalities": ["HE", "PAS"],
        "reference_modality": "LF",
        "coordinate_space": "reference_level0_pixels",
        "pixel_center": "integer",
    }
    assert ManifestMetadata.from_mapping(_metadata().to_dict()) == _metadata()


def test_metadata_accepts_builder_producer_fields() -> None:
    data = {**_metadata().to_dict(), "created_at": "now", "record_count": 3, "splits": {}}

    assert ManifestMetadata.from_mapping(data) == _metadata()


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"schema_version": "3.0"}, "exactly 4.0"),
        ({"schema_version": 4.0}, "must be a string"),
        ({"target_modalities": "HE"}, "list of names"),
        ({"target_modalities": []}, "non-empty"),
        ({"target_modalities": ["HE", "HE"]}, "non-empty and unique"),
        ({"input_modalities": ["LF", 3]}, "list of names"),
        ({"target_modalities": ["H&E"]}, "unsafe identifiers"),
        ({"target_modalities": ["LF"]}, "differ from all input"),
        ({"reference_modality": "HE"}, "one of input_modalities"),
        ({"coordinate_space": "model_pixels"}, "coordinate_space"),
        ({"pixel_center": "half"}, "pixel_center"),
        ({"record_count": "3"}, "record_count must be a int"),
        ({"record_count": True}, "record_count must be a int"),
        ({"extra": 1}, "unknown fields"),
    ],
)
def test_metadata_is_strict(change: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        ManifestMetadata.from_mapping({**_metadata().to_dict(), **change})


def test_v3_metadata_is_rejected_not_converted() -> None:
    v3 = {
        "schema_version": "3.0",
        "input_modalities": ["LF"],
        "reference_modality": "LF",
        "target_modality": "HE",
    }
    with pytest.raises(ValueError, match="missing required fields|unknown fields"):
        ManifestMetadata.from_mapping(v3)


@pytest.mark.parametrize("targets", [("HE",), ("PAS", "HE")])
def test_rows_round_trip_with_ordered_targets_and_optional_masks(
    tmp_path: Path, targets: tuple[str, ...]
) -> None:
    masks: dict[str, Path | None] = dict.fromkeys(targets)
    masks[targets[0]] = Path(f"splits/train/s/a__foreground_mask__{targets[0]}.png")
    manifest = _manifest(tmp_path, _record(targets=targets, masks=masks), targets=targets)
    path = tmp_path / "manifest.csv"

    manifest.to_csv(path)

    assert manifest.fieldnames == (
        "sample_id",
        "set_id",
        "split",
        "input__LF",
        "input__AF",
        *(f"target__{name}" for name in targets),
        *(f"foreground_mask__{name}" for name in targets),
        "x",
        "y",
        "width",
        "height",
    )
    assert path.read_text(encoding="utf-8").splitlines()[0] == ",".join(manifest.fieldnames)
    loaded = DatasetManifest.from_csv(path, tmp_path, manifest.metadata)
    assert loaded.records == manifest.records
    record = loaded.records[0]
    assert tuple(record.target_paths) == targets
    assert record.foreground_mask_paths[targets[0]] is not None
    assert all(record.foreground_mask_paths[name] is None for name in targets[1:])
    assert (record.x, record.y, record.width, record.height) == (3, 5, 16, 8)


def _write_csv(tmp_path: Path, header: list[str], *rows: list[str]) -> Path:
    path = tmp_path / "manifest.csv"
    path.write_text("\n".join(",".join(line) for line in (header, *rows)) + "\n", encoding="utf-8")
    return path


def _row(**overrides: str) -> list[str]:
    values = {
        "sample_id": "a",
        "set_id": "s",
        "split": "train",
        "input__LF": "i/lf.png",
        "input__AF": "i/af.png",
        "target__HE": "t/he.png",
        "target__PAS": "t/pas.png",
        "foreground_mask__HE": "",
        "foreground_mask__PAS": "",
        "x": "0",
        "y": "0",
        "width": "4",
        "height": "4",
    }
    values.update(overrides)
    return [values[name] for name in manifest_fieldnames(_metadata())]


@pytest.mark.parametrize(
    ("header", "match"),
    [
        (["sample_id", "set_id", "split", "input__LF", "target_path"], "exact v4 columns"),
        (
            [
                *manifest_fieldnames(_metadata())[:6],
                "target__HE",
                *manifest_fieldnames(_metadata())[7:],
            ],
            "duplicate columns",
        ),
        (list(manifest_fieldnames(_metadata()))[::-1], "exact v4 columns"),
        ([*manifest_fieldnames(_metadata()), "notes"], "exact v4 columns"),
        (
            [name for name in manifest_fieldnames(_metadata()) if name != "target__PAS"],
            "exact v4 columns",
        ),
        (
            [
                "sample_id",
                "set_id",
                "split",
                "input__LF",
                "input__AF",
                "target_path",
                "foreground_mask_path",
                "x",
                "y",
                "width",
                "height",
            ],
            "exact v4 columns",
        ),
    ],
)
def test_csv_header_must_be_exact(tmp_path: Path, header: list[str], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        DatasetManifest.from_csv(_write_csv(tmp_path, header), tmp_path, _metadata())


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"target__PAS": ""}, "must not be empty"),
        ({"input__AF": "/abs.png"}, "relative, non-traversing"),
        ({"target__HE": "../out.png"}, "relative, non-traversing"),
        ({"foreground_mask__HE": "../m.png"}, "relative, non-traversing"),
        ({"x": "one"}, "x must be an integer"),
        ({"width": "0"}, "dimensions positive"),
        ({"y": "-1"}, "nonnegative"),
        ({"split": "holdout"}, "Invalid split"),
        ({"sample_id": " "}, "must not be empty"),
        ({"target__PAS": "i/lf.png"}, "must all differ"),
        ({"target__PAS": "t/he.png"}, "must all differ"),
    ],
)
def test_csv_rows_are_validated(tmp_path: Path, overrides: dict[str, str], match: str) -> None:
    path = _write_csv(tmp_path, list(manifest_fieldnames(_metadata())), _row(**overrides))
    with pytest.raises(ValueError, match=match):
        DatasetManifest.from_csv(path, tmp_path, _metadata())


def test_csv_rows_with_missing_or_extra_cells_are_rejected(tmp_path: Path) -> None:
    header = list(manifest_fieldnames(_metadata()))
    for row in (_row()[:-1], [*_row(), "extra"]):
        with pytest.raises(ValueError, match="cells"):
            DatasetManifest.from_csv(_write_csv(tmp_path, header, row), tmp_path, _metadata())


@pytest.mark.parametrize(
    ("records", "match"),
    [
        ((_record("a"), _record("a", "val")), "multiple splits"),
        ((_record("a"), _record("a")), "Duplicate"),
    ],
)
def test_sample_identity_is_unique(
    tmp_path: Path, records: tuple[ManifestRecord, ...], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        _manifest(tmp_path, *records).validate()


def test_path_assignments_are_unique_across_records(tmp_path: Path) -> None:
    first = _record("a")
    shared_input = make_manifest_record(
        "b",
        "train",
        x=0,
        y=0,
        input_paths={"LF": first.input_paths["LF"], "AF": Path("x/af.png")},
        target_paths={"HE": Path("x/he.png"), "PAS": Path("x/pas.png")},
    )
    shared_target = make_manifest_record(
        "b",
        "train",
        x=0,
        y=0,
        input_paths={"LF": Path("x/lf.png"), "AF": Path("x/af.png")},
        target_paths={"HE": first.target_paths["PAS"], "PAS": Path("x/pas.png")},
    )
    input_as_target = make_manifest_record(
        "b",
        "train",
        x=0,
        y=0,
        input_paths={"LF": Path("x/lf.png"), "AF": Path("x/af.png")},
        target_paths={"HE": first.input_paths["AF"], "PAS": Path("x/pas.png")},
    )
    for second, match in (
        (shared_input, "Duplicate input paths"),
        (shared_target, "Duplicate target paths"),
        (input_as_target, "both an input and a target"),
    ):
        with pytest.raises(ValueError, match=match):
            _manifest(tmp_path, first, second).validate()


def test_records_must_carry_exactly_the_declared_named_images(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="target keys"):
        _manifest(tmp_path, _record(targets=("HE",))).validate()
    with pytest.raises(ValueError, match="target keys"):
        _manifest(tmp_path, _record(targets=("PAS", "HE"))).validate()
    with pytest.raises(ValueError, match="input keys"):
        _manifest(tmp_path, _record(inputs=("LF",))).validate()
    with pytest.raises(ValueError, match="name exactly the targets"):
        _record(masks={"HE": None})


def test_validation_checks_files_symlinks_and_required_splits(tmp_path: Path) -> None:
    root = tmp_path / "root"
    manifest = _manifest(root)
    record = manifest.records[0]
    with pytest.raises(FileNotFoundError, match="Manifest file not found"):
        manifest.validate(check_files_exist=True)
    for path in (*record.input_paths.values(), *record.target_paths.values()):
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_bytes(b"x")
    manifest.validate(check_files_exist=True, require_splits={"train"})
    with pytest.raises(ValueError, match="no records"):
        manifest.validate(require_splits={"test"})

    outside = tmp_path / "outside.png"
    outside.write_bytes(b"x")
    escaping = root / record.target_paths["PAS"]
    escaping.unlink()
    os.symlink(outside, escaping)
    with pytest.raises(ValueError, match="resolves outside the dataset root"):
        manifest.validate(check_files_exist=True)


def test_load_manifest_rejects_v3_metadata_with_a_clear_error(tmp_path: Path) -> None:
    (tmp_path / "manifests").mkdir()
    (tmp_path / "manifests" / "manifest.csv").write_text("sample_id\n", encoding="utf-8")
    (tmp_path / "manifests" / "manifest_metadata.json").write_text(
        json.dumps({**_metadata().to_dict(), "schema_version": "3.0"}), encoding="utf-8"
    )
    project = ProjectConfig(
        dataset_root=tmp_path, results_path=tmp_path, run_name="r", image_size=(8, 8)
    )
    with pytest.raises(ValueError, match="exactly 4.0"):
        load_manifest_or_raise(project)


def test_model_io_must_be_subsets_of_the_manifest_modalities(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    require_model_modalities(manifest, ("AF",), ("PAS",))
    require_model_modalities(manifest, ("AF", "LF"), ("PAS", "HE"))
    with pytest.raises(ValueError, match="not manifest target modalities"):
        require_model_modalities(manifest, ("LF",), ("IHC",))
    with pytest.raises(ValueError, match="not manifest input modalities"):
        require_model_modalities(manifest, ("HE",), ("PAS",))
