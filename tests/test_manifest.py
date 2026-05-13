from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import pytest

from virtual_staining.config.project import ProjectConfig
from virtual_staining.data.manifest import DatasetManifest, ManifestRecord, load_manifest_or_raise


def _make_record(sample_id: str, split: str = "train") -> ManifestRecord:
    return ManifestRecord(
        sample_id=sample_id,
        split=split,  # type: ignore[arg-type]
        input_path=Path(f"splits/{split}/{sample_id}_source.tif"),
        target_path=Path(f"splits/{split}/{sample_id}_target.tif"),
        input_modality="label_free",
        target_modality="stained",
        x=0,
        y=0,
        width=256,
        height=256,
    )


def test_manifest_record_frozen() -> None:
    rec = _make_record("00000_00000")
    with pytest.raises((AttributeError, TypeError)):
        rec.sample_id = "other"  # type: ignore[misc]


def test_manifest_record_empty_sample_id_raises() -> None:
    with pytest.raises(ValueError, match="sample_id"):
        ManifestRecord(
            sample_id="",
            split="train",
            input_path=Path("a/src.tif"),
            target_path=Path("a/tgt.tif"),
            input_modality="label_free",
            target_modality="stained",
            x=0,
            y=0,
            width=256,
            height=256,
        )


def test_manifest_record_invalid_split_raises() -> None:
    with pytest.raises(ValueError, match="split"):
        ManifestRecord(
            sample_id="abc",
            split="validation",  # type: ignore[arg-type]
            input_path=Path("a/src.tif"),
            target_path=Path("a/tgt.tif"),
            input_modality="label_free",
            target_modality="stained",
            x=0,
            y=0,
            width=256,
            height=256,
        )


def test_manifest_record_empty_input_modality_raises() -> None:
    with pytest.raises(ValueError, match="input_modality"):
        ManifestRecord(
            sample_id="abc",
            split="train",
            input_path=Path("a/src.tif"),
            target_path=Path("a/tgt.tif"),
            input_modality="",
            target_modality="stained",
            x=0,
            y=0,
            width=256,
            height=256,
        )


def test_manifest_record_empty_target_modality_raises() -> None:
    with pytest.raises(ValueError, match="target_modality"):
        ManifestRecord(
            sample_id="abc",
            split="train",
            input_path=Path("a/src.tif"),
            target_path=Path("a/tgt.tif"),
            input_modality="label_free",
            target_modality="",
            x=0,
            y=0,
            width=256,
            height=256,
        )


def test_manifest_record_negative_x_raises() -> None:
    with pytest.raises(ValueError, match="x must be"):
        ManifestRecord(
            sample_id="abc",
            split="train",
            input_path=Path("a/src.tif"),
            target_path=Path("a/tgt.tif"),
            input_modality="label_free",
            target_modality="stained",
            x=-1,
            y=0,
            width=256,
            height=256,
        )


def test_manifest_record_negative_y_raises() -> None:
    with pytest.raises(ValueError, match="y must be"):
        ManifestRecord(
            sample_id="abc",
            split="train",
            input_path=Path("a/src.tif"),
            target_path=Path("a/tgt.tif"),
            input_modality="label_free",
            target_modality="stained",
            x=0,
            y=-1,
            width=256,
            height=256,
        )


def test_manifest_record_zero_width_raises() -> None:
    with pytest.raises(ValueError, match="width"):
        ManifestRecord(
            sample_id="abc",
            split="train",
            input_path=Path("a/src.tif"),
            target_path=Path("a/tgt.tif"),
            input_modality="label_free",
            target_modality="stained",
            x=0,
            y=0,
            width=0,
            height=256,
        )


def test_manifest_record_zero_height_raises() -> None:
    with pytest.raises(ValueError, match="height"):
        ManifestRecord(
            sample_id="abc",
            split="train",
            input_path=Path("a/src.tif"),
            target_path=Path("a/tgt.tif"),
            input_modality="label_free",
            target_modality="stained",
            x=0,
            y=0,
            width=256,
            height=0,
        )


def test_manifest_record_same_input_target_path_raises() -> None:
    path = Path("a/file.tif")
    with pytest.raises(ValueError, match="different"):
        ManifestRecord(
            sample_id="abc",
            split="train",
            input_path=path,
            target_path=path,
            input_modality="label_free",
            target_modality="stained",
            x=0,
            y=0,
            width=256,
            height=256,
        )


def test_manifest_record_absolute_path_raises() -> None:
    with pytest.raises(ValueError, match="relative"):
        ManifestRecord(
            sample_id="abc",
            split="train",
            input_path=Path("/tmp/source.tif"),
            target_path=Path("splits/train/abc_target.tif"),
            input_modality="label_free",
            target_modality="stained",
            x=0,
            y=0,
            width=256,
            height=256,
        )


def test_manifest_record_empty_path_raises() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        ManifestRecord(
            sample_id="abc",
            split="train",
            input_path=Path(""),
            target_path=Path("splits/train/abc_target.tif"),
            input_modality="label_free",
            target_modality="stained",
            x=0,
            y=0,
            width=256,
            height=256,
        )


def test_manifest_record_traversal_path_raises() -> None:
    with pytest.raises(ValueError, match=r"\.\."):
        ManifestRecord(
            sample_id="abc",
            split="train",
            input_path=Path("../outside/source.tif"),
            target_path=Path("splits/train/abc_target.tif"),
            input_modality="label_free",
            target_modality="stained",
            x=0,
            y=0,
            width=256,
            height=256,
        )


def test_manifest_record_nested_relative_path_passes() -> None:
    rec = ManifestRecord(
        sample_id="abc",
        split="train",
        input_path=Path("dataset_train/00512_09216_source.tif"),
        target_path=Path("dataset_train/00512_09216_target.tif"),
        input_modality="label_free",
        target_modality="stained",
        x=512,
        y=9216,
        width=256,
        height=256,
    )

    assert rec.input_path == Path("dataset_train/00512_09216_source.tif")


def test_dataset_manifest_filter_split() -> None:
    records = (_make_record("a", "train"), _make_record("b", "val"), _make_record("c", "train"))
    manifest = DatasetManifest(records=records, dataset_root=Path("/tmp"))
    train = manifest.filter_split("train")
    assert len(train) == 2
    assert all(r.split == "train" for r in train.records)


def test_dataset_manifest_filter_split_empty_result() -> None:
    records = (_make_record("a", "train"),)
    manifest = DatasetManifest(records=records, dataset_root=Path("/tmp"))
    assert len(manifest.filter_split("val")) == 0


def test_dataset_manifest_validate_duplicate_ids_raises() -> None:
    records = (_make_record("dup"), _make_record("dup"))
    manifest = DatasetManifest(records=records, dataset_root=Path("/tmp"))
    with pytest.raises(ValueError, match="Duplicate sample_ids"):
        manifest.validate()


def test_dataset_manifest_validate_unique_ids_passes() -> None:
    records = (_make_record("a"), _make_record("b"))
    manifest = DatasetManifest(records=records, dataset_root=Path("/tmp"))
    manifest.validate()  # must not raise


def test_validate_duplicate_input_path_raises() -> None:
    shared_input = Path("dataset_train/shared_source.tif")
    records = (
        ManifestRecord(
            "a",
            "train",
            shared_input,
            Path("a/tgt.tif"),
            "lf",
            "st",
            0,
            0,
            256,
            256,
        ),
        ManifestRecord(
            "b",
            "train",
            shared_input,
            Path("b/tgt.tif"),
            "lf",
            "st",
            64,
            0,
            256,
            256,
        ),
    )
    manifest = DatasetManifest(records=records, dataset_root=Path("/tmp"))

    with pytest.raises(ValueError, match="input_path"):
        manifest.validate()


def test_validate_duplicate_target_path_raises() -> None:
    shared_target = Path("dataset_train/shared_target.tif")
    records = (
        ManifestRecord(
            "a",
            "train",
            Path("a/src.tif"),
            shared_target,
            "lf",
            "st",
            0,
            0,
            256,
            256,
        ),
        ManifestRecord(
            "b",
            "train",
            Path("b/src.tif"),
            shared_target,
            "lf",
            "st",
            64,
            0,
            256,
            256,
        ),
    )
    manifest = DatasetManifest(records=records, dataset_root=Path("/tmp"))

    with pytest.raises(ValueError, match="target_path"):
        manifest.validate()


def test_validate_sample_in_train_and_val_raises() -> None:
    records = (
        ManifestRecord(
            "abc",
            "train",
            Path("train/src.tif"),
            Path("train/tgt.tif"),
            "lf",
            "st",
            0,
            0,
            256,
            256,
        ),
        ManifestRecord(
            "abc",
            "val",
            Path("val/src.tif"),
            Path("val/tgt.tif"),
            "lf",
            "st",
            0,
            0,
            256,
            256,
        ),
    )
    manifest = DatasetManifest(records=records, dataset_root=Path("/tmp"))

    with pytest.raises(ValueError, match="multiple splits"):
        manifest.validate()


def test_validate_sample_in_train_and_discarded_passes() -> None:
    records = (
        ManifestRecord(
            "abc",
            "train",
            Path("train/src.tif"),
            Path("train/tgt.tif"),
            "lf",
            "st",
            0,
            0,
            256,
            256,
        ),
        ManifestRecord(
            "abc",
            "discarded",
            Path("discarded/src.tif"),
            Path("discarded/tgt.tif"),
            "lf",
            "st",
            0,
            0,
            256,
            256,
        ),
    )
    manifest = DatasetManifest(records=records, dataset_root=Path("/tmp"))
    manifest.validate()


def test_validate_require_splits_raises_for_missing_split(tmp_path: Path) -> None:
    manifest = DatasetManifest(records=(_make_record("a", "train"),), dataset_root=tmp_path)

    with pytest.raises(ValueError, match="val"):
        manifest.validate(require_splits={"train", "val"})


def test_validate_require_splits_raises_for_missing_test_split(tmp_path: Path) -> None:
    manifest = DatasetManifest(records=(_make_record("a", "train"),), dataset_root=tmp_path)

    with pytest.raises(ValueError, match="test"):
        manifest.validate(require_splits={"test"})


def test_validate_require_splits_passes_when_all_present(tmp_path: Path) -> None:
    records = (_make_record("a", "train"), _make_record("b", "val"))
    manifest = DatasetManifest(records=records, dataset_root=tmp_path)

    manifest.validate(require_splits={"train", "val"})


def test_validate_check_files_exist_raises_for_missing_input_file(tmp_path: Path) -> None:
    manifest = DatasetManifest(records=(_make_record("a", "train"),), dataset_root=tmp_path)

    with pytest.raises(FileNotFoundError, match="Input file not found"):
        manifest.validate(check_files_exist=True)


def test_dataset_manifest_len() -> None:
    records = tuple(_make_record(str(i)) for i in range(5))
    manifest = DatasetManifest(records=records, dataset_root=Path("/tmp"))
    assert len(manifest) == 5


def test_dataset_manifest_roundtrip_csv(tmp_path: Path) -> None:
    records = (_make_record("a", "train"), _make_record("b", "val"))
    manifest = DatasetManifest(records=records, dataset_root=tmp_path)
    csv_path = tmp_path / "manifest.csv"
    manifest.to_csv(csv_path)

    loaded = DatasetManifest.from_csv(csv_path, dataset_root=tmp_path)
    assert len(loaded) == 2
    assert loaded.records[0].sample_id == "a"
    assert loaded.records[1].split == "val"
    assert loaded.records[0].x == 0
    assert loaded.records[0].width == 256


def test_dataset_manifest_roundtrip_preserves_paths(tmp_path: Path) -> None:
    rec = _make_record("x", "test")
    manifest = DatasetManifest(records=(rec,), dataset_root=tmp_path)
    csv_path = tmp_path / "manifest.csv"
    manifest.to_csv(csv_path)
    loaded = DatasetManifest.from_csv(csv_path, dataset_root=tmp_path)
    assert loaded.records[0].input_path == Path("splits/test/x_source.tif")


def test_load_manifest_or_raise_raises_if_missing(tmp_path: Path) -> None:
    project = ProjectConfig(
        dataset_root=tmp_path / "dataset",
        results_path=tmp_path / "results",
        run_name="run",
        image_size=(256, 256),
        manifest_path_override=tmp_path / "dataset" / "manifests" / "manifest.csv",
    )

    with pytest.raises(FileNotFoundError, match="Manifest not found"):
        load_manifest_or_raise(project)


def test_load_manifest_or_raise_returns_manifest(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    manifest_path = dataset_root / "manifests" / "manifest.csv"
    manifest_path.parent.mkdir(parents=True)
    manifest = DatasetManifest(records=(_make_record("a"),), dataset_root=dataset_root)
    manifest.to_csv(manifest_path)
    project = ProjectConfig(
        dataset_root=dataset_root,
        results_path=tmp_path / "results",
        run_name="run",
        image_size=(256, 256),
    )

    loaded = load_manifest_or_raise(project)

    assert len(loaded.records) == 1
    assert loaded.records[0].sample_id == "a"


def test_dataset_manifest_from_csv_rejects_traversal_path(tmp_path: Path) -> None:
    csv_path = tmp_path / "manifest.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_id",
                "split",
                "input_path",
                "target_path",
                "input_modality",
                "target_modality",
                "x",
                "y",
                "width",
                "height",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "sample_id": "abc",
                "split": "train",
                "input_path": "../outside/source.tif",
                "target_path": "splits/train/abc_target.tif",
                "input_modality": "label_free",
                "target_modality": "stained",
                "x": 0,
                "y": 0,
                "width": 256,
                "height": 256,
            }
        )

    with pytest.raises(ValueError, match=r"\.\."):
        DatasetManifest.from_csv(csv_path, dataset_root=tmp_path)


def test_from_csv_missing_column_raises(tmp_path: Path) -> None:
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text(
        "sample_id,split,input_path,target_path,input_modality,target_modality,x,y,height\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="width"):
        DatasetManifest.from_csv(csv_path, tmp_path)


def test_from_csv_unexpected_column_raises(tmp_path: Path) -> None:
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text(
        (
            "sample_id,split,input_path,target_path,input_modality,target_modality,"
            "x,y,width,height,extra\n"
            "abc,train,a/s.tif,a/t.tif,lf,st,0,0,256,256,ignored\n"
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unexpected columns"):
        DatasetManifest.from_csv(csv_path, tmp_path)


def test_from_csv_invalid_integer_raises(tmp_path: Path) -> None:
    csv_path = tmp_path / "bad.csv"
    header = (
        "sample_id,split,input_path,target_path,input_modality,target_modality,x,y,width,height\n"
    )
    row = "abc,train,a/s.tif,a/t.tif,lf,st,NOT_AN_INT,0,256,256\n"
    csv_path.write_text(header + row, encoding="utf-8")

    with pytest.raises(ValueError, match="field 'x' must be an integer"):
        DatasetManifest.from_csv(csv_path, tmp_path)


def test_from_csv_invalid_split_raises(tmp_path: Path) -> None:
    csv_path = tmp_path / "bad.csv"
    header = (
        "sample_id,split,input_path,target_path,input_modality,target_modality,x,y,width,height\n"
    )
    row = "abc,INVALID_SPLIT,a/s.tif,a/t.tif,lf,st,0,0,256,256\n"
    csv_path.write_text(header + row, encoding="utf-8")

    with pytest.raises(ValueError, match="split must be one of"):
        DatasetManifest.from_csv(csv_path, tmp_path)


def test_from_csv_empty_sample_id_raises(tmp_path: Path) -> None:
    csv_path = tmp_path / "bad.csv"
    header = (
        "sample_id,split,input_path,target_path,input_modality,target_modality,x,y,width,height\n"
    )
    row = ",train,a/s.tif,a/t.tif,lf,st,0,0,256,256\n"
    csv_path.write_text(header + row, encoding="utf-8")

    with pytest.raises(ValueError, match="sample_id"):
        DatasetManifest.from_csv(csv_path, tmp_path)


def test_from_csv_warns_on_schema_version_mismatch(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    manifest = DatasetManifest(records=(_make_record("a", "train"),), dataset_root=tmp_path)
    manifest_path = tmp_path / "manifests" / "manifest.csv"
    manifest.to_csv(manifest_path)
    metadata_path = manifest_path.parent / "manifest_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "schema_version": "0.9",
                "created_at": "2026-01-01T00:00:00+00:00",
                "record_count": 1,
                "splits": {"train": 1, "val": 0, "test": 0},
            }
        ),
        encoding="utf-8",
    )

    with caplog.at_level(logging.WARNING):
        DatasetManifest.from_csv(manifest_path, dataset_root=tmp_path)

    assert "schema version mismatch" in caplog.text.lower()


def test_from_csv_silent_when_metadata_absent(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    manifest = DatasetManifest(records=(_make_record("a", "train"),), dataset_root=tmp_path)
    manifest_path = tmp_path / "manifests" / "manifest.csv"
    manifest.to_csv(manifest_path)

    with caplog.at_level(logging.WARNING):
        DatasetManifest.from_csv(manifest_path, dataset_root=tmp_path)

    assert "schema version" not in caplog.text.lower()
