from __future__ import annotations

from pathlib import Path

from tests.manifest_helpers import make_manifest_records, write_manifest_csv
from virtual_staining.data.manifest import DatasetManifest


def test_make_manifest_records_defaults() -> None:
    records = make_manifest_records(3)

    assert len(records) == 3
    assert len({record.sample_id for record in records}) == 3
    assert {record.split for record in records} == {"train", "val"}


def test_make_manifest_records_custom_splits() -> None:
    records = make_manifest_records(3, splits=["train", "val", "test"])

    assert [record.split for record in records] == ["train", "val", "test"]


def test_write_manifest_csv_round_trip(tmp_path: Path) -> None:
    records = make_manifest_records(2, splits=["train", "val"])

    path = write_manifest_csv(tmp_path, records)

    assert path.exists()
    loaded = DatasetManifest.from_csv(path, dataset_root=tmp_path)
    assert len(loaded.records) == 2
    assert loaded.records[0].sample_id == records[0].sample_id
