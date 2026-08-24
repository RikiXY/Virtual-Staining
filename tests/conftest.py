from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from tests.image_helpers import write_rgb_image
from tests.manifest_helpers import make_manifest_record, manifest_metadata
from virtual_staining.data.manifest import DatasetManifest, ManifestRecord


@dataclass(frozen=True)
class ManifestDataset:
    root: Path
    manifest_path: Path
    manifest: DatasetManifest
    train_records: tuple[ManifestRecord, ...]
    val_records: tuple[ManifestRecord, ...]
    test_records: tuple[ManifestRecord, ...]


def _write_tiny_image(path: Path) -> None:
    """Write a minimal valid RGB TIFF."""
    write_rgb_image(path, size=(3, 3))


@pytest.fixture
def manifest_dataset(tmp_path: Path) -> ManifestDataset:
    """Minimal manifest-backed dataset with one train, one val, and one test record."""
    records: list[ManifestRecord] = []
    split_samples = (
        ("train", "00000_00000"),
        ("val", "00256_00000"),
        ("test", "00512_00000"),
    )
    for split, sample_id in split_samples:
        x_str, y_str = sample_id.split("_", maxsplit=1)
        input_path = Path(f"splits/{split}/{sample_id}__input__label_free.tif")
        target_path = Path(f"splits/{split}/{sample_id}__target.tif")
        _write_tiny_image(tmp_path / input_path)
        _write_tiny_image(tmp_path / target_path)
        records.append(
            make_manifest_record(
                sample_id,
                split,
                input_paths={"label_free": input_path},
                target_path=target_path,
                x=int(x_str),
                y=int(y_str),
            )
        )

    manifest = DatasetManifest(
        records=tuple(records), dataset_root=tmp_path, metadata=manifest_metadata()
    )
    manifest_path = tmp_path / "manifests" / "manifest.csv"
    manifest.to_csv(manifest_path)

    return ManifestDataset(
        root=tmp_path,
        manifest_path=manifest_path,
        manifest=manifest,
        train_records=tuple(record for record in records if record.split == "train"),
        val_records=tuple(record for record in records if record.split == "val"),
        test_records=tuple(record for record in records if record.split == "test"),
    )
