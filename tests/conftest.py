from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
from PIL import Image

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
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (3, 3)).save(path)


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
        input_path = Path(f"dataset_{split}/{sample_id}_source.tif")
        target_path = Path(f"dataset_{split}/{sample_id}_target.tif")
        _write_tiny_image(tmp_path / input_path)
        _write_tiny_image(tmp_path / target_path)
        records.append(
            ManifestRecord(
                sample_id=sample_id,
                split=split,  # type: ignore[arg-type]
                input_path=input_path,
                target_path=target_path,
                input_modality="label_free",
                target_modality="stained",
                x=int(x_str),
                y=int(y_str),
                width=256,
                height=256,
            )
        )

    manifest = DatasetManifest(records=tuple(records), dataset_root=tmp_path)
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
