from __future__ import annotations

from pathlib import Path

from PIL import Image

from virtual_staining.data.dataset import PairedManifestDataset
from virtual_staining.data.manifest import DatasetManifest, ManifestRecord


def _make_image(path: Path) -> None:
    Image.new("RGB", (4, 4), color=(128, 64, 32)).save(path)


def test_paired_manifest_dataset_smoke(tmp_path: Path) -> None:
    (tmp_path / "splits" / "train").mkdir(parents=True)
    (tmp_path / "splits" / "val").mkdir(parents=True)
    Image.new("RGB", (32, 32), color=(128, 64, 32)).save(
        tmp_path / "splits" / "train" / "00000_source.tif"
    )
    Image.new("RGB", (32, 32), color=(128, 64, 32)).save(
        tmp_path / "splits" / "train" / "00000_target.tif"
    )
    Image.new("RGB", (32, 32), color=(128, 64, 32)).save(
        tmp_path / "splits" / "val" / "00001_source.tif"
    )
    Image.new("RGB", (32, 32), color=(128, 64, 32)).save(
        tmp_path / "splits" / "val" / "00001_target.tif"
    )

    manifest = DatasetManifest(
        records=(
            ManifestRecord(
                sample_id="00000",
                split="train",
                input_path=Path("splits/train/00000_source.tif"),
                target_path=Path("splits/train/00000_target.tif"),
                input_modality="label_free",
                target_modality="stained",
                x=0,
                y=0,
                width=32,
                height=32,
            ),
            ManifestRecord(
                sample_id="00001",
                split="val",
                input_path=Path("splits/val/00001_source.tif"),
                target_path=Path("splits/val/00001_target.tif"),
                input_modality="label_free",
                target_modality="stained",
                x=32,
                y=32,
                width=32,
                height=32,
            ),
        ),
        dataset_root=tmp_path,
    )

    dataset = PairedManifestDataset(manifest)
    assert len(dataset) == 2

    source, target = dataset[0]
    assert isinstance(source, Image.Image)
    assert isinstance(target, Image.Image)
    assert dataset.sample_ids == ["00000", "00001"]


def test_paired_manifest_dataset_getitem_image_size(tmp_path: Path) -> None:
    (tmp_path / "splits" / "train").mkdir(parents=True)
    Image.new("RGB", (64, 48)).save(tmp_path / "splits" / "train" / "a_source.tif")
    Image.new("RGB", (64, 48)).save(tmp_path / "splits" / "train" / "a_target.tif")
    record = ManifestRecord(
        sample_id="a",
        split="train",
        input_path=Path("splits/train/a_source.tif"),
        target_path=Path("splits/train/a_target.tif"),
        input_modality="label_free",
        target_modality="stained",
        x=0,
        y=0,
        width=64,
        height=48,
    )
    manifest = DatasetManifest(records=(record,), dataset_root=tmp_path)
    inp, tgt = PairedManifestDataset(manifest)[0]
    assert inp.size == (64, 48)
    assert tgt.size == (64, 48)


def test_paired_manifest_dataset_sample_ids_ordered(tmp_path: Path) -> None:
    (tmp_path / "splits" / "train").mkdir(parents=True)
    records = tuple(
        ManifestRecord(
            sample_id=sid,
            split="train",
            input_path=Path(f"splits/train/{sid}_source.tif"),
            target_path=Path(f"splits/train/{sid}_target.tif"),
            input_modality="label_free",
            target_modality="stained",
            x=0,
            y=0,
            width=32,
            height=32,
        )
        for sid in ["c", "a", "b"]
    )
    manifest = DatasetManifest(records=records, dataset_root=tmp_path)
    assert PairedManifestDataset(manifest).sample_ids == ["c", "a", "b"]
