from __future__ import annotations

from pathlib import Path

from PIL import Image

from tests.image_helpers import write_rgb_pair
from tests.manifest_helpers import make_manifest_record
from virtual_staining.data.dataset import PairedManifestDataset
from virtual_staining.data.manifest import DatasetManifest


def test_paired_manifest_dataset_smoke(tmp_path: Path) -> None:
    (tmp_path / "splits" / "train").mkdir(parents=True)
    (tmp_path / "splits" / "val").mkdir(parents=True)
    for split, sample_id in [("train", "00000"), ("val", "00001")]:
        write_rgb_pair(
            tmp_path / "splits" / split,
            sample_id,
            size=(32, 32),
            ext=".tif",
            color=(128, 64, 32),
        )

    manifest = DatasetManifest(
        records=(
            make_manifest_record(
                "00000",
                "train",
                ext=".tif",
                x=0,
                y=0,
                width=32,
                height=32,
            ),
            make_manifest_record(
                "00001",
                "val",
                ext=".tif",
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
    write_rgb_pair(tmp_path / "splits" / "train", "a", size=(64, 48), ext=".tif")
    record = make_manifest_record(
        "a",
        "train",
        ext=".tif",
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
        make_manifest_record(
            sid,
            "train",
            ext=".tif",
            x=0,
            y=0,
            width=32,
            height=32,
        )
        for sid in ["c", "a", "b"]
    )
    manifest = DatasetManifest(records=records, dataset_root=tmp_path)
    assert PairedManifestDataset(manifest).sample_ids == ["c", "a", "b"]
