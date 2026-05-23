from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image
from torchvision import transforms

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


def test_paired_manifest_dataset_virtual_expansion_maps_to_original_records(
    tmp_path: Path,
) -> None:
    split_dir = tmp_path / "splits" / "train"
    split_dir.mkdir(parents=True)
    write_rgb_pair(split_dir, "a", size=(16, 16), ext=".png", color=(10, 20, 30))
    write_rgb_pair(split_dir, "b", size=(16, 16), ext=".png", color=(40, 50, 60))
    records = (
        make_manifest_record("a", "train", ext=".png", x=0, y=0, width=16, height=16),
        make_manifest_record("b", "train", ext=".png", x=16, y=0, width=16, height=16),
    )
    manifest = DatasetManifest(records=records, dataset_root=tmp_path)

    dataset = PairedManifestDataset(manifest, virtual_expansion_factor=3)

    assert len(dataset) == 6
    assert dataset.sample_ids == ["a", "b"]
    first_source, _ = dataset[0]
    wrapped_source, _ = dataset[2]
    assert first_source.getpixel((0, 0)) == (10, 20, 30)
    assert wrapped_source.getpixel((0, 0)) == first_source.getpixel((0, 0))


def test_paired_manifest_dataset_can_return_foreground_mask(tmp_path: Path) -> None:
    split_dir = tmp_path / "splits" / "train"
    split_dir.mkdir(parents=True)
    write_rgb_pair(split_dir, "00000_00000", size=(16, 16), ext=".png")
    Image.new("L", (16, 16), color=255).save(split_dir / "00000_00000_foreground_mask.png")
    record = make_manifest_record(
        "00000_00000",
        "train",
        ext=".png",
        x=0,
        y=0,
        width=16,
        height=16,
    )
    manifest = DatasetManifest(records=(record,), dataset_root=tmp_path)
    dataset = PairedManifestDataset(
        manifest,
        transform=transforms.ToTensor(),
        mask_transform=transforms.ToTensor(),
        include_foreground_mask=True,
    )

    source, target, masks = dataset[0]

    assert source.shape == target.shape
    assert masks["foreground_mask"].shape == (1, 16, 16)


def test_paired_manifest_dataset_uses_paired_transform_for_images_and_mask(
    tmp_path: Path,
) -> None:
    split_dir = tmp_path / "splits" / "train"
    split_dir.mkdir(parents=True)
    write_rgb_pair(split_dir, "00000_00000", size=(16, 16), ext=".png")
    Image.new("L", (16, 16), color=255).save(split_dir / "00000_00000_foreground_mask.png")
    record = make_manifest_record(
        "00000_00000",
        "train",
        ext=".png",
        x=0,
        y=0,
        width=16,
        height=16,
    )
    manifest = DatasetManifest(records=(record,), dataset_root=tmp_path)

    def _paired_transform(
        source: Image.Image,
        target: Image.Image,
        mask: Image.Image | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        assert mask is not None
        return (
            transforms.ToTensor()(source),
            transforms.ToTensor()(target),
            transforms.ToTensor()(mask),
        )

    dataset = PairedManifestDataset(
        manifest,
        paired_transform=_paired_transform,
        include_foreground_mask=True,
    )

    source, target, masks = dataset[0]

    assert source.shape == target.shape == (3, 16, 16)
    assert masks["foreground_mask"].shape == (1, 16, 16)


def test_paired_manifest_dataset_missing_foreground_mask_raises(tmp_path: Path) -> None:
    split_dir = tmp_path / "splits" / "train"
    split_dir.mkdir(parents=True)
    write_rgb_pair(split_dir, "00000_00000", size=(16, 16), ext=".png")
    record = make_manifest_record(
        "00000_00000",
        "train",
        ext=".png",
        x=0,
        y=0,
        width=16,
        height=16,
    )
    manifest = DatasetManifest(records=(record,), dataset_root=tmp_path)
    dataset = PairedManifestDataset(manifest, include_foreground_mask=True)

    with pytest.raises(FileNotFoundError, match="Foreground mask"):
        dataset[0]
