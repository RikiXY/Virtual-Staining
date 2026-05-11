from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from virtual_staining.data.dataset import PairedHistologyDataset, PairedManifestDataset
from virtual_staining.data.manifest import DatasetManifest, ManifestRecord


def _make_image(path: Path) -> None:
    Image.new("RGB", (4, 4), color=(128, 64, 32)).save(path)


@pytest.fixture()
def dataset_dir(tmp_path: Path) -> Path:
    # Two complete pairs
    _make_image(tmp_path / "00000_00000_source.png")
    _make_image(tmp_path / "00000_00000_target.png")
    _make_image(tmp_path / "00001_00001_source.png")
    _make_image(tmp_path / "00001_00001_target.png")
    # Mask files - must be skipped
    _make_image(tmp_path / "mask_00002_00002_source.png")
    _make_image(tmp_path / "00002_mask_00002_target.png")
    # Unmatched source - must be skipped (no matching target)
    _make_image(tmp_path / "00003_00003_source.png")
    # Unmatched target - must be skipped (no matching source)
    _make_image(tmp_path / "00004_00004_target.png")
    # Too few stem parts - must be skipped
    _make_image(tmp_path / "invalid.png")
    return tmp_path


def test_finds_correct_pair_count(dataset_dir: Path) -> None:
    dataset = PairedHistologyDataset(dataset_dir)
    assert len(dataset) == 2


def test_source_and_target_are_matched(dataset_dir: Path) -> None:
    dataset = PairedHistologyDataset(dataset_dir)
    for source_path, target_path in dataset.pairs:
        src_stem = Path(source_path).stem
        tgt_stem = Path(target_path).stem
        assert src_stem.lower().endswith("_source")
        assert tgt_stem.lower().endswith("_target")
        assert src_stem[: -len("_source")] == tgt_stem[: -len("_target")]


def test_skips_mask_files(dataset_dir: Path) -> None:
    dataset = PairedHistologyDataset(dataset_dir)
    all_names = [Path(path).name for pair in dataset.pairs for path in pair]
    assert all("mask_" not in name and "_mask_" not in name for name in all_names)


def test_skips_unmatched_files(dataset_dir: Path) -> None:
    dataset = PairedHistologyDataset(dataset_dir)
    sample_ids = {Path(src).stem[: -len("_source")] for src, _ in dataset.pairs}
    assert "00003_00003" not in sample_ids
    assert "00004_00004" not in sample_ids


def test_empty_directory(tmp_path: Path) -> None:
    dataset = PairedHistologyDataset(tmp_path)
    assert len(dataset) == 0


def test_getitem_returns_pil_images(dataset_dir: Path) -> None:
    dataset = PairedHistologyDataset(dataset_dir)
    source, target = dataset[0]
    assert isinstance(source, Image.Image)
    assert isinstance(target, Image.Image)
    assert source.mode == "RGB"
    assert target.mode == "RGB"


def test_multiunderscore_sample_id(tmp_path: Path) -> None:
    _make_image(tmp_path / "slide_A_patch_001_source.png")
    _make_image(tmp_path / "slide_A_patch_001_target.png")
    dataset = PairedHistologyDataset(tmp_path)
    assert len(dataset) == 1
    src, tgt = dataset.pairs[0]
    assert Path(src).stem == "slide_A_patch_001_source"
    assert Path(tgt).stem == "slide_A_patch_001_target"


def test_coordinate_sample_id(tmp_path: Path) -> None:
    _make_image(tmp_path / "00512_09216_source.tif")
    _make_image(tmp_path / "00512_09216_target.tif")
    dataset = PairedHistologyDataset(tmp_path)
    assert len(dataset) == 1
    src, tgt = dataset.pairs[0]
    assert Path(src).stem == "00512_09216_source"
    assert Path(tgt).stem == "00512_09216_target"


def test_duplicate_source_raises(tmp_path: Path) -> None:
    _make_image(tmp_path / "sample_source.tif")
    _make_image(tmp_path / "sample_source.png")
    _make_image(tmp_path / "sample_target.png")
    with pytest.raises(ValueError, match="Duplicate source"):
        PairedHistologyDataset(tmp_path)


def test_duplicate_target_raises(tmp_path: Path) -> None:
    _make_image(tmp_path / "sample_source.png")
    _make_image(tmp_path / "sample_target.tif")
    _make_image(tmp_path / "sample_target.png")
    with pytest.raises(ValueError, match="Duplicate target"):
        PairedHistologyDataset(tmp_path)


def test_paired_manifest_dataset_smoke(tmp_path: Path) -> None:
    Image.new("RGB", (32, 32), color=(128, 64, 32)).save(tmp_path / "00000_source.tif")
    Image.new("RGB", (32, 32), color=(128, 64, 32)).save(tmp_path / "00000_target.tif")
    Image.new("RGB", (32, 32), color=(128, 64, 32)).save(tmp_path / "00001_source.tif")
    Image.new("RGB", (32, 32), color=(128, 64, 32)).save(tmp_path / "00001_target.tif")

    manifest = DatasetManifest(
        records=(
            ManifestRecord(
                sample_id="00000",
                split="train",
                input_path=Path("00000_source.tif"),
                target_path=Path("00000_target.tif"),
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
                input_path=Path("00001_source.tif"),
                target_path=Path("00001_target.tif"),
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
    records = tuple(
        ManifestRecord(
            sample_id=sid,
            split="train",
            input_path=Path(f"{sid}_source.tif"),
            target_path=Path(f"{sid}_target.tif"),
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
