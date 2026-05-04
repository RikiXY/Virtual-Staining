from __future__ import annotations

from pathlib import Path

from PIL import Image
import pytest

from virtual_staining.data.dataset import PairedHistologyDataset


def _make_image(path: Path) -> None:
    Image.new("RGB", (4, 4), color=(128, 64, 32)).save(path)


@pytest.fixture
def dataset_dir(tmp_path):
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


def test_finds_correct_pair_count(dataset_dir):
    ds = PairedHistologyDataset(dataset_dir)
    assert len(ds) == 2


def test_source_and_target_are_matched(dataset_dir):
    ds = PairedHistologyDataset(dataset_dir)
    for source_path, target_path in ds.pairs:
        assert Path(source_path).stem.lower().endswith("_source")
        assert Path(target_path).stem.lower().endswith("_target")
        src_parts = Path(source_path).stem.split("_")
        tgt_parts = Path(target_path).stem.split("_")
        assert src_parts[0] == tgt_parts[0]
        assert src_parts[1] == tgt_parts[1]


def test_skips_mask_files(dataset_dir):
    ds = PairedHistologyDataset(dataset_dir)
    all_names = [Path(p).name for pair in ds.pairs for p in pair]
    assert all("mask_" not in name and "_mask_" not in name for name in all_names)


def test_skips_unmatched_files(dataset_dir):
    ds = PairedHistologyDataset(dataset_dir)
    keys = {Path(src).stem.split("_")[1] for src, _ in ds.pairs}
    assert "00003" not in keys
    assert "00004" not in keys


def test_empty_directory(tmp_path):
    ds = PairedHistologyDataset(tmp_path)
    assert len(ds) == 0


def test_getitem_returns_pil_images(dataset_dir):
    ds = PairedHistologyDataset(dataset_dir)
    source, target = ds[0]
    assert isinstance(source, Image.Image)
    assert isinstance(target, Image.Image)
    assert source.mode == "RGB"
    assert target.mode == "RGB"
