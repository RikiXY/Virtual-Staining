from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image
from torchvision import transforms

from tests.manifest_helpers import manifest_metadata
from virtual_staining.data.dataset import PairedManifestDataset
from virtual_staining.data.manifest import DatasetManifest, ManifestRecord


def _write_record(root: Path, sample_id: str = "a", *, mask: bool = False) -> DatasetManifest:
    split = root / "splits" / "train"
    split.mkdir(parents=True, exist_ok=True)
    inputs = {}
    for name, color in (("LF", (10, 20, 30)), ("AF", (40, 50, 60))):
        path = Path(f"splits/train/{sample_id}__input__{name}.png")
        Image.new("RGB", (16, 16), color=color).save(root / path)
        inputs[name] = path
    target = Path(f"splits/train/{sample_id}__target.png")
    Image.new("RGB", (16, 16), color=(70, 80, 90)).save(root / target)
    mask_path = None
    if mask:
        mask_path = Path(f"splits/train/{sample_id}__foreground_mask.png")
        Image.new("L", (16, 16), color=255).save(root / mask_path)
    metadata = manifest_metadata(("LF", "AF"))
    return DatasetManifest(
        (ManifestRecord(sample_id, "set-1", "train", inputs, target, 0, 0, 16, 16, mask_path),),
        root,
        metadata,
    )


def test_dataset_returns_named_mapping_in_configured_order(tmp_path: Path) -> None:
    dataset = PairedManifestDataset(_write_record(tmp_path), input_names=("AF", "LF"))
    sample = dataset[0]
    assert tuple(sample) == ("inputs", "target", "masks")
    assert tuple(sample["inputs"]) == ("AF", "LF")
    assert all(isinstance(value, Image.Image) for value in sample["inputs"].values())
    assert sample["target"].size == (16, 16)


def test_dataset_transform_and_mask_mapping(tmp_path: Path) -> None:
    dataset = PairedManifestDataset(
        _write_record(tmp_path, mask=True),
        input_names=("LF", "AF"),
        transform=transforms.ToTensor(),
        include_foreground_mask=True,
    )
    sample = dataset[0]
    assert tuple(sample["inputs"]) == ("LF", "AF")
    assert all(value.shape == (3, 16, 16) for value in sample["inputs"].values())
    assert sample["target"].shape == (3, 16, 16)
    assert sample["masks"]["foreground_mask"].shape == (1, 16, 16)


def test_dataset_virtual_expansion_repeats_records(tmp_path: Path) -> None:
    dataset = PairedManifestDataset(_write_record(tmp_path), virtual_expansion_factor=3)
    assert len(dataset) == 3
    assert dataset[0]["inputs"]["LF"].getpixel((0, 0)) == dataset[2]["inputs"]["LF"].getpixel(
        (0, 0)
    )


def test_dataset_missing_foreground_mask_raises(tmp_path: Path) -> None:
    dataset = PairedManifestDataset(_write_record(tmp_path), include_foreground_mask=True)
    with pytest.raises(FileNotFoundError, match="Foreground mask"):
        dataset[0]


def test_dataset_rejects_unknown_input_name(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown input names"):
        PairedManifestDataset(_write_record(tmp_path), input_names=("missing",))
