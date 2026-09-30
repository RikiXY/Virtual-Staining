from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

from tests.manifest_helpers import make_manifest_record, manifest_metadata
from virtual_staining.data.dataset import PairedManifestDataset
from virtual_staining.data.manifest import DatasetManifest
from virtual_staining.models.io_contract import build_model_input_transform
from virtual_staining.training.helpers import unpack_batch

_COLORS = {"LF": 10, "AF": 40, "HE": 70, "PAS": 100}


def _write_record(
    root: Path,
    sample_id: str = "a",
    *,
    targets: tuple[str, ...] = ("HE", "PAS"),
    masks: tuple[str, ...] = (),
) -> DatasetManifest:
    split = root / "splits" / "train"
    split.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name in ("LF", "AF", *targets):
        role = "input" if name in ("LF", "AF") else "target"
        paths[name] = Path(f"splits/train/{sample_id}__{role}__{name}.png")
        Image.new("RGB", (16, 16), color=(_COLORS[name],) * 3).save(root / paths[name])
    mask_paths: dict[str, Path | None] = dict.fromkeys(targets)
    for index, name in enumerate(masks):
        mask_path = Path(f"splits/train/{sample_id}__foreground_mask__{name}.png")
        mask = Image.new("L", (16, 16), color=0)
        # Distinct per-target masks: the first target's covers one column more.
        mask.paste(255, (0, 0, 4 + index, 16))
        mask.save(root / mask_path)
        mask_paths[name] = mask_path
    record = make_manifest_record(
        sample_id,
        "train",
        x=0,
        y=0,
        input_paths={name: paths[name] for name in ("LF", "AF")},
        target_paths={name: paths[name] for name in targets},
        foreground_mask_paths=mask_paths,
    )
    return DatasetManifest((record,), root, manifest_metadata(("LF", "AF"), targets))


def test_dataset_returns_named_mappings_in_configured_order(tmp_path: Path) -> None:
    dataset = PairedManifestDataset(
        _write_record(tmp_path), input_names=("AF", "LF"), target_names=("PAS", "HE")
    )
    sample = dataset[0]
    assert tuple(sample) == ("inputs", "targets", "masks")
    assert tuple(sample["inputs"]) == ("AF", "LF")
    assert tuple(sample["targets"]) == ("PAS", "HE")
    assert sample["targets"]["PAS"].getpixel((0, 0)) == (100, 100, 100)
    assert sample["masks"] == {}
    assert "target" not in sample


def test_dataset_selects_input_and_output_subsets(tmp_path: Path) -> None:
    dataset = PairedManifestDataset(
        _write_record(tmp_path), input_names=("AF",), target_names=("PAS",)
    )
    sample = dataset[0]

    assert tuple(sample["inputs"]) == ("AF",)
    assert tuple(sample["targets"]) == ("PAS",)


def test_one_target_uses_the_same_plural_mapping(tmp_path: Path) -> None:
    sample = PairedManifestDataset(_write_record(tmp_path, targets=("HE",)))[0]

    assert tuple(sample["targets"]) == ("HE",)


def test_dataset_masks_are_target_specific(tmp_path: Path) -> None:
    dataset = PairedManifestDataset(
        _write_record(tmp_path, masks=("HE", "PAS")),
        transform=transforms.ToTensor(),
        include_foreground_mask=True,
    )
    sample = dataset[0]
    masks = sample["masks"]["foreground_mask"]

    assert tuple(masks) == ("HE", "PAS")
    assert masks["HE"].shape == masks["PAS"].shape == (1, 16, 16)
    assert masks["HE"].sum() == 4 * 16 and masks["PAS"].sum() == 5 * 16
    assert all(value.shape == (3, 16, 16) for value in sample["targets"].values())


def test_masks_follow_the_resized_model_grid_as_binary_values(tmp_path: Path) -> None:
    dataset = PairedManifestDataset(
        _write_record(tmp_path, masks=("HE", "PAS")),
        transform=build_model_input_transform((8, 8)),
        include_foreground_mask=True,
    )
    mask = dataset[0]["masks"]["foreground_mask"]["HE"]

    assert mask.shape == (1, 8, 8)
    assert set(mask.unique().tolist()) <= {0.0, 1.0}


def test_collated_batches_are_named_nchw_and_validated(tmp_path: Path) -> None:
    dataset = PairedManifestDataset(
        _write_record(tmp_path, masks=("HE", "PAS")),
        target_names=("PAS", "HE"),
        transform=build_model_input_transform((16, 16)),
        include_foreground_mask=True,
        virtual_expansion_factor=2,
    )
    batch = next(iter(DataLoader(dataset, batch_size=2)))

    inputs, targets, masks = unpack_batch(batch, torch.device("cpu"), ("LF", "AF"), ("PAS", "HE"))
    assert {name: tuple(t.shape) for name, t in targets.items()} == {
        "PAS": (2, 3, 16, 16),
        "HE": (2, 3, 16, 16),
    }
    assert tuple(masks["foreground_mask"]) == ("PAS", "HE")
    assert masks["foreground_mask"]["HE"].shape == (2, 1, 16, 16)
    assert tuple(inputs) == ("LF", "AF")
    with pytest.raises(TypeError, match="targets must match configured names"):
        unpack_batch(batch, torch.device("cpu"), ("LF", "AF"), ("HE", "PAS"))
    with pytest.raises(TypeError, match="inputs must match configured names"):
        unpack_batch(batch, torch.device("cpu"), ("AF", "LF"), ("PAS", "HE"))


def test_unpack_batch_rejects_singular_target_and_bad_shapes() -> None:
    rgb = torch.zeros(2, 3, 8, 8)
    with pytest.raises(TypeError, match="exactly inputs, targets, and masks"):
        unpack_batch(
            {"inputs": {"LF": rgb}, "target": rgb, "masks": {}},
            torch.device("cpu"),
            ("LF",),
            ("HE",),
        )
    with pytest.raises(TypeError, match="RGB NCHW"):
        unpack_batch(
            {"inputs": {"LF": rgb}, "targets": {"HE": rgb[:, :1]}, "masks": {}},
            torch.device("cpu"),
            ("LF",),
            ("HE",),
        )
    with pytest.raises(ValueError, match="batch/spatial shape"):
        unpack_batch(
            {"inputs": {"LF": rgb}, "targets": {"HE": torch.zeros(2, 3, 4, 4)}, "masks": {}},
            torch.device("cpu"),
            ("LF",),
            ("HE",),
        )
    with pytest.raises(TypeError, match="must map exactly the targets"):
        unpack_batch(
            {
                "inputs": {"LF": rgb},
                "targets": {"HE": rgb, "PAS": rgb},
                "masks": {"foreground_mask": {"HE": torch.zeros(2, 1, 8, 8)}},
            },
            torch.device("cpu"),
            ("LF",),
            ("HE", "PAS"),
        )


def test_dataset_virtual_expansion_repeats_records(tmp_path: Path) -> None:
    dataset = PairedManifestDataset(_write_record(tmp_path), virtual_expansion_factor=3)
    assert len(dataset) == 3
    assert dataset[0]["inputs"]["LF"].getpixel((0, 0)) == dataset[2]["inputs"]["LF"].getpixel(
        (0, 0)
    )


def test_dataset_missing_foreground_mask_names_the_target(tmp_path: Path) -> None:
    dataset = PairedManifestDataset(
        _write_record(tmp_path, masks=("HE",)), include_foreground_mask=True
    )
    with pytest.raises(FileNotFoundError, match="target 'PAS'"):
        dataset[0]


@pytest.mark.parametrize(
    ("input_names", "target_names", "match"),
    [
        (("missing",), None, "Unknown input names"),
        (None, ("IHC",), "Unknown target names"),
        (None, ("HE", "HE"), "non-empty and unique"),
        (None, (), "non-empty and unique"),
    ],
)
def test_dataset_rejects_unknown_or_duplicate_names(
    tmp_path: Path,
    input_names: tuple[str, ...] | None,
    target_names: tuple[str, ...] | None,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        PairedManifestDataset(
            _write_record(tmp_path), input_names=input_names, target_names=target_names
        )
