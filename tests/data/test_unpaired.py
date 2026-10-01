from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from tests.image_helpers import write_rgb_image
from virtual_staining.data.unpaired import UnpairedImageDataset, resolve_domain_images
from virtual_staining.models.io_contract import build_model_input_transform

_TRANSFORM = build_model_input_transform((8, 8))


def _images(root: Path, names: list[str]) -> list[Path]:
    return [write_rgb_image(root / name, size=(8, 8)) for name in names]


def test_directory_root_resolution_is_recursive_sorted_and_filters_extensions(
    tmp_path: Path,
) -> None:
    _images(tmp_path / "domains" / "a" / "train", ["b.png", "sub/a.tif", "c.jpg"])
    (tmp_path / "domains" / "a" / "train" / "notes.txt").write_text("x")
    _images(tmp_path / "domains" / "a" / "val", ["z.png"])

    paths = resolve_domain_images("domains/a", "train", tmp_path)

    assert [path.relative_to(tmp_path / "domains/a/train").as_posix() for path in paths] == [
        "b.png",
        "c.jpg",
        "sub/a.tif",
    ]


def test_split_pattern_resolution(tmp_path: Path) -> None:
    _images(tmp_path / "prepared" / "val" / "stained", ["x/2.tif", "1.tif", "skip.png"])

    paths = resolve_domain_images("prepared/{split}/stained/**/*.tif", "val", tmp_path)

    assert [path.name for path in paths] == ["1.tif", "2.tif"]


def test_absolute_domain_spec_ignores_dataset_root(tmp_path: Path) -> None:
    _images(tmp_path / "abs" / "test", ["1.png"])

    assert resolve_domain_images(str(tmp_path / "abs"), "test", tmp_path / "elsewhere")


def test_missing_domain_split_is_rejected(tmp_path: Path) -> None:
    _images(tmp_path / "a" / "train", ["1.png"])

    with pytest.raises(FileNotFoundError, match="no 'val' split"):
        resolve_domain_images("a", "val", tmp_path)


@pytest.mark.parametrize("spec", ["a", "a/{split}/*.png"])
def test_empty_domain_is_rejected(tmp_path: Path, spec: str) -> None:
    (tmp_path / "a" / "train").mkdir(parents=True)

    with pytest.raises(ValueError, match="matched no supported images"):
        resolve_domain_images(spec, "train", tmp_path)


def _dataset(tmp_path: Path, *, sizes: tuple[int, int], seed: int | None) -> UnpairedImageDataset:
    paths_a = _images(tmp_path / "a", [f"{index}.png" for index in range(sizes[0])])
    paths_b = _images(tmp_path / "b", [f"{index}.png" for index in range(sizes[1])])
    return UnpairedImageDataset(paths_a, paths_b, transform=_TRANSFORM, pairing_seed=seed)


def test_unequal_domains_use_max_length_and_wrap_domain_a(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, sizes=(2, 5), seed=None)

    assert len(dataset) == 5
    sample = dataset[4]
    assert Path(sample["path_a"]).name == "0.png"
    assert Path(sample["path_b"]).name == "4.png"
    assert sample["domain_a"].shape == (3, 8, 8)
    assert set(sample) == {"domain_a", "domain_b", "path_a", "path_b"}
    with pytest.raises(IndexError):
        dataset[5]


def test_empty_domain_list_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        UnpairedImageDataset([], [Path("b.png")], transform=_TRANSFORM)


def test_validation_traversal_is_deterministic_and_epoch_independent(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, sizes=(3, 4), seed=None)
    first = [dataset.domain_b_index(index) for index in range(len(dataset))]
    dataset.set_epoch(5)

    assert first == [0, 1, 2, 3]
    assert [dataset.domain_b_index(index) for index in range(len(dataset))] == first


def test_seeded_train_pairing_is_reproducible_and_varies_by_epoch(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, sizes=(2, 40), seed=11)
    same_seed = UnpairedImageDataset(
        dataset.paths_a, dataset.paths_b, transform=_TRANSFORM, pairing_seed=11
    )
    other_seed = UnpairedImageDataset(
        dataset.paths_a, dataset.paths_b, transform=_TRANSFORM, pairing_seed=12
    )

    def pairing(ds: UnpairedImageDataset, epoch: int) -> list[int]:
        ds.set_epoch(epoch)
        return [ds.domain_b_index(index) for index in range(len(ds))]

    epoch_0 = pairing(dataset, 0)
    assert epoch_0 == pairing(same_seed, 0)
    assert epoch_0 != pairing(dataset, 1)
    assert epoch_0 != pairing(other_seed, 0)
    assert pairing(dataset, 0) == epoch_0
    assert epoch_0 != list(range(40))


def test_seeded_pairing_is_identical_across_worker_counts(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, sizes=(3, 6), seed=5)
    dataset.set_epoch(2)

    def loaded(num_workers: int) -> list[str]:
        loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=num_workers)
        return [path for batch in loader for path in batch["path_b"]]

    serial = loaded(0)
    assert serial == loaded(2)
    assert serial == [str(dataset.paths_b[dataset.domain_b_index(i)]) for i in range(6)]


def test_training_loader_batches_carry_both_domains(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, sizes=(2, 3), seed=1)

    batch = next(iter(DataLoader(dataset, batch_size=3)))

    assert isinstance(batch["domain_a"], torch.Tensor)
    assert batch["domain_a"].shape == batch["domain_b"].shape == (3, 3, 8, 8)
    assert len(batch["path_a"]) == 3
