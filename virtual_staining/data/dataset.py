from __future__ import annotations

from collections.abc import Callable
from glob import glob
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset

from virtual_staining.data.manifest import DatasetManifest


class PairedManifestDataset(Dataset):
    def __init__(
        self,
        manifest: DatasetManifest,
        input_names: tuple[str, ...] | None = None,
        transform: Callable[[Any], Any] | None = None,
        paired_transform: Callable[..., tuple[dict[str, Any], Any, dict[str, Any]]] | None = None,
        include_foreground_mask: bool = False,
        virtual_expansion_factor: int = 1,
    ) -> None:
        if virtual_expansion_factor < 1:
            raise ValueError("virtual_expansion_factor must be greater than or equal to 1")
        names = manifest.metadata.input_modalities if input_names is None else tuple(input_names)
        if not names or len(set(names)) != len(names):
            raise ValueError("input_names must be non-empty and unique")
        if any(name not in manifest.metadata.input_modalities for name in names):
            raise ValueError(f"Unknown input names: {names}")
        self.manifest = manifest
        self.input_names = names
        self.transform = transform
        self.paired_transform = paired_transform
        self.include_foreground_mask = include_foreground_mask
        self.virtual_expansion_factor = virtual_expansion_factor

    def __len__(self) -> int:
        return len(self.manifest) * self.virtual_expansion_factor

    def __getitem__(self, idx: int) -> Any:
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        if not self.manifest.records:
            raise IndexError("Cannot index an empty PairedManifestDataset")
        record = self.manifest.records[idx % len(self.manifest.records)]
        inputs = {
            name: Image.open(self.manifest.dataset_root / record.input_paths[name]).convert("RGB")
            for name in self.input_names
        }
        target = Image.open(self.manifest.dataset_root / record.target_path).convert("RGB")
        masks: dict[str, Image.Image] = {}
        if self.include_foreground_mask:
            if record.foreground_mask_path is None:
                raise FileNotFoundError(f"Foreground mask path is missing for {record.sample_id!r}")
            masks["foreground_mask"] = Image.open(
                self.manifest.dataset_root / record.foreground_mask_path
            ).convert("L")
        if self.paired_transform is not None:
            inputs, target, masks = self.paired_transform(inputs, target, masks)
        elif self.transform is not None:
            inputs = {name: self.transform(image) for name, image in inputs.items()}
            target = self.transform(target)
            masks = {name: self.transform(mask) for name, mask in masks.items()}
        return {"inputs": inputs, "target": target, "masks": masks}

    @property
    def sample_ids(self) -> list[str]:
        return [record.sample_id for record in self.manifest.records]


SUPPORTED_IMAGE_SUFFIXES = frozenset({".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"})


def list_domain_images(directory: str | Path) -> tuple[Path, ...]:
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"Unpaired domain directory not found: {root}")
    paths = tuple(
        sorted(
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
        )
    )
    if not paths:
        raise ValueError(f"Unpaired domain directory contains no supported images: {root}")
    return paths


def resolve_domain_images(
    dataset_root: Path,
    domain_spec: Path,
    split: str,
) -> tuple[Path, ...]:
    """Resolve either a domain root or a ``{split}`` glob pattern."""
    raw_spec = str(domain_spec)
    if "{split}" not in raw_spec:
        root = domain_spec if domain_spec.is_absolute() else dataset_root / domain_spec
        return list_domain_images(root / split)

    rendered = raw_spec.replace("{split}", split)
    pattern = Path(rendered)
    if not pattern.is_absolute():
        pattern = dataset_root / pattern
    paths = tuple(
        sorted(
            Path(path)
            for path in glob(str(pattern), recursive=True)
            if Path(path).is_file() and Path(path).suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
        )
    )
    if not paths:
        raise ValueError(
            f"Unpaired domain pattern contains no supported images for split {split!r}: {pattern}"
        )
    return paths


class UnpairedImageDataset(Dataset):
    """Samples two image domains independently and permits unequal domain sizes."""

    def __init__(
        self,
        domain_a_paths: tuple[Any, ...],
        domain_b_paths: tuple[Any, ...],
        transform: Callable[[Any], Any] | None = None,
        *,
        random_pairing: bool = True,
    ) -> None:
        if not domain_a_paths or not domain_b_paths:
            raise ValueError("Both unpaired domains must contain at least one image")
        self.domain_a_paths = tuple(domain_a_paths)
        self.domain_b_paths = tuple(domain_b_paths)
        self.transform = transform
        self.random_pairing = random_pairing

    def __len__(self) -> int:
        return max(len(self.domain_a_paths), len(self.domain_b_paths))

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        path_a = self.domain_a_paths[idx % len(self.domain_a_paths)]
        if self.random_pairing:
            index_b = int(torch.randint(len(self.domain_b_paths), (1,)).item())
        else:
            index_b = idx % len(self.domain_b_paths)
        path_b = self.domain_b_paths[index_b]
        image_a: Any = Image.open(path_a).convert("RGB")
        image_b: Any = Image.open(path_b).convert("RGB")
        if self.transform is not None:
            image_a = self.transform(image_a)
            image_b = self.transform(image_b)
        return {
            "domain_a": image_a,
            "domain_b": image_b,
            "path_a": str(path_a),
            "path_b": str(path_b),
        }
