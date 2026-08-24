from __future__ import annotations

from collections.abc import Callable
from typing import Any

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
