from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from virtual_staining.data.manifest import DatasetManifest


def _mask_tensor(mask: Image.Image, like: object) -> torch.Tensor:
    """A binary 0/1 ``1HW`` mask on the grid of the transformed image ``like``."""
    if isinstance(like, torch.Tensor):
        height, width = like.shape[-2:]
        mask = mask.resize((width, height), Image.Resampling.NEAREST)
    array = np.asarray(mask.convert("L"), dtype=np.uint8)
    return torch.from_numpy(array.copy()).float().div(255.0).unsqueeze(0)


class PairedManifestDataset(Dataset):
    """Corresponding supervision: selected named inputs and targets of each record.

    Items are ``{"inputs": {name: image}, "targets": {name: image}, "masks": masks}``
    with names in the selected order; ``masks`` is ``{"foreground_mask": {target: mask}}``
    when requested (every selected target must then have its own mask) and ``{}``
    otherwise. A mask of one target is never used for another.
    """

    def __init__(
        self,
        manifest: DatasetManifest,
        input_names: tuple[str, ...] | None = None,
        target_names: tuple[str, ...] | None = None,
        transform: Callable[[Any], Any] | None = None,
        paired_transform: Callable[..., tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]
        | None = None,
        include_foreground_mask: bool = False,
        virtual_expansion_factor: int = 1,
    ) -> None:
        if virtual_expansion_factor < 1:
            raise ValueError("virtual_expansion_factor must be greater than or equal to 1")
        metadata = manifest.metadata
        inputs = metadata.input_modalities if input_names is None else tuple(input_names)
        targets = metadata.target_modalities if target_names is None else tuple(target_names)
        for role, names, available in (
            ("input", inputs, metadata.input_modalities),
            ("target", targets, metadata.target_modalities),
        ):
            if not names or len(set(names)) != len(names):
                raise ValueError(f"{role}_names must be non-empty and unique")
            unknown = [name for name in names if name not in available]
            if unknown:
                raise ValueError(f"Unknown {role} names {unknown}; manifest has {available}")
        self.manifest = manifest
        self.input_names = inputs
        self.target_names = targets
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
        root = self.manifest.dataset_root
        inputs: dict[str, Any] = {
            name: Image.open(root / record.input_paths[name]).convert("RGB")
            for name in self.input_names
        }
        targets: dict[str, Any] = {
            name: Image.open(root / record.target_paths[name]).convert("RGB")
            for name in self.target_names
        }
        target_masks: dict[str, Any] = {}
        if self.include_foreground_mask:
            for name in self.target_names:
                mask_path = record.foreground_mask_paths[name]
                if mask_path is None:
                    raise FileNotFoundError(
                        f"Foreground mask of target {name!r} is missing for {record.sample_id!r}"
                    )
                target_masks[name] = Image.open(root / mask_path).convert("L")
        masks: dict[str, Any] = {"foreground_mask": target_masks} if target_masks else {}
        if self.paired_transform is not None:
            inputs, targets, masks = self.paired_transform(inputs, targets, masks)
        elif self.transform is not None:
            inputs = {name: self.transform(image) for name, image in inputs.items()}
            targets = {name: self.transform(image) for name, image in targets.items()}
            like = targets[self.target_names[0]]
            masks = {
                source: {name: _mask_tensor(mask, like) for name, mask in by_target.items()}
                for source, by_target in masks.items()
            }
        return {"inputs": inputs, "targets": targets, "masks": masks}

    @property
    def sample_ids(self) -> list[str]:
        return [record.sample_id for record in self.manifest.records]
