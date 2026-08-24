from __future__ import annotations

from collections.abc import Callable
from typing import Any

from PIL import Image
from torch.utils.data import Dataset

from virtual_staining.data.manifest import DatasetManifest


class PairedManifestDataset(Dataset):
    """A paired image dataset driven by a DatasetManifest."""

    def __init__(
        self,
        manifest: DatasetManifest,
        transform: Callable[..., Any] | None = None,
        mask_transform: Callable[..., Any] | None = None,
        paired_transform: Callable[
            [Image.Image, Image.Image, Image.Image | None],
            tuple[Any, Any, Any | None],
        ]
        | None = None,
        include_foreground_mask: bool = False,
        virtual_expansion_factor: int = 1,
    ) -> None:
        if virtual_expansion_factor < 1:
            raise ValueError("virtual_expansion_factor must be greater than or equal to 1")
        self.manifest = manifest
        self.transform = transform
        self.mask_transform = mask_transform
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
        input_path = self.manifest.dataset_root / record.input_path
        target_path = self.manifest.dataset_root / record.target_path

        input_image = Image.open(input_path).convert("RGB")
        target_image = Image.open(target_path).convert("RGB")

        mask_image: Image.Image | None = None
        if self.include_foreground_mask:
            if record.foreground_mask_path is None:
                raise FileNotFoundError(f"Foreground mask path is missing for {record.sample_id!r}")
            mask_path = self.manifest.dataset_root / record.foreground_mask_path
            mask_image = Image.open(mask_path).convert("L")

        if self.paired_transform is not None:
            input_image, target_image, mask_image = self.paired_transform(
                input_image,
                target_image,
                mask_image,
            )
        else:
            if self.transform is not None:
                input_image = self.transform(input_image)
                target_image = self.transform(target_image)
            if mask_image is not None and self.mask_transform is not None:
                mask_image = self.mask_transform(mask_image)

        if self.include_foreground_mask:
            assert mask_image is not None
            return input_image, target_image, {"foreground_mask": mask_image}

        return input_image, target_image

    @property
    def sample_ids(self) -> list[str]:
        """Return ordered list of sample IDs from the manifest."""
        return [record.sample_id for record in self.manifest.records]
