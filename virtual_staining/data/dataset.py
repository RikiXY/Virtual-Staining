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
    ) -> None:
        self.manifest = manifest
        self.transform = transform

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> tuple[Any, Any]:
        record = self.manifest.records[idx]
        input_path = self.manifest.dataset_root / record.input_path
        target_path = self.manifest.dataset_root / record.target_path

        input_image = Image.open(input_path).convert("RGB")
        target_image = Image.open(target_path).convert("RGB")

        if self.transform is not None:
            input_image = self.transform(input_image)
            target_image = self.transform(target_image)

        return input_image, target_image

    @property
    def sample_ids(self) -> list[str]:
        """Return ordered list of sample IDs from the manifest."""
        return [record.sample_id for record in self.manifest.records]
