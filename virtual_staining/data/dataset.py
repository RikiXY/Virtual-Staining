from __future__ import annotations

import os
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from virtual_staining.data.manifest import DatasetManifest


class PairedHistologyDataset(Dataset):
    VALID_IMAGE_EXTENSIONS = {".tif", ".tiff", ".png"}

    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        # Pairs are built only once at the start, so during training
        # and validation we do not need to scan the directory every time.
        self.pairs = self._discover_pairs()

    def _discover_pairs(self):
        sources: dict[str, Path] = {}
        targets: dict[str, Path] = {}

        for filename in sorted(os.listdir(self.folder_path)):
            file_path = Path(self.folder_path) / filename

            if not file_path.is_file():
                continue

            if file_path.suffix.lower() not in self.VALID_IMAGE_EXTENSIONS:
                continue

            stem = file_path.stem

            if stem.startswith("mask_") or "_mask_" in stem:
                continue

            stem_lower = stem.lower()
            if stem_lower.endswith("_source"):
                sample_id = stem[: -len("_source")]
                if sample_id in sources:
                    raise ValueError(
                        f"Duplicate source file for sample ID {sample_id!r}: "
                        f"{sources[sample_id]} and {file_path}"
                    )
                sources[sample_id] = file_path
            elif stem_lower.endswith("_target"):
                sample_id = stem[: -len("_target")]
                if sample_id in targets:
                    raise ValueError(
                        f"Duplicate target file for sample ID {sample_id!r}: "
                        f"{targets[sample_id]} and {file_path}"
                    )
                targets[sample_id] = file_path

        return [(sources[sid], targets[sid]) for sid in sorted(set(sources) & set(targets))]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        source_path, target_path = self.pairs[idx]

        source_image = Image.open(source_path).convert("RGB")
        target_image = Image.open(target_path).convert("RGB")

        if self.transform:
            # The same transformation is applied to both source and target.
            source_image = self.transform(source_image)
            target_image = self.transform(target_image)

        return source_image, target_image


class PairedManifestDataset(Dataset):
    """A paired image dataset driven by a DatasetManifest."""

    def __init__(
        self,
        manifest: DatasetManifest,
        transform=None,
    ) -> None:
        self.manifest = manifest
        self.transform = transform

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx):
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
