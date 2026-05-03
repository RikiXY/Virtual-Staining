from __future__ import annotations

import os
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


class PairedHistologyDataset(Dataset):
    VALID_IMAGE_EXTENSIONS = {".tif", ".tiff", ".png"}

    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        # Pairs are built only once at the start, so during training
        # and validation we do not need to scan the directory every time.
        self.pairs = self._discover_pairs()

    def _discover_pairs(self):
        grouped = {}

        for filename in sorted(os.listdir(self.folder_path)):
            file_path = Path(self.folder_path) / filename

            if not file_path.is_file():
                continue

            suffix = Path(filename).suffix.lower()
            if suffix not in self.VALID_IMAGE_EXTENSIONS:
                continue

            stem = Path(filename).stem

            if stem.startswith("mask_") or "_mask_" in stem:
                continue

            parts = stem.split("_")
            if len(parts) < 3:
                continue

            key = (parts[0], parts[1])
            grouped.setdefault(key, []).append(file_path)

        samples = []
        for key in sorted(grouped):
            files = grouped[key]

            source_path = None
            target_path = None

            for file_path in files:
                stem = Path(file_path).stem.lower()

                if stem.endswith("_source"):
                    source_path = file_path
                elif stem.endswith("_target"):
                    target_path = file_path

            if source_path is None or target_path is None:
                continue

            samples.append((source_path, target_path))

        return samples

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
