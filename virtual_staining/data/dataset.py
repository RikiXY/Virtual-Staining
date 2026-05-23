from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
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
            mask_path = _find_foreground_mask_path(
                self.manifest.dataset_root,
                record.sample_id,
                record.input_path,
                record.target_path,
            )
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


def _find_foreground_mask_path(
    dataset_root: Path,
    sample_id: str,
    input_path: Path,
    target_path: Path,
) -> Path:
    input_path = dataset_root / input_path
    target_path = dataset_root / target_path
    candidates = [
        target_path.with_name(f"{sample_id}_foreground_mask{target_path.suffix}"),
        input_path.with_name(f"{sample_id}_foreground_mask{input_path.suffix}"),
    ]
    target_mask_name = target_path.name.replace("_target", "_foreground_mask")
    if target_mask_name != target_path.name:
        candidates.append(target_path.with_name(target_mask_name))
    input_mask_name = input_path.name.replace("_source", "_foreground_mask")
    if input_mask_name != input_path.name:
        candidates.append(input_path.with_name(input_mask_name))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Foreground mask not found for sample "
        f"{sample_id!r}. Expected one of: {', '.join(str(path) for path in candidates)}"
    )
