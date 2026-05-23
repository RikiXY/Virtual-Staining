from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from virtual_staining.training.augmentation import PairedAlbumentationsTransform
from virtual_staining.training.config import AugmentationIntensity

pytest.importorskip("albumentations")


@pytest.mark.parametrize("intensity", ["light", "medium", "strong"])
def test_paired_albumentations_transform_preserves_shape_range_and_mask_contract(
    intensity: AugmentationIntensity,
) -> None:
    base = np.arange(16 * 16 * 3, dtype=np.uint8).reshape(16, 16, 3)
    image = Image.fromarray(base, mode="RGB")
    mask = Image.fromarray(np.full((16, 16), 255, dtype=np.uint8), mode="L")
    transform = PairedAlbumentationsTransform(
        image_size=(16, 16),
        intensity=intensity,
        seed=123,
    )

    source, target, transformed_mask = transform(image, image, mask)

    assert source.shape == target.shape == (3, 16, 16)
    assert transformed_mask is not None
    assert transformed_mask.shape == (1, 16, 16)
    assert source.min() >= -1.0
    assert source.max() <= 1.0
    assert target.min() >= -1.0
    assert target.max() <= 1.0


def test_light_paired_albumentations_transform_keeps_identical_pair_identical() -> None:
    base = np.arange(16 * 16 * 3, dtype=np.uint8).reshape(16, 16, 3)
    image = Image.fromarray(base, mode="RGB")
    transform = PairedAlbumentationsTransform(
        image_size=(16, 16),
        intensity="light",
        seed=123,
    )

    source, target, _ = transform(image, image, None)

    assert torch.equal(source, target)
