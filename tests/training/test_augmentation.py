from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from virtual_staining.config.training import AugmentationIntensity
from virtual_staining.training.augmentation import PairedAlbumentationsTransform

pytest.importorskip("albumentations")


@pytest.mark.parametrize("intensity", ["light", "medium", "strong"])
def test_named_transform_preserves_shape_range_and_mask_contract(
    intensity: AugmentationIntensity,
) -> None:
    base = np.arange(16 * 16 * 3, dtype=np.uint8).reshape(16, 16, 3)
    image = Image.fromarray(base, mode="RGB")
    mask = Image.fromarray(np.full((16, 16), 255, dtype=np.uint8), mode="L")
    transform = PairedAlbumentationsTransform(
        image_size=(16, 16),
        intensity=intensity,
        seed=123,
        input_names=("LF", "AF"),
        reference_modality="LF",
    )
    inputs, target, masks = transform({"LF": image, "AF": image}, image, {"foreground_mask": mask})
    assert tuple(inputs) == ("LF", "AF")
    assert all(value.shape == (3, 16, 16) for value in inputs.values())
    assert target.shape == (3, 16, 16)
    assert masks["foreground_mask"].shape == (1, 16, 16)
    assert set(torch.unique(masks["foreground_mask"]).tolist()) <= {0.0, 1.0}
    assert all(value.min() >= -1.0 and value.max() <= 1.0 for value in (*inputs.values(), target))


def test_light_transform_keeps_identical_inputs_identical() -> None:
    base = np.arange(16 * 16 * 3, dtype=np.uint8).reshape(16, 16, 3)
    image = Image.fromarray(base, mode="RGB")
    transform = PairedAlbumentationsTransform(
        image_size=(16, 16),
        intensity="light",
        seed=123,
        input_names=("LF", "AF"),
        reference_modality="LF",
    )
    inputs, target, _ = transform({"LF": image, "AF": image}, image, {})
    assert torch.equal(inputs["LF"], inputs["AF"])
    assert torch.equal(inputs["LF"], target)


def test_transform_rejects_wrong_input_order() -> None:
    image = Image.new("RGB", (4, 4))
    transform = PairedAlbumentationsTransform(
        image_size=(4, 4),
        intensity="light",
        seed=1,
        input_names=("LF", "AF"),
        reference_modality="LF",
    )
    with pytest.raises(ValueError, match="exact configured order"):
        transform({"AF": image, "LF": image}, image, {})
