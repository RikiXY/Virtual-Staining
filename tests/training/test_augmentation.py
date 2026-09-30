from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from virtual_staining.config.training import AugmentationConfig, AugmentationIntensity
from virtual_staining.training.augmentation import (
    PairedAlbumentationsTransform,
    build_training_paired_transform,
    photometric_stream_seed,
)

_BASE = np.arange(16 * 16 * 3, dtype=np.uint8).reshape(16, 16, 3)


def _image() -> Image.Image:
    return Image.fromarray(_BASE, mode="RGB")


def _mask(columns: int) -> Image.Image:
    array = np.zeros((16, 16), dtype=np.uint8)
    array[:, :columns] = 255
    return Image.fromarray(array, mode="L")


def _transform(
    intensity: AugmentationIntensity = "light",
    *,
    seed: int | None = 123,
    photometric: tuple[str, ...] = (),
    targets: tuple[str, ...] = ("HE", "PAS"),
) -> PairedAlbumentationsTransform:
    return PairedAlbumentationsTransform(
        image_size=(16, 16),
        intensity=intensity,
        seed=seed,
        input_names=("LF", "AF"),
        target_names=targets,
        photometric_inputs=photometric,
    )


def _apply(transform: PairedAlbumentationsTransform, *, masks: bool = True):  # type: ignore[no-untyped-def]
    image = _image()
    return transform(
        {"LF": image, "AF": image},
        {"HE": image, "PAS": image},
        {"foreground_mask": {"HE": _mask(6), "PAS": _mask(6)}} if masks else {},
    )


@pytest.mark.parametrize("intensity", ["light", "medium", "strong"])
def test_named_transform_preserves_shape_range_and_mask_contract(
    intensity: AugmentationIntensity,
) -> None:
    inputs, targets, masks = _apply(_transform(intensity, photometric=("LF",)))

    assert tuple(inputs) == ("LF", "AF") and tuple(targets) == ("HE", "PAS")
    assert all(value.shape == (3, 16, 16) for value in (*inputs.values(), *targets.values()))
    assert tuple(masks["foreground_mask"]) == ("HE", "PAS")
    for mask in masks["foreground_mask"].values():
        assert mask.shape == (1, 16, 16)
        assert set(torch.unique(mask).tolist()) <= {0.0, 1.0}
    assert all(
        value.min() >= -1.0 and value.max() <= 1.0
        for value in (*inputs.values(), *targets.values())
    )


@pytest.mark.parametrize("seed", range(8))
def test_one_geometry_realization_covers_every_input_target_and_mask(seed: int) -> None:
    inputs, targets, masks = _apply(_transform("strong", seed=seed))

    # No photometric input: identical sources stay identical after the shared geometry.
    assert torch.equal(inputs["LF"], inputs["AF"])
    assert torch.equal(inputs["LF"], targets["HE"]) and torch.equal(targets["HE"], targets["PAS"])
    he, pas = masks["foreground_mask"]["HE"], masks["foreground_mask"]["PAS"]
    assert torch.equal(he, pas)
    # Masks move with the images (nearest-neighbour vs continuous interpolation aside).
    image = _transform("strong", seed=seed)(
        {"LF": _image(), "AF": _image()},
        {"HE": _mask(6).convert("RGB"), "PAS": _mask(6).convert("RGB")},
        {"foreground_mask": {"HE": _mask(6), "PAS": _mask(6)}},
    )[1]["HE"][0]
    agreement = ((image > 0) == (he[0] > 0.5)).float().mean()
    assert agreement > 0.9


def test_targets_and_masks_never_receive_photometric_transforms() -> None:
    changed = False
    for seed in range(20):
        inputs, targets, masks = _apply(_transform("strong", seed=seed, photometric=("LF",)))
        assert torch.equal(inputs["AF"], targets["HE"])
        assert torch.equal(targets["HE"], targets["PAS"])
        assert set(torch.unique(masks["foreground_mask"]["PAS"]).tolist()) <= {0.0, 1.0}
        changed |= not torch.equal(inputs["LF"], inputs["AF"])
    assert changed  # the selected input did receive photometric transforms


def test_each_photometric_input_has_its_own_stable_stream() -> None:
    assert photometric_stream_seed(7, "LF") == photometric_stream_seed(7, "LF")
    assert photometric_stream_seed(7, "LF") != photometric_stream_seed(7, "AF")
    assert photometric_stream_seed(7, "LF") != photometric_stream_seed(8, "LF")
    assert 0 <= photometric_stream_seed(7, "LF") < 2**32

    independent = False
    for seed in range(20):
        first = _apply(_transform("strong", seed=seed, photometric=("LF", "AF")))
        second = _apply(_transform("strong", seed=seed, photometric=("LF", "AF")))
        for left, right in zip(first[0].values(), second[0].values(), strict=True):
            assert torch.equal(left, right)  # same complete setup -> same realization
        independent |= not torch.equal(first[0]["LF"], first[0]["AF"])
    assert independent


def test_light_transform_keeps_identical_inputs_identical() -> None:
    inputs, targets, masks = _apply(_transform("light"), masks=False)

    assert torch.equal(inputs["LF"], inputs["AF"])
    assert torch.equal(inputs["LF"], targets["HE"])
    assert masks == {}


def test_one_target_uses_the_same_plural_contract() -> None:
    image = _image()
    _, targets, masks = _transform(targets=("HE",))(
        {"LF": image, "AF": image}, {"HE": image}, {"foreground_mask": {"HE": _mask(4)}}
    )

    assert tuple(targets) == ("HE",) and tuple(masks["foreground_mask"]) == ("HE",)


def test_transform_rejects_wrong_names_order_and_geometry() -> None:
    image = Image.new("RGB", (4, 4))
    transform = _transform()
    with pytest.raises(ValueError, match="exact configured order"):
        transform({"AF": image, "LF": image}, {"HE": image, "PAS": image}, {})
    with pytest.raises(ValueError, match="Targets must have exact configured order"):
        transform({"LF": image, "AF": image}, {"PAS": image, "HE": image}, {})
    with pytest.raises(ValueError, match="cover exactly the targets"):
        transform(
            {"LF": image, "AF": image},
            {"HE": image, "PAS": image},
            {"foreground_mask": {"HE": image}},
        )
    with pytest.raises(ValueError, match="not selected inputs"):
        _transform(photometric=("HE",))
    with pytest.raises(ValueError, match="square image_size"):
        PairedAlbumentationsTransform(
            image_size=(16, 8),
            intensity="light",
            seed=1,
            input_names=("LF",),
            target_names=("HE",),
            photometric_inputs=(),
        )


def test_disabled_augmentation_builds_no_transform() -> None:
    config = AugmentationConfig(enabled=False, expansion_factor=4, photometric_inputs=())

    assert config.effective_expansion_factor == 1
    assert (
        build_training_paired_transform(
            config, image_size=(16, 8), seed=1, input_names=("LF",), target_names=("HE",)
        )
        is None
    )
    enabled = AugmentationConfig(enabled=True, expansion_factor=4, photometric_inputs=())
    assert enabled.effective_expansion_factor == 4
    with pytest.raises(ValueError, match="must be resolved"):
        build_training_paired_transform(
            AugmentationConfig(enabled=True),
            image_size=(16, 16),
            seed=1,
            input_names=("LF",),
            target_names=("HE",),
        )
