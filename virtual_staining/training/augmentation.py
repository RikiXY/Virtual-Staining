from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

import albumentations as A
import cv2
import numpy as np
import torch
from PIL import Image

from virtual_staining.config.training import AugmentationConfig, AugmentationIntensity


def photometric_stream_seed(seed: int, modality: str) -> int:
    """Stable 32-bit seed of one input's photometric stream (SHA-256, never ``hash()``)."""
    digest = hashlib.sha256(f"photometric:{seed}:{modality}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


class PairedAlbumentationsTransform:
    """One geometry realization for every selected input, target and target mask.

    Images keep continuous interpolation and masks nearest-neighbour. Each input in
    ``photometric_inputs`` gets its own photometric stream seeded from the run seed and
    its name; targets and masks never receive photometric transforms.
    """

    def __init__(
        self,
        *,
        image_size: tuple[int, int],
        intensity: AugmentationIntensity,
        seed: int | None,
        input_names: tuple[str, ...],
        target_names: tuple[str, ...],
        photometric_inputs: tuple[str, ...],
    ) -> None:
        if not input_names or len(set(input_names)) != len(input_names):
            raise ValueError("input_names must be non-empty and unique")
        if not target_names or len(set(target_names)) != len(target_names):
            raise ValueError("target_names must be non-empty and unique")
        unknown = [name for name in photometric_inputs if name not in input_names]
        if unknown:
            raise ValueError(f"photometric_inputs {unknown} are not selected inputs")
        width, height = image_size
        if width != height:
            raise ValueError("paired augmentation requires a square image_size (RandomRotate90)")
        self.input_names = input_names
        self.target_names = target_names
        additional_targets = {f"input__{name}": "image" for name in input_names[1:]}
        additional_targets.update({f"target__{name}": "image" for name in target_names})
        additional_targets.update({f"mask__{name}": "mask" for name in target_names})
        self._geometry = A.Compose(
            _geometry_transforms(width=width, height=height, intensity=intensity),
            additional_targets=additional_targets,
            seed=seed,
        )
        photometric = _photometric_transforms(intensity)
        self._photometric = {
            name: A.Compose(
                _photometric_transforms(intensity),
                seed=None if seed is None else photometric_stream_seed(seed, name),
            )
            for name in (photometric_inputs if photometric else ())
        }

    def __call__(
        self,
        inputs: Mapping[str, Image.Image],
        targets: Mapping[str, Image.Image],
        masks: Mapping[str, Mapping[str, Image.Image]],
    ) -> tuple[
        dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, dict[str, torch.Tensor]]
    ]:
        if tuple(inputs) != self.input_names:
            raise ValueError(f"Inputs must have exact configured order {self.input_names}")
        if tuple(targets) != self.target_names:
            raise ValueError(f"Targets must have exact configured order {self.target_names}")
        unknown = [source for source in masks if source != "foreground_mask"]
        if unknown:
            raise ValueError(f"Unsupported mask sources {unknown}")
        target_masks = masks.get("foreground_mask", {})
        if target_masks and tuple(target_masks) != self.target_names:
            raise ValueError(f"Masks must cover exactly the targets {self.target_names}")
        data: dict[str, np.ndarray] = {"image": _pil_rgb_to_array(inputs[self.input_names[0]])}
        data.update(
            {f"input__{name}": _pil_rgb_to_array(inputs[name]) for name in self.input_names[1:]}
        )
        data.update({f"target__{name}": _pil_rgb_to_array(targets[name]) for name in targets})
        data.update(
            {
                f"mask__{name}": np.asarray(mask.convert("L"), dtype=np.uint8)
                for name, mask in target_masks.items()
            }
        )
        transformed = self._geometry(**data)
        arrays = {self.input_names[0]: transformed["image"]}
        arrays.update({name: transformed[f"input__{name}"] for name in self.input_names[1:]})
        for name, photometric in self._photometric.items():
            arrays[name] = photometric(image=arrays[name])["image"]
        return (
            {name: _rgb_array_to_normalized_tensor(arrays[name]) for name in self.input_names},
            {
                name: _rgb_array_to_normalized_tensor(transformed[f"target__{name}"])
                for name in self.target_names
            },
            (
                {
                    "foreground_mask": {
                        name: _mask_array_to_tensor(transformed[f"mask__{name}"])
                        for name in self.target_names
                    }
                }
                if target_masks
                else {}
            ),
        )


def build_training_paired_transform(
    config: AugmentationConfig,
    *,
    image_size: tuple[int, int],
    seed: int | None,
    input_names: tuple[str, ...],
    target_names: tuple[str, ...],
) -> PairedAlbumentationsTransform | None:
    """The paired training transform, or None when augmentation is disabled.

    ``config`` must be resolved (``RunConfig`` fills the effective ``photometric_inputs``).
    """
    if not config.enabled:
        return None
    if config.photometric_inputs is None:
        raise ValueError("augmentation.photometric_inputs must be resolved before use")
    return PairedAlbumentationsTransform(
        image_size=image_size,
        intensity=config.intensity,
        seed=seed,
        input_names=input_names,
        target_names=target_names,
        photometric_inputs=config.photometric_inputs,
    )


def _geometry_transforms(*, width: int, height: int, intensity: AugmentationIntensity) -> list[Any]:
    affine_by_intensity = {
        "light": {"scale": (0.98, 1.02), "translate": 0.01, "rotate": 3, "p": 0.25},
        "medium": {"scale": (0.95, 1.05), "translate": 0.03, "rotate": 7, "p": 0.40},
        "strong": {"scale": (0.90, 1.10), "translate": 0.06, "rotate": 12, "p": 0.65},
    }
    affine = affine_by_intensity[intensity]
    translate, rotate = float(affine["translate"]), float(affine["rotate"])
    return [
        A.Resize(
            height=height,
            width=width,
            interpolation=cv2.INTER_LINEAR,
            mask_interpolation=cv2.INTER_NEAREST,
            p=1.0,
        ),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(
            scale=affine["scale"],
            translate_percent={"x": (-translate, translate), "y": (-translate, translate)},
            rotate=(-rotate, rotate),
            interpolation=cv2.INTER_LINEAR,
            mask_interpolation=cv2.INTER_NEAREST,
            border_mode=cv2.BORDER_REFLECT_101,
            fill=0,
            fill_mask=0,
            keep_ratio=True,
            fit_output=False,
            balanced_scale=True,
            p=affine["p"],
        ),
    ]


def _photometric_transforms(intensity: AugmentationIntensity) -> list[Any]:
    if intensity == "light":
        return []
    limits = {
        "medium": (0.08, 0.25, (90, 110), (0.1, 0.6), (0.005, 0.02)),
        "strong": (0.15, 0.35, (85, 115), (0.2, 1.0), (0.01, 0.04)),
    }
    brightness, probability, gamma, blur, noise = limits[intensity]
    return [
        A.RandomBrightnessContrast(
            brightness_limit=(-brightness, brightness),
            contrast_limit=(-brightness, brightness),
            p=probability,
        ),
        A.RandomGamma(gamma_limit=gamma, p=probability),
        A.OneOf(
            [
                A.GaussianBlur(sigma_limit=blur, blur_limit=0, p=1.0),
                A.GaussNoise(std_range=noise, mean_range=(0.0, 0.0), p=1.0),
            ],
            p=probability,
        ),
    ]


def _pil_rgb_to_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _rgb_array_to_normalized_tensor(array: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1))).float().div(255.0)
    return tensor.sub(0.5).div(0.5)


def _mask_array_to_tensor(array: np.ndarray) -> torch.Tensor:
    array = np.ascontiguousarray(array).copy()
    array = array[None, :, :] if array.ndim == 2 else array.transpose(2, 0, 1)
    return torch.from_numpy(array).float().div(255.0)
