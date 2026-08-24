from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from PIL import Image

from virtual_staining.training.config import AugmentationConfig, AugmentationIntensity


class PairedAlbumentationsTransform:
    def __init__(
        self,
        *,
        image_size: tuple[int, int],
        intensity: AugmentationIntensity,
        seed: int | None,
        input_names: tuple[str, ...],
        reference_modality: str,
    ) -> None:
        A, cv2 = _import_albumentations()
        if not input_names or len(set(input_names)) != len(input_names):
            raise ValueError("input_names must be non-empty and unique")
        self.input_names = input_names
        self.reference_modality = reference_modality
        width, height = image_size
        additional_targets = {"target": "image", "mask__foreground_mask": "mask"}
        additional_targets.update({f"input__{name}": "image" for name in input_names[1:]})
        self._geometry = A.Compose(
            _geometry_transforms(A, cv2, width=width, height=height, intensity=intensity),
            additional_targets=additional_targets,
            seed=seed,
        )
        photometric = _photometric_transforms(A, intensity)
        self._photometric = (
            A.Compose(photometric, seed=None if seed is None else seed + 1) if photometric else None
        )

    def __call__(
        self,
        inputs: Mapping[str, Image.Image],
        target: Image.Image,
        masks: Mapping[str, Image.Image],
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict[str, torch.Tensor]]:
        if tuple(inputs) != self.input_names:
            raise ValueError(f"Inputs must have exact configured order {self.input_names}")
        data: dict[str, np.ndarray] = {
            "image": _pil_rgb_to_array(inputs[self.input_names[0]]),
            "target": _pil_rgb_to_array(target),
        }
        data.update(
            {f"input__{name}": _pil_rgb_to_array(inputs[name]) for name in self.input_names[1:]}
        )
        for name, mask in masks.items():
            data[f"mask__{name}"] = np.asarray(mask.convert("L"), dtype=np.uint8)
        transformed = self._geometry(**data)
        arrays = {self.input_names[0]: transformed["image"]}
        arrays.update({name: transformed[f"input__{name}"] for name in self.input_names[1:]})
        if self._photometric is not None and self.reference_modality in arrays:
            arrays[self.reference_modality] = self._photometric(
                image=arrays[self.reference_modality]
            )["image"]
        return (
            {name: _rgb_array_to_normalized_tensor(arrays[name]) for name in self.input_names},
            _rgb_array_to_normalized_tensor(transformed["target"]),
            {name: _mask_array_to_tensor(transformed[f"mask__{name}"]) for name in masks},
        )


def build_training_paired_transform(
    config: AugmentationConfig,
    *,
    image_size: tuple[int, int],
    seed: int | None,
    input_names: tuple[str, ...],
    reference_modality: str,
) -> PairedAlbumentationsTransform | None:
    if not config.enabled:
        return None
    return PairedAlbumentationsTransform(
        image_size=image_size,
        intensity=config.intensity,
        seed=seed,
        input_names=input_names,
        reference_modality=reference_modality,
    )


def _import_albumentations() -> tuple[Any, Any]:
    try:
        import albumentations as A
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "augmentation.enabled=true requires the 'albumentations' dependency."
        ) from exc
    return A, cv2


def _geometry_transforms(
    A: Any, cv2: Any, *, width: int, height: int, intensity: AugmentationIntensity
) -> list[Any]:
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


def _photometric_transforms(A: Any, intensity: AugmentationIntensity) -> list[Any]:
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
