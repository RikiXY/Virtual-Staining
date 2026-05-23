from __future__ import annotations

from typing import Any

import numpy as np
import torch
from PIL import Image

from virtual_staining.training.config import AugmentationConfig, AugmentationIntensity


class PairedAlbumentationsTransform:
    """Apply aligned source/target geometry and source-only photometric augmentation."""

    def __init__(
        self,
        *,
        image_size: tuple[int, int],
        intensity: AugmentationIntensity,
        seed: int | None,
    ) -> None:
        A, cv2 = _import_albumentations()
        width, height = image_size
        self._geometry = A.Compose(
            _geometry_transforms(A, cv2, width=width, height=height, intensity=intensity),
            additional_targets={"target": "image"},
            seed=seed,
        )
        photometric = _photometric_transforms(A, intensity)
        self._photometric = (
            A.Compose(photometric, seed=None if seed is None else seed + 1) if photometric else None
        )

    def __call__(
        self,
        source: Image.Image,
        target: Image.Image,
        mask: Image.Image | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        data: dict[str, np.ndarray] = {
            "image": _pil_rgb_to_array(source),
            "target": _pil_rgb_to_array(target),
        }
        if mask is not None:
            data["mask"] = np.asarray(mask.convert("L"), dtype=np.uint8)

        transformed = self._geometry(**data)
        source_array = transformed["image"]
        target_array = transformed["target"]
        mask_array = transformed.get("mask")

        if self._photometric is not None:
            source_array = self._photometric(image=source_array)["image"]

        return (
            _rgb_array_to_normalized_tensor(source_array),
            _rgb_array_to_normalized_tensor(target_array),
            _mask_array_to_tensor(mask_array) if mask_array is not None else None,
        )


def build_training_paired_transform(
    config: AugmentationConfig,
    *,
    image_size: tuple[int, int],
    seed: int | None,
) -> PairedAlbumentationsTransform | None:
    if not config.enabled:
        return None
    return PairedAlbumentationsTransform(
        image_size=image_size,
        intensity=config.intensity,
        seed=seed,
    )


def _import_albumentations() -> tuple[Any, Any]:
    try:
        import albumentations as A
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "augmentation.enabled=true requires the 'albumentations' dependency. "
            "Install project dependencies before training with augmentation enabled."
        ) from exc
    return A, cv2


def _geometry_transforms(
    A: Any,
    cv2: Any,
    *,
    width: int,
    height: int,
    intensity: AugmentationIntensity,
) -> list[Any]:
    affine_by_intensity = {
        "light": {"scale": (0.98, 1.02), "translate": 0.01, "rotate": 3, "p": 0.25},
        "medium": {"scale": (0.95, 1.05), "translate": 0.03, "rotate": 7, "p": 0.40},
        "strong": {"scale": (0.90, 1.10), "translate": 0.06, "rotate": 12, "p": 0.65},
    }
    affine = affine_by_intensity[intensity]
    translate = float(affine["translate"])
    rotate = float(affine["rotate"])
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
    if intensity == "medium":
        return [
            A.RandomBrightnessContrast(
                brightness_limit=(-0.08, 0.08),
                contrast_limit=(-0.08, 0.08),
                p=0.35,
            ),
            A.RandomGamma(gamma_limit=(90, 110), p=0.25),
            A.OneOf(
                [
                    A.GaussianBlur(sigma_limit=(0.1, 0.6), blur_limit=0, p=1.0),
                    A.GaussNoise(std_range=(0.005, 0.02), mean_range=(0.0, 0.0), p=1.0),
                ],
                p=0.25,
            ),
        ]
    return [
        A.RandomBrightnessContrast(
            brightness_limit=(-0.15, 0.15),
            contrast_limit=(-0.15, 0.15),
            p=0.50,
        ),
        A.RandomGamma(gamma_limit=(85, 115), p=0.35),
        A.OneOf(
            [
                A.GaussianBlur(sigma_limit=(0.2, 1.0), blur_limit=0, p=1.0),
                A.GaussNoise(std_range=(0.01, 0.04), mean_range=(0.0, 0.0), p=1.0),
            ],
            p=0.45,
        ),
    ]


def _pil_rgb_to_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _rgb_array_to_normalized_tensor(array: np.ndarray) -> torch.Tensor:
    array = np.ascontiguousarray(array.transpose(2, 0, 1))
    tensor = torch.from_numpy(array).float().div(255.0)
    return tensor.sub(0.5).div(0.5)


def _mask_array_to_tensor(array: np.ndarray) -> torch.Tensor:
    array = np.ascontiguousarray(array)
    array = array[None, :, :] if array.ndim == 2 else array.transpose(2, 0, 1)
    return torch.from_numpy(array).float().div(255.0)
