from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

VALID_IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
)


def open_rgb(path: str | Path) -> Image.Image:
    """Opens an image file and returns it as an RGB PIL image."""
    image_path = Path(path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if not image_path.is_file():
        raise FileNotFoundError(f"Not a file: {image_path}")

    with Image.open(image_path) as img:
        return img.convert("RGB")


def load_rgb_image(path: str | Path) -> np.ndarray:
    """Loads an image from disk and returns it as a uint8 RGB array."""
    image_path = Path(path)

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as exc:
        raise RuntimeError(f"Could not open image: {image_path}") from exc

    return np.array(image)


def to_float01(image: np.ndarray | Image.Image) -> np.ndarray:
    """Converts an image to float32 [0, 1]."""
    return np.asarray(image, dtype=np.float32) / 255.0
