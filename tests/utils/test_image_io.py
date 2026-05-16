from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tests.image_helpers import make_rgb_image, write_rgb_image
from virtual_staining.utils.image_io import (
    VALID_IMAGE_EXTENSIONS,
    load_rgb_image,
    open_rgb,
    to_float01,
)

# ---------------------------------------------------------------------------
# VALID_IMAGE_EXTENSIONS
# ---------------------------------------------------------------------------


def test_valid_extensions_contains_expected() -> None:
    assert ".png" in VALID_IMAGE_EXTENSIONS
    assert ".tif" in VALID_IMAGE_EXTENSIONS
    assert ".tiff" in VALID_IMAGE_EXTENSIONS


def test_valid_extensions_contains_jpg() -> None:
    assert ".jpg" in VALID_IMAGE_EXTENSIONS
    assert ".jpeg" in VALID_IMAGE_EXTENSIONS


# ---------------------------------------------------------------------------
# open_rgb
# ---------------------------------------------------------------------------


def test_open_rgb_returns_pil_image(tmp_path: Path) -> None:
    image_path = tmp_path / "img.png"
    write_rgb_image(image_path, size=(4, 4), color=(128, 64, 32))
    image = open_rgb(image_path)
    assert isinstance(image, Image.Image)
    assert image.mode == "RGB"


def test_open_rgb_raises_on_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        open_rgb(tmp_path / "nonexistent.png")


# ---------------------------------------------------------------------------
# load_rgb_image
# ---------------------------------------------------------------------------


def test_load_rgb_image_returns_uint8_array(tmp_path: Path) -> None:
    image_path = tmp_path / "img.png"
    write_rgb_image(image_path, size=(4, 4), color=(10, 20, 30))
    image = load_rgb_image(image_path)
    assert isinstance(image, np.ndarray)
    assert image.dtype == np.uint8
    assert image.shape == (4, 4, 3)


def test_load_rgb_image_correct_pixel_values(tmp_path: Path) -> None:
    image_path = tmp_path / "img.png"
    write_rgb_image(image_path, size=(4, 4), color=(100, 150, 200))
    image = load_rgb_image(image_path)
    np.testing.assert_array_equal(image[0, 0], np.array([100, 150, 200], dtype=np.uint8))


def test_load_rgb_image_raises_on_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_rgb_image(tmp_path / "nonexistent.png")


# ---------------------------------------------------------------------------
# to_float01
# ---------------------------------------------------------------------------


def test_to_float01_from_array() -> None:
    image = np.array([[[0, 128, 255]]], dtype=np.uint8)
    result = to_float01(image)
    assert result.dtype == np.float32
    assert result[0, 0, 0] == pytest.approx(0.0)
    assert result[0, 0, 1] == pytest.approx(128 / 255.0)
    assert result[0, 0, 2] == pytest.approx(1.0)


def test_to_float01_from_pil_image() -> None:
    image = make_rgb_image(size=(2, 2), color=(255, 0, 128))
    result = to_float01(image)
    assert result.dtype == np.float32
    assert result[0, 0, 0] == pytest.approx(1.0)
    assert result[0, 0, 1] == pytest.approx(0.0)
    assert result[0, 0, 2] == pytest.approx(128 / 255.0)
