from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from virtual_staining.utils.image_io import (
    VALID_IMAGE_EXTENSIONS,
    open_rgb,
    load_rgb_image,
    to_float01,
)


def _save_rgb(path, color=(128, 64, 32)):
    Image.new("RGB", (4, 4), color=color).save(path)


# ---------------------------------------------------------------------------
# VALID_IMAGE_EXTENSIONS
# ---------------------------------------------------------------------------

def test_valid_extensions_contains_expected():
    assert ".png" in VALID_IMAGE_EXTENSIONS
    assert ".tif" in VALID_IMAGE_EXTENSIONS
    assert ".tiff" in VALID_IMAGE_EXTENSIONS


def test_valid_extensions_excludes_jpg():
    assert ".jpg" not in VALID_IMAGE_EXTENSIONS


# ---------------------------------------------------------------------------
# open_rgb
# ---------------------------------------------------------------------------

def test_open_rgb_returns_pil_image(tmp_path):
    p = tmp_path / "img.png"
    _save_rgb(p)
    img = open_rgb(p)
    assert isinstance(img, Image.Image)
    assert img.mode == "RGB"


def test_open_rgb_raises_on_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        open_rgb(tmp_path / "nonexistent.png")


# ---------------------------------------------------------------------------
# load_rgb_image
# ---------------------------------------------------------------------------

def test_load_rgb_image_returns_uint8_array(tmp_path):
    p = tmp_path / "img.png"
    _save_rgb(p, color=(10, 20, 30))
    arr = load_rgb_image(p)
    assert isinstance(arr, np.ndarray)
    assert arr.dtype == np.uint8
    assert arr.shape == (4, 4, 3)


def test_load_rgb_image_correct_pixel_values(tmp_path):
    p = tmp_path / "img.png"
    _save_rgb(p, color=(100, 150, 200))
    arr = load_rgb_image(p)
    assert arr[0, 0, 0] == 100
    assert arr[0, 0, 1] == 150
    assert arr[0, 0, 2] == 200


def test_load_rgb_image_raises_on_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_rgb_image(tmp_path / "nonexistent.png")


# ---------------------------------------------------------------------------
# to_float01
# ---------------------------------------------------------------------------

def test_to_float01_from_array():
    arr = np.array([[[0, 128, 255]]], dtype=np.uint8)
    result = to_float01(arr)
    assert result.dtype == np.float32
    assert result[0, 0, 0] == pytest.approx(0.0)
    assert result[0, 0, 1] == pytest.approx(128 / 255.0)
    assert result[0, 0, 2] == pytest.approx(1.0)


def test_to_float01_from_pil_image():
    img = Image.new("RGB", (2, 2), color=(255, 0, 128))
    result = to_float01(img)
    assert result.dtype == np.float32
    assert result[0, 0, 0] == pytest.approx(1.0)
    assert result[0, 0, 1] == pytest.approx(0.0)
