from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image, TiffImagePlugin

from tests.image_helpers import make_rgb_image, write_rgb_image
from virtual_staining.utils.image_io import (
    SUPPORTED_IMAGE_BACKENDS,
    VALID_IMAGE_EXTENSIONS,
    ImageMetadata,
    PillowRegionImageReader,
    convert_to_pyramidal_tiff,
    load_grayscale_image,
    load_rgb_image,
    open_image_reader,
    open_rgb,
    read_full_image,
    read_image_metadata,
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
# RegionImageReader
# ---------------------------------------------------------------------------


def test_region_image_reader_reports_metadata(tmp_path: Path) -> None:
    image_path = tmp_path / "img.png"
    write_rgb_image(image_path, size=(8, 6), color=(10, 20, 30))

    reader = PillowRegionImageReader(image_path)

    assert reader.size == (8, 6)
    assert reader.metadata.width == 8
    assert reader.metadata.height == 6


def test_open_image_reader_returns_default_region_reader(tmp_path: Path) -> None:
    image_path = tmp_path / "img.png"
    write_rgb_image(image_path, size=(8, 6), color=(10, 20, 30))

    reader = open_image_reader(image_path)

    assert isinstance(reader, PillowRegionImageReader)
    assert reader.size == (8, 6)


@pytest.mark.parametrize("backend", sorted(SUPPORTED_IMAGE_BACKENDS))
def test_open_image_reader_accepts_supported_backends(tmp_path: Path, monkeypatch, backend) -> None:
    from virtual_staining.utils import image_io

    reader = object()
    monkeypatch.setattr(image_io, "PillowRegionImageReader", lambda _path: reader)
    monkeypatch.setattr(image_io, "OpenSlideRegionImageReader", lambda _path: reader)
    monkeypatch.setattr(image_io, "detect_openslide_format", lambda _path: None)
    assert open_image_reader(tmp_path / "image.png", backend=backend) is reader


def test_open_image_reader_rejects_unknown_backend_before_opening(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="backend must be auto, pillow, or openslide"):
        open_image_reader(tmp_path / "missing.png", backend="unknown")


def test_region_image_reader_reads_bgr_region(tmp_path: Path) -> None:
    image_path = tmp_path / "img.png"
    write_rgb_image(image_path, size=(8, 6), color=(10, 20, 30))

    reader = PillowRegionImageReader(image_path)
    region = reader.read_region(2, 1, 3, 2)

    assert region.shape == (2, 3, 3)
    np.testing.assert_array_equal(region[0, 0], np.array([30, 20, 10], dtype=np.uint8))


def test_region_image_reader_pads_out_of_bounds_with_white(tmp_path: Path) -> None:
    image_path = tmp_path / "img.png"
    write_rgb_image(image_path, size=(4, 4), color=(10, 20, 30))

    reader = PillowRegionImageReader(image_path)
    region = reader.read_region(-1, -1, 3, 3)

    assert region.shape == (3, 3, 3)
    np.testing.assert_array_equal(region[0, 0], np.array([255, 255, 255], dtype=np.uint8))
    np.testing.assert_array_equal(region[1, 1], np.array([30, 20, 10], dtype=np.uint8))


def test_region_image_reader_reads_scaled_preview(tmp_path: Path) -> None:
    image_path = tmp_path / "img.png"
    write_rgb_image(image_path, size=(8, 6), color=(10, 20, 30))

    reader = PillowRegionImageReader(image_path)
    preview = reader.read_preview(0.5)

    assert preview.shape == (3, 4, 3)


def test_image_io_metadata_and_full_read_use_reader_contract(tmp_path: Path) -> None:
    image_path = tmp_path / "img.png"
    write_rgb_image(image_path, size=(8, 6), color=(10, 20, 30))

    metadata = read_image_metadata(image_path, backend="pillow")
    image = read_full_image(image_path, backend="pillow")

    assert metadata == ImageMetadata(
        width=8,
        height=6,
        level_dimensions=((8, 6),),
        level_downsamples=(1.0,),
    )
    assert image.shape == (6, 8, 3)
    np.testing.assert_array_equal(image[0, 0], np.array([30, 20, 10], dtype=np.uint8))


def test_load_grayscale_image_returns_uint8(tmp_path: Path) -> None:
    image_path = tmp_path / "mask.png"
    Image.new("L", (4, 3), color=127).save(image_path)

    image = load_grayscale_image(image_path)

    assert image.dtype == np.uint8
    assert image.shape == (3, 4)
    assert image[0, 0] == 127


def test_convert_to_pyramidal_tiff_preserves_dimensions(tmp_path: Path) -> None:
    source = tmp_path / "source.tif"
    output = tmp_path / "output.tif"
    write_rgb_image(source, size=(320, 288), color=(10, 20, 30))

    convert_to_pyramidal_tiff(source, output)

    metadata = read_image_metadata(output, backend="openslide")
    assert (metadata.width, metadata.height) == (320, 288)
    assert metadata.level_count > 1


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


def test_auto_reader_uses_openslide_for_recognized_formats(monkeypatch: pytest.MonkeyPatch) -> None:
    from virtual_staining.utils import image_io

    reader = object()
    monkeypatch.setattr(image_io.openslide.OpenSlide, "detect_format", lambda path: "generic-tiff")
    monkeypatch.setattr(image_io, "OpenSlideRegionImageReader", lambda path: reader)
    assert open_image_reader("slide.tif") is reader


@pytest.mark.parametrize("error_type", [ImportError, OSError, RuntimeError])
def test_auto_reader_propagates_broken_openslide(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error_type: type[Exception]
) -> None:
    from virtual_staining.utils import image_io

    image_path = tmp_path / "image.png"
    write_rgb_image(image_path)

    def fail(path: str) -> None:
        raise error_type("broken OpenSlide runtime")

    monkeypatch.setattr(image_io.openslide.OpenSlide, "detect_format", fail)
    with pytest.raises(error_type, match="broken OpenSlide runtime"):
        open_image_reader(image_path)


def test_explicit_openslide_rejects_unsupported_image(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    write_rgb_image(image_path)
    with pytest.raises(ValueError, match="OpenSlide does not support"):
        open_image_reader(image_path, backend="openslide")


@pytest.mark.parametrize(
    ("suffix", "mode"),
    [
        (".png", "RGB"),
        (".png", "L"),
        (".png", "1"),
        (".png", "P"),
        (".jpg", "RGB"),
        (".jpeg", "L"),
        (".tif", "RGB"),
        (".tiff", "RGB"),
    ],
)
def test_native_conversion_pixels_and_tiff_contract(tmp_path: Path, suffix: str, mode: str) -> None:
    from virtual_staining.applications.convert import convert_images

    source = tmp_path / f"source{suffix}"
    y, x = np.indices((288, 640))
    pixels = np.stack((x % 256, y % 256, (x + y) % 256), axis=-1).astype(np.uint8)
    image = Image.fromarray(pixels).convert(mode)
    image.save(source, dpi=(300, 150))
    with Image.open(source) as decoded:
        expected = np.array(decoded.convert("RGB"))
    (output,) = convert_images((source,), tmp_path / "output")
    with Image.open(output) as tiff:
        assert isinstance(tiff, TiffImagePlugin.TiffImageFile)
        np.testing.assert_array_equal(np.array(tiff), expected)
        assert tiff.tag_v2[259] == 5  # LZW
        assert tiff.tag_v2[322] == tiff.tag_v2[323] == 256
        assert tiff.tag_v2[277] == 3
        assert tiff.tag_v2[258] == (8, 8, 8)
    with output.open("rb") as handle:
        assert handle.read(4) in (b"II+\x00", b"MM\x00+")
    reader = open_image_reader(output, backend="openslide")
    try:
        assert reader.size == (640, 288)
        assert reader.metadata.level_dimensions == ((640, 288), (320, 144), (160, 72))
        assert reader.metadata.mpp_x is None
        assert reader.metadata.mpp_y is None
        np.testing.assert_array_equal(reader.read_full()[:, :, ::-1], expected)
        assert reader.read_preview(0.25).shape == (72, 160, 3)
    finally:
        reader.close()


@pytest.mark.parametrize("suffix", [".jpg", ".png"])
@pytest.mark.parametrize("orientation", range(1, 9))
def test_native_conversion_applies_exif_orientation(
    tmp_path: Path, suffix: str, orientation: int
) -> None:
    from PIL import ImageOps

    source = tmp_path / f"source{suffix}"
    pixels = np.zeros((48, 80, 3), dtype=np.uint8)
    pixels[:24, :40] = (230, 10, 20)
    pixels[24:, :40] = (10, 220, 30)
    pixels[:24, 40:] = (20, 30, 210)
    image = Image.fromarray(pixels)
    exif = Image.Exif()
    exif[274] = orientation
    exif[40962] = 9999  # Stale EXIF geometry must not override the decoded dimensions.
    image.save(source, exif=exif)
    with Image.open(source) as decoded:
        expected = np.array(ImageOps.exif_transpose(decoded).convert("RGB"))
    output = tmp_path / "output.tif"
    convert_to_pyramidal_tiff(source, output)
    actual = read_full_image(output, backend="openslide")[:, :, ::-1]
    np.testing.assert_array_equal(actual, expected)
    with Image.open(output) as tiff:
        assert tiff.getexif().get(274, 1) == 1


@pytest.mark.parametrize(
    "variant",
    [
        "RGBA",
        "LA",
        "palette_alpha",
        "rgb_key",
        "gray_key",
        "16bit",
        "CMYK",
        "animated",
        "orientation",
    ],
)
def test_conversion_rejects_unsupported_source_semantics(tmp_path: Path, variant: str) -> None:
    from virtual_staining.applications.convert import convert_images

    source = tmp_path / ("source.jpg" if variant in {"CMYK", "orientation"} else "source.png")
    options = {}
    if variant in {"RGBA", "LA", "CMYK"}:
        image = Image.new(variant, (40, 30))
    elif variant == "16bit":
        image = Image.fromarray(np.full((30, 40), 40000, dtype=np.uint16))
    elif variant in {"palette_alpha", "rgb_key", "gray_key"}:
        mode = {"palette_alpha": "P", "rgb_key": "RGB", "gray_key": "L"}[variant]
        image = Image.new(mode, (40, 30))
        options["transparency"] = (0, 0, 0) if mode == "RGB" else 0
    else:
        image = Image.new("RGB", (40, 30))
        if variant == "animated":
            options.update(save_all=True, append_images=[Image.new("RGB", (40, 30), "red")])
        else:
            exif = Image.Exif()
            exif[274] = 9
            options["exif"] = exif
    image.save(source, **options)
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="[Uu]nsupported|orientation") as error:
        convert_images((source,), output)
    assert str(source) in str(error.value)
    assert list(output.iterdir()) == []


@pytest.mark.parametrize("suffix", [".png", ".jpg"])
@pytest.mark.parametrize("damage", ["invalid", "truncated"])
def test_native_conversion_rejects_broken_images(tmp_path: Path, suffix: str, damage: str) -> None:
    from virtual_staining.applications.convert import convert_images

    source = tmp_path / f"source{suffix}"
    image = Image.new("RGB", (640, 288), (10, 100, 200))
    image.save(source)
    data = source.read_bytes()
    source.write_bytes(b"not an image" if damage == "invalid" else data[: len(data) // 2])
    output = tmp_path / "output"
    with pytest.raises((ValueError, RuntimeError, OSError)):
        convert_images((source,), output)
    assert list(output.iterdir()) == []


@pytest.mark.parametrize("suffix", [".png", ".jpg"])
def test_conversion_rejects_mislabeled_sources(tmp_path: Path, suffix: str) -> None:
    source = tmp_path / f"source{suffix}"
    Image.new("RGB", (20, 10)).save(source, format="BMP")
    with pytest.raises(ValueError, match="decoded BMP"):
        convert_to_pyramidal_tiff(source, tmp_path / "output.tif")


def test_conversion_rejects_16bit_rgb_png_without_truncating(tmp_path: Path) -> None:
    import pyvips

    source = tmp_path / "source.png"
    image = pyvips.Image.black(40, 30, bands=3).cast("ushort") + 40000
    image.cast("ushort").pngsave(str(source), bitdepth=16)
    with pytest.raises(ValueError, match="Unsupported bit depth"):
        convert_to_pyramidal_tiff(source, tmp_path / "output.tif")


def test_conversion_rejects_even_opaque_alpha(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    Image.new("RGBA", (20, 10), (10, 20, 30, 255)).save(source)
    with pytest.raises(ValueError, match="Transparency/alpha"):
        convert_to_pyramidal_tiff(source, tmp_path / "output.tif")


@pytest.mark.parametrize("suffix", [".png", ".jpg"])
def test_conversion_rejects_truncated_end_marker(tmp_path: Path, suffix: str) -> None:
    source = tmp_path / f"source{suffix}"
    Image.new("RGB", (32, 16)).save(source)
    source.write_bytes(source.read_bytes()[:-12])
    with pytest.raises((ValueError, RuntimeError)):
        convert_to_pyramidal_tiff(source, tmp_path / "output.tif")


def test_png_color_metadata_does_not_change_encoded_pixels(tmp_path: Path) -> None:
    import struct

    from PIL.PngImagePlugin import PngInfo

    source = tmp_path / "source.png"
    metadata = PngInfo()
    metadata.add(b"gAMA", struct.pack(">I", 100000))
    Image.new("RGB", (32, 16), (23, 124, 241)).save(source, pnginfo=metadata)
    output = tmp_path / "output.tif"
    convert_to_pyramidal_tiff(source, output)
    with Image.open(output) as decoded:
        assert decoded.getpixel((0, 0)) == (23, 124, 241)
        assert "gamma" not in decoded.info


@pytest.mark.parametrize("suffix", [".png", ".jpg"])
def test_conversion_rejects_malformed_exif(tmp_path: Path, suffix: str) -> None:
    source = tmp_path / f"source{suffix}"
    Image.new("RGB", (32, 16)).save(source, exif=b"Exif\0\0invalid")
    with pytest.raises(ValueError, match="Invalid conversion source"):
        convert_to_pyramidal_tiff(source, tmp_path / "output.tif")
