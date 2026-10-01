from __future__ import annotations

import logging
import math
import struct
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np
import openslide
from PIL import Image

VALID_IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
)
SUPPORTED_IMAGE_BACKENDS: frozenset[str] = frozenset({"auto", "pillow", "openslide"})
_TIFF_RESOLUTION_UNIT_TAG = 296
_TIFF_RESUNIT_NONE = 1

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ImageMetadata:
    width: int
    height: int
    level_count: int = 1
    level_dimensions: tuple[tuple[int, int], ...] = ()
    level_downsamples: tuple[float, ...] = ()
    mpp_x: float | None = None
    mpp_y: float | None = None
    vendor: str | None = None


class RegionImageReader(Protocol):
    """Minimal region-readable image interface used by dataset preparation."""

    path: Path

    @property
    def size(self) -> tuple[int, int]: ...

    @property
    def metadata(self) -> ImageMetadata: ...

    def read_region(self, x: int, y: int, width: int, height: int) -> np.ndarray: ...

    def read_preview(self, scale: float) -> np.ndarray: ...

    def read_full(self) -> np.ndarray: ...

    def close(self) -> None: ...


def _pil_to_bgr_array(img: Image.Image) -> np.ndarray:
    rgb = np.array(img.convert("RGB"), dtype=np.uint8)
    return rgb[:, :, ::-1].copy()


class PillowRegionImageReader:
    """Pillow-backed region reader for standard local image formats."""

    def __init__(self, path: str | Path) -> None:
        image_path = Path(path)
        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")
        self.path = image_path
        self._original_max_image_pixels = Image.MAX_IMAGE_PIXELS
        try:
            Image.MAX_IMAGE_PIXELS = None
            with Image.open(image_path) as img:
                self._size = img.size
        finally:
            Image.MAX_IMAGE_PIXELS = self._original_max_image_pixels

    @property
    def size(self) -> tuple[int, int]:
        return self._size

    @property
    def metadata(self) -> ImageMetadata:
        width, height = self._size
        return ImageMetadata(
            width=width, height=height, level_dimensions=(self._size,), level_downsamples=(1.0,)
        )

    def _open(self) -> Image.Image:
        original_max_image_pixels = Image.MAX_IMAGE_PIXELS
        Image.MAX_IMAGE_PIXELS = None
        try:
            return Image.open(self.path).convert("RGB")
        finally:
            Image.MAX_IMAGE_PIXELS = original_max_image_pixels

    def read_region(self, x: int, y: int, width: int, height: int) -> np.ndarray:
        if width <= 0 or height <= 0:
            raise ValueError("Region width and height must be positive")

        image_w, image_h = self._size
        crop_left = max(0, x)
        crop_top = max(0, y)
        crop_right = min(image_w, x + width)
        crop_bottom = min(image_h, y + height)

        region = Image.new("RGB", (width, height), color=(255, 255, 255))
        if crop_left < crop_right and crop_top < crop_bottom:
            with self._open() as img:
                crop = img.crop((crop_left, crop_top, crop_right, crop_bottom)).copy()
            region.paste(crop, (crop_left - x, crop_top - y))
        return _pil_to_bgr_array(region)

    def read_preview(self, scale: float) -> np.ndarray:
        if not (0.0 < scale <= 1.0):
            raise ValueError(f"Preview scale must be in (0.0, 1.0], got {scale}")
        image_w, image_h = self._size
        preview_size = (
            max(1, math.floor(image_w * scale)),
            max(1, math.floor(image_h * scale)),
        )
        with self._open() as img:
            if preview_size != img.size:
                img = img.resize(preview_size, Image.Resampling.BILINEAR)
            return _pil_to_bgr_array(img)

    def read_full(self) -> np.ndarray:
        return self.read_preview(1.0)

    def close(self) -> None:
        return None


class OpenSlideRegionImageReader:
    """OpenSlide-backed level-0 region reader."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"Image not found: {self.path}")
        if detect_openslide_format(self.path) is None:
            raise ValueError(f"OpenSlide does not support: {self.path}")
        self._slide: Any = openslide.OpenSlide(str(self.path))

    @property
    def size(self) -> tuple[int, int]:
        return tuple(self._slide.dimensions)

    @property
    def metadata(self) -> ImageMetadata:
        properties = self._slide.properties

        def optional_float(name: str) -> float | None:
            try:
                return float(properties[name])
            except (KeyError, TypeError, ValueError):
                return None

        width, height = self.size
        return ImageMetadata(
            width=width,
            height=height,
            level_count=int(self._slide.level_count),
            level_dimensions=tuple(tuple(size) for size in self._slide.level_dimensions),
            level_downsamples=tuple(float(value) for value in self._slide.level_downsamples),
            mpp_x=optional_float("openslide.mpp-x"),
            mpp_y=optional_float("openslide.mpp-y"),
            vendor=properties.get("openslide.vendor"),
        )

    def read_region(self, x: int, y: int, width: int, height: int) -> np.ndarray:
        if width <= 0 or height <= 0:
            raise ValueError("Region width and height must be positive")
        image = self._slide.read_region((x, y), 0, (width, height)).convert("RGB")
        return _pil_to_bgr_array(image)

    def read_preview(self, scale: float) -> np.ndarray:
        if not (0.0 < scale <= 1.0):
            raise ValueError(f"Preview scale must be in (0.0, 1.0], got {scale}")
        width, height = self.size
        output_size = (max(1, math.floor(width * scale)), max(1, math.floor(height * scale)))
        level = self._slide.get_best_level_for_downsample(1.0 / scale)
        level_size = tuple(self._slide.level_dimensions[level])
        image = self._slide.read_region((0, 0), level, level_size).convert("RGB")
        if image.size != output_size:
            image = image.resize(output_size, Image.Resampling.BILINEAR)
        return _pil_to_bgr_array(image)

    def read_full(self) -> np.ndarray:
        width, height = self.size
        return self.read_region(0, 0, width, height)

    def close(self) -> None:
        self._slide.close()


def open_image_reader(path: str | Path, backend: str = "auto") -> RegionImageReader:
    if backend not in SUPPORTED_IMAGE_BACKENDS:
        raise ValueError("backend must be auto, pillow, or openslide")
    if backend == "pillow":
        return PillowRegionImageReader(path)
    if backend == "openslide":
        return OpenSlideRegionImageReader(path)
    detected = detect_openslide_format(path)
    if detected is not None:
        return OpenSlideRegionImageReader(path)
    return PillowRegionImageReader(path)


def detect_openslide_format(path: str | Path) -> str | None:
    return openslide.OpenSlide.detect_format(str(path))


def read_image_metadata(path: str | Path, backend: str = "auto") -> ImageMetadata:
    reader = open_image_reader(path, backend=backend)
    try:
        return reader.metadata
    finally:
        reader.close()


def read_full_image(path: str | Path, backend: str = "auto") -> np.ndarray:
    reader = open_image_reader(path, backend=backend)
    try:
        return reader.read_full()
    finally:
        reader.close()


def open_rgb(path: str | Path) -> Image.Image:
    image_path = Path(path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if not image_path.is_file():
        raise FileNotFoundError(f"Not a file: {image_path}")

    with Image.open(image_path) as img:
        return img.convert("RGB")


def load_rgb_image(path: str | Path) -> np.ndarray:
    image_path = Path(path)

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as exc:
        raise RuntimeError(f"Could not open image: {image_path}") from exc

    return np.array(image)


def load_grayscale_image(path: str | Path) -> np.ndarray:
    image_path = Path(path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Could not open image: {image_path}")
    return image


def _load_pyvips() -> Any:
    try:
        import pyvips
    except OSError as exc:
        raise RuntimeError(
            "Could not load native libvips; run inside the managed environment (nix develop)"
        ) from exc
    return pyvips


def _pyramid_options(metadata: ImageMetadata | None = None) -> dict[str, object]:
    options: dict[str, object] = {
        "tile": True,
        "tile_width": 256,
        "tile_height": 256,
        "pyramid": True,
        "bigtiff": True,
        "compression": "lzw",
    }
    if metadata is not None:
        if metadata.mpp_x is not None:
            options["xres"] = 1000.0 / metadata.mpp_x
        if metadata.mpp_y is not None:
            options["yres"] = 1000.0 / metadata.mpp_y
    return options


def _clear_bigtiff_resolution_unit(path: Path) -> None:
    """Mark a libvips BigTIFF's resolution as unitless in every IFD, in place.

    libvips always writes a resolution tag (1 px/mm by default), which OpenSlide would
    report as a fabricated 1000 um/px; a unitless resolution reads back as unknown MPP.
    """
    with path.open("r+b") as handle:
        order = {b"II": "<", b"MM": ">"}[handle.read(2)]
        magic, _, _, offset = struct.unpack(order + "HHHQ", handle.read(14))
        if magic != 43:
            raise RuntimeError(f"Expected a BigTIFF: {path}")
        while offset:
            handle.seek(offset)
            (count,) = struct.unpack(order + "Q", handle.read(8))
            for index in range(count):
                entry = offset + 8 + index * 20
                handle.seek(entry)
                (tag,) = struct.unpack(order + "H", handle.read(2))
                if tag == _TIFF_RESOLUTION_UNIT_TAG:
                    handle.seek(entry + 12)
                    handle.write(struct.pack(order + "H", _TIFF_RESUNIT_NONE))
            handle.seek(offset + 8 + count * 20)
            (offset,) = struct.unpack(order + "Q", handle.read(8))


def convert_to_pyramidal_tiff(source_path: str | Path, output_path: str | Path) -> None:
    source = Path(source_path)
    output = Path(output_path)
    expected = read_image_metadata(source, backend="pillow")
    pyvips = _load_pyvips()
    try:
        image = pyvips.Image.new_from_file(str(source), access="sequential")
        image.tiffsave(str(output), **_pyramid_options())
    except pyvips.Error as exc:
        raise RuntimeError(f"Could not convert {source}: {exc}") from exc
    _clear_bigtiff_resolution_unit(output)

    actual = read_image_metadata(output, backend="openslide")
    expected_size = (expected.width, expected.height)
    actual_size = (actual.width, actual.height)
    if actual_size != expected_size:
        raise RuntimeError(
            f"Converted dimensions differ for {source}: expected {expected_size}, got {actual_size}"
        )


def write_pyramidal_tiff_from_raw_rgb(
    raw_path: str | Path, output_path: str | Path, metadata: ImageMetadata
) -> None:
    """Write, reopen and verify a pyramidal TIFF, then atomically replace ``output_path``.

    Verification reads headers (geometry, pyramid levels, MPP) and three sample pixels,
    never the whole slide. MPP is written only when both axes are known; TIFF has one
    resolution unit, so a half-known MPP is published as unknown rather than fabricated.
    """
    raw = Path(raw_path)
    output = Path(output_path)
    pyvips = _load_pyvips()
    width, height = metadata.width, metadata.height
    if (metadata.mpp_x is None) != (metadata.mpp_y is None):
        logger.warning("Only one MPP axis is known; the output MPP is left unknown")
        metadata = replace(metadata, mpp_x=None, mpp_y=None)
    generated_path = raw.with_suffix(".tif")
    try:
        image = pyvips.Image.rawload(
            str(raw),
            width,
            height,
            3,
            format="uchar",
            interpretation="srgb",
        )
        image.tiffsave(str(generated_path), **_pyramid_options(metadata))
    except pyvips.Error as exc:
        raise RuntimeError(f"Could not write pyramidal TIFF: {exc}") from exc
    if metadata.mpp_x is None:
        _clear_bigtiff_resolution_unit(generated_path)

    generated = OpenSlideRegionImageReader(generated_path)
    raw_pixels: np.memmap | None = None
    try:
        raw_pixels = np.memmap(raw, mode="r", dtype=np.uint8, shape=(height, width, 3))
        expected_size = (width, height)
        if generated.size != expected_size:
            raise RuntimeError(
                f"Generated dimensions differ: expected {expected_size}, got {generated.size}"
            )

        generated_metadata = generated.metadata
        if width > 256 or height > 256:
            downsamples = generated_metadata.level_downsamples
            if (
                generated_metadata.level_count <= 1
                or not downsamples
                or not math.isclose(downsamples[0], 1.0)
                or any(
                    current <= previous
                    for previous, current in zip(downsamples, downsamples[1:], strict=False)
                )
            ):
                raise RuntimeError("Generated TIFF failed the pyramidal level contract")

        for axis in ("x", "y"):
            expected_mpp = getattr(metadata, f"mpp_{axis}")
            actual_mpp = getattr(generated_metadata, f"mpp_{axis}")
            if expected_mpp is None:
                if actual_mpp is not None:
                    raise RuntimeError(f"Generated mpp_{axis} is {actual_mpp}, expected unknown")
                continue
            if actual_mpp is None or not math.isclose(actual_mpp, expected_mpp, rel_tol=1e-3):
                raise RuntimeError(
                    f"Generated mpp_{axis} differs: expected {expected_mpp}, got {actual_mpp}"
                )

        coordinates = {(0, 0), (width // 2, height // 2), (width - 1, height - 1)}
        for x, y in coordinates:
            actual = generated.read_region(x, y, 1, 1)[0, 0, ::-1]
            if not np.array_equal(actual, raw_pixels[y, x]):
                raise RuntimeError(
                    f"Generated pixel differs at ({x}, {y}): "
                    f"expected {raw_pixels[y, x].tolist()}, got {actual.tolist()}"
                )
    finally:
        if raw_pixels is not None:
            del raw_pixels
        generated.close()
    generated_path.replace(output)


def to_float01(image: np.ndarray | Image.Image) -> np.ndarray:
    return np.asarray(image, dtype=np.float32) / 255.0
