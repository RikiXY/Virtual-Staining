from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from PIL import Image

VALID_IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
)


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
    def size(self) -> tuple[int, int]:
        """Return image size as ``(width, height)``."""
        ...

    @property
    def metadata(self) -> ImageMetadata:
        """Return basic image metadata."""
        ...

    def read_region(self, x: int, y: int, width: int, height: int) -> np.ndarray:
        """Read a BGR uint8 region, padding out-of-bounds areas with white."""
        ...

    def read_preview(self, scale: float) -> np.ndarray:
        """Read a BGR uint8 downscaled preview."""
        ...

    def read_full(self) -> np.ndarray:
        """Read the full image as BGR uint8."""
        ...

    def close(self) -> None:
        """Release backend resources."""
        ...


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
    """Optional OpenSlide-backed level-0 region reader."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"Image not found: {self.path}")
        if detect_openslide_format(self.path) is None:
            raise ValueError(f"OpenSlide does not support: {self.path}")
        import openslide  # pyright: ignore[reportMissingImports]

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
    """Open a local image with Pillow or the optional native WSI backend."""
    if backend not in {"auto", "pillow", "openslide"}:
        raise ValueError("backend must be auto, pillow, or openslide")
    if backend == "pillow":
        return PillowRegionImageReader(path)
    if backend == "openslide":
        return OpenSlideRegionImageReader(path)
    try:
        detected = detect_openslide_format(path)
    except RuntimeError:
        return PillowRegionImageReader(path)
    if detected is not None:
        return OpenSlideRegionImageReader(path)
    return PillowRegionImageReader(path)


def detect_openslide_format(path: str | Path) -> str | None:
    """Return the OpenSlide format name without decoding the image."""
    try:
        import openslide  # pyright: ignore[reportMissingImports]
    except ImportError as exc:
        raise RuntimeError(
            "OpenSlide is unavailable; install the 'wsi' extra and native OpenSlide"
        ) from exc
    return openslide.OpenSlide.detect_format(str(path))


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
