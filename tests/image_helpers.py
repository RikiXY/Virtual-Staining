from __future__ import annotations

from pathlib import Path

from PIL import Image


def make_rgb_image(
    *,
    size: tuple[int, int] = (16, 16),
    color: tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:
    """Return a small RGB image."""
    return Image.new("RGB", size, color=color)


def write_rgb_image(
    path: Path,
    *,
    size: tuple[int, int] = (16, 16),
    color: tuple[int, int, int] = (0, 0, 0),
) -> Path:
    """Write a small RGB image and return its path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    make_rgb_image(size=size, color=color).save(path)
    return path


def write_rgb_pair(
    directory: Path,
    sample_id: str = "00000_00000",
    *,
    size: tuple[int, int] = (16, 16),
    ext: str = ".png",
    color: tuple[int, int, int] = (0, 0, 0),
) -> tuple[Path, Path]:
    """Write <sample_id>_source/<sample_id>_target RGB images."""
    source = write_rgb_image(
        directory / f"{sample_id}_source{ext}",
        size=size,
        color=color,
    )
    target = write_rgb_image(
        directory / f"{sample_id}_target{ext}",
        size=size,
        color=color,
    )
    return source, target
