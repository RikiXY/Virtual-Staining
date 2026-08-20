from __future__ import annotations

import subprocess
import uuid
from pathlib import Path

from virtual_staining.utils.image_io import OpenSlideRegionImageReader, PillowRegionImageReader


def _conversion_paths(inputs: tuple[Path, ...], output_dir: Path) -> tuple[tuple[Path, Path], ...]:
    conversions: list[tuple[Path, Path]] = []
    for input_path in inputs:
        source = input_path.resolve()
        if source.is_file():
            if source.suffix.lower() not in {".tif", ".tiff"}:
                raise ValueError(f"Input must be a TIFF file: {source}")
            conversions.append((source, output_dir / source.name))
            continue
        if not source.is_dir():
            raise FileNotFoundError(f"Input TIFF or directory not found: {source}")

        matches = [
            path
            for path in source.rglob("*")
            if path.is_file()
            and path.suffix.lower() in {".tif", ".tiff"}
            and not path.resolve().is_relative_to(output_dir)
        ]
        if not matches:
            raise ValueError(f"Directory contains no TIFF files: {source}")
        conversions.extend(
            (path.resolve(), output_dir / path.relative_to(source)) for path in sorted(matches)
        )
    return tuple(conversions)


def convert_images(inputs: tuple[Path, ...], output_dir: Path) -> tuple[Path, ...]:
    """Convert TIFF images to lossless tiled pyramidal BigTIFFs."""
    output_dir = output_dir.resolve()
    if not inputs:
        raise ValueError("At least one input TIFF is required")
    conversions = _conversion_paths(inputs, output_dir)
    destinations = tuple(destination for _, destination in conversions)
    if len(set(destinations)) != len(destinations):
        raise ValueError("Input files map to duplicate destinations")
    for destination in destinations:
        if destination.exists():
            raise FileExistsError(f"Destination already exists: {destination}")

    output_dir.mkdir(parents=True, exist_ok=True)
    completed: list[Path] = []
    for source, destination in conversions:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.parent / f".{destination.stem}.{uuid.uuid4().hex}.tmp.tif"
        try:
            try:
                subprocess.run(
                    [
                        "vips",
                        "tiffsave",
                        str(source),
                        str(temporary),
                        "--tile",
                        "--pyramid",
                        "--bigtiff",
                        "--compression=lzw",
                        "--tile-width=256",
                        "--tile-height=256",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except FileNotFoundError as exc:
                raise RuntimeError(
                    "libvips is required; run this command inside 'nix develop'"
                ) from exc
            except subprocess.CalledProcessError as exc:
                detail = (exc.stderr or "").strip() or (exc.stdout or "").strip() or str(exc)
                raise RuntimeError(f"Could not convert {source}: {detail}") from exc

            expected_size = PillowRegionImageReader(source).size
            reader = OpenSlideRegionImageReader(temporary)
            try:
                if reader.size != expected_size:
                    raise RuntimeError(
                        f"Converted dimensions differ for {source}: "
                        f"expected {expected_size}, got {reader.size}"
                    )
            finally:
                reader.close()
            temporary.replace(destination)
            completed.append(destination)
        finally:
            temporary.unlink(missing_ok=True)

    return tuple(completed)


__all__ = ["convert_images"]
