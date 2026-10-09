from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

from virtual_staining.utils.image_io import convert_to_pyramidal_tiff

logger = logging.getLogger(__name__)
_SOURCE_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


def _conversion_paths(inputs: tuple[Path, ...], output_dir: Path) -> tuple[tuple[Path, Path], ...]:
    conversions: list[tuple[Path, Path]] = []
    for input_path in inputs:
        source = input_path.resolve()
        if source.is_file():
            if source.suffix.lower() not in _SOURCE_EXTENSIONS:
                raise ValueError(f"Input must be a TIFF, PNG or JPEG file: {source}")
            matches = [(source, Path(source.name))]
        elif source.is_dir():
            if source.is_relative_to(output_dir):
                raise ValueError(
                    f"Input directory {source} is inside output directory {output_dir}"
                )
            matches = [
                (path.resolve(), path.relative_to(source))
                for path in sorted(source.rglob("*"))
                if path.is_file()
                and path.suffix.lower() in _SOURCE_EXTENSIONS
                and not path.resolve().is_relative_to(output_dir)
            ]
            if not matches:
                raise ValueError(f"Directory contains no TIFF, PNG or JPEG files: {source}")
        else:
            raise FileNotFoundError(f"Input TIFF, PNG, JPEG, or directory not found: {source}")
        for path, relative in matches:
            if relative.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                relative = relative.with_suffix(".tif")
            destination = output_dir / relative
            conversions.append((path, destination.parent.resolve() / destination.name))
    return tuple(conversions)


def convert_images(inputs: tuple[Path, ...], output_dir: Path) -> tuple[Path, ...]:
    output_dir = output_dir.resolve()
    if not inputs:
        raise ValueError("At least one input TIFF, PNG or JPEG is required")
    conversions = _conversion_paths(inputs, output_dir)
    destinations: dict[Path, Path] = {}
    sources: dict[tuple[int, int], tuple[Path, Path]] = {}
    for source, destination in conversions:
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"Destination already exists: {destination} (source: {source})")
        if destination in destinations:
            raise ValueError(
                f"Input files map to duplicate destinations: {destinations[destination]} and "
                f"{source} -> {destination}"
            )
        stat = source.stat()
        identity = (stat.st_dev, stat.st_ino)
        if identity in sources:
            previous, previous_destination = sources[identity]
            raise ValueError(
                f"Duplicate source selection: {previous} -> {previous_destination} and "
                f"{source} -> {destination}"
            )
        destinations[destination] = source
        sources[identity] = (source, destination)

    for destination, source in destinations.items():
        for parent in destination.parents:
            if parent in destinations:
                raise ValueError(
                    f"Destination path conflict: {destinations[parent]} -> {parent} blocks "
                    f"{source} -> {destination}"
                )
            if parent.exists() and not parent.is_dir():
                raise ValueError(
                    f"Destination parent is not a directory: {parent} (source: {source})"
                )

    completed: list[Path] = []
    total = len(conversions)
    for index, (source, destination) in enumerate(conversions, start=1):
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(
            prefix=f".{destination.stem}.", suffix=".tmp.tif", dir=destination.parent
        )
        temporary = Path(name)
        try:
            os.close(descriptor)
            logger.info("[%d/%d] Converting %s -> %s", index, total, source, destination)
            convert_to_pyramidal_tiff(source, temporary)
            try:
                os.link(temporary, destination)
            except FileExistsError as exc:
                raise FileExistsError(
                    f"Destination already exists: {destination} (source: {source})"
                ) from exc
            except OSError as exc:
                raise OSError(
                    exc.errno,
                    f"Cannot publish {source} -> {destination}: atomic no-replace publication "
                    f"requires hard links on the same filesystem: {exc}",
                ) from exc
            completed.append(destination)
            logger.info("[%d/%d] Converted %s", index, total, destination)
        finally:
            temporary.unlink(missing_ok=True)

    return tuple(completed)


__all__ = ["convert_images"]
