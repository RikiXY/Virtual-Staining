from __future__ import annotations

import logging
import math
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

from virtual_staining.config.run import RunConfig
from virtual_staining.experiment.run_paths import RunPaths
from virtual_staining.inference.outputs import generated_filename_for_sample
from virtual_staining.inference.runner import (
    build_inference_transform,
    load_inference_generator,
    predict_batch,
    resolve_inference_device,
)
from virtual_staining.utils.image_io import (
    VALID_IMAGE_EXTENSIONS,
    ImageMetadata,
    OpenSlideRegionImageReader,
    RegionImageReader,
    open_image_reader,
    open_rgb,
)

logger = logging.getLogger(__name__)
DEFAULT_TILE_OVERLAP = 16
SingleInferenceMode = Literal["auto", "resize", "tile"]
SUPPORTED_OUTPUT_FORMATS: frozenset[str] = frozenset(
    {"same", "bmp", "jpeg", "jpg", "png", "tif", "tiff"}
)


@dataclass(frozen=True)
class SingleInferenceResult:
    input_paths: dict[str, Path]
    output_path: Path
    checkpoint_path: Path
    image_size: tuple[int, int]
    mode: str
    device: str


@dataclass(frozen=True)
class DirectoryInferenceResult:
    input_dirs: dict[str, Path]
    output_dir: Path
    checkpoint_path: Path
    image_size: tuple[int, int]
    device: str
    results: tuple[SingleInferenceResult, ...]


@dataclass(frozen=True)
class _InferenceRuntime:
    paths: RunPaths
    generator: torch.nn.Module
    checkpoint_path: Path
    image_size: tuple[int, int]
    device: torch.device


def _generator_input_names(generator: torch.nn.Module) -> tuple[str, ...]:
    names = getattr(generator, "input_names", None)
    if not isinstance(names, tuple) or not all(isinstance(name, str) for name in names):
        raise ValueError("Inference generator must expose tuple[str, ...] input_names")
    return names


def _sample_id_from_input_path(input_path: Path) -> str:
    stem = input_path.stem
    if stem.endswith("_source"):
        return stem[: -len("_source")]
    return stem


def _generated_filename_for_input(input_path: Path, output_suffix: str) -> str:
    return generated_filename_for_sample(_sample_id_from_input_path(input_path), output_suffix)


def _validate_supported_image_path(path: Path, *, label: str) -> None:
    suffix = path.suffix.lower()
    if suffix not in VALID_IMAGE_EXTENSIONS:
        raise ValueError(
            f"{label} must use one of {sorted(VALID_IMAGE_EXTENSIONS)}, got {suffix!r}"
        )


def _validate_output_format(output_format: str) -> str:
    normalized = output_format.lower().lstrip(".")
    if normalized not in SUPPORTED_OUTPUT_FORMATS:
        raise ValueError(
            f"output_format must be one of {sorted(SUPPORTED_OUTPUT_FORMATS)}, "
            f"got {output_format!r}"
        )
    return normalized


def _output_suffix_for_input(input_path: Path, output_format: str) -> str:
    normalized = _validate_output_format(output_format)
    if normalized == "same":
        return input_path.suffix.lower()
    return f".{normalized}"


def _validate_mode(mode: str) -> SingleInferenceMode:
    if mode not in {"auto", "resize", "tile"}:
        raise ValueError("mode must be one of: auto, resize, tile")
    return cast(SingleInferenceMode, mode)


def _resolve_mode(
    mode: SingleInferenceMode, input_size: tuple[int, int], image_size: tuple[int, int]
) -> Literal["resize", "tile"]:
    if mode == "auto":
        return "resize" if input_size == image_size else "tile"
    return mode


def _validate_tile_overlap(tile_size: tuple[int, int], tile_overlap: int) -> None:
    if tile_overlap < 0:
        raise ValueError("tile_overlap must be greater than or equal to 0")
    if tile_overlap >= min(tile_size):
        raise ValueError(
            "tile_overlap must be smaller than both configured image_size dimensions; "
            f"got tile_overlap={tile_overlap}, image_size={tile_size}"
        )


def _tile_starts(length: int, tile_length: int, stride: int) -> list[int]:
    if length <= tile_length:
        return [0]

    last_start = length - tile_length
    starts = list(range(0, last_start + 1, stride))
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def _pad_tile(tile: Image.Image, tile_size: tuple[int, int]) -> Image.Image:
    if tile.size == tile_size:
        return tile
    padded = Image.new("RGB", tile_size, color=(255, 255, 255))
    padded.paste(tile, (0, 0))
    return padded


def _build_no_resize_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )


def _predict_images(
    images: dict[str, Image.Image],
    generator: torch.nn.Module,
    device: torch.device,
    transform: transforms.Compose,
) -> torch.Tensor:
    inputs: dict[str, torch.Tensor] = {}
    for name in _generator_input_names(generator):
        source_tensor = transform(images[name])
        if not isinstance(source_tensor, torch.Tensor):
            raise TypeError("Inference transform must return a torch.Tensor")
        inputs[name] = source_tensor.unsqueeze(0)
    return predict_batch(generator, inputs, device)[0].cpu()


def _run_resized_prediction(
    images: dict[str, Image.Image],
    generator: torch.nn.Module,
    device: torch.device,
    image_size: tuple[int, int],
) -> torch.Tensor:
    transform = build_inference_transform(image_size)
    return _predict_images(images, generator, device, transform)


def _run_tiled_prediction(
    images: dict[str, Image.Image],
    generator: torch.nn.Module,
    device: torch.device,
    image_size: tuple[int, int],
    tile_overlap: int,
) -> torch.Tensor:
    _validate_tile_overlap(image_size, tile_overlap)

    image_w, image_h = images[_generator_input_names(generator)[0]].size
    tile_w, tile_h = image_size
    stride_w = tile_w - tile_overlap
    stride_h = tile_h - tile_overlap
    x_starts = _tile_starts(image_w, tile_w, stride_w)
    y_starts = _tile_starts(image_h, tile_h, stride_h)

    transform = _build_no_resize_transform()
    accumulator = torch.zeros((3, image_h, image_w), dtype=torch.float32)
    weights = torch.zeros((1, image_h, image_w), dtype=torch.float32)

    for y in y_starts:
        for x in x_starts:
            tiles = {
                name: _pad_tile(
                    image.crop((x, y, min(x + tile_w, image_w), min(y + tile_h, image_h))),
                    image_size,
                )
                for name, image in images.items()
            }
            actual_w = min(x + tile_w, image_w) - x
            actual_h = min(y + tile_h, image_h) - y
            predicted = _predict_images(tiles, generator, device, transform)
            predicted = predicted[:, :actual_h, :actual_w]
            accumulator[:, y : y + actual_h, x : x + actual_w] += predicted
            weights[:, y : y + actual_h, x : x + actual_w] += 1.0

    return (accumulator / weights.clamp_min(1.0)).clamp(0, 1)


def _write_tiled_rgb(
    readers: Mapping[str, RegionImageReader],
    output_path: Path,
    generator: torch.nn.Module,
    device: torch.device,
    image_size: tuple[int, int],
    tile_overlap: int,
) -> None:
    """Run tiled inference into a disk-backed RGB byte buffer."""
    _validate_tile_overlap(image_size, tile_overlap)

    image_w, image_h = readers[_generator_input_names(generator)[0]].size
    tile_w, tile_h = image_size
    x_starts = _tile_starts(image_w, tile_w, tile_w - tile_overlap)
    y_starts = _tile_starts(image_h, tile_h, tile_h - tile_overlap)
    x_weights = np.zeros(image_w, dtype=np.uint32)
    y_weights = np.zeros(image_h, dtype=np.uint32)
    for x in x_starts:
        x_weights[x : min(x + tile_w, image_w)] += 1
    for y in y_starts:
        y_weights[y : min(y + tile_h, image_h)] += 1

    accumulator_path = output_path.with_suffix(".float32")
    accumulator = np.memmap(
        accumulator_path, mode="w+", dtype=np.float32, shape=(image_h, image_w, 3)
    )
    transform = _build_no_resize_transform()
    try:
        for y in y_starts:
            for x in x_starts:
                actual_w = min(tile_w, image_w - x)
                actual_h = min(tile_h, image_h - y)
                images = {
                    name: Image.fromarray(
                        reader.read_region(x, y, actual_w, actual_h)[:, :, ::-1].copy()
                    )
                    for name, reader in readers.items()
                }
                images = {name: _pad_tile(image, image_size) for name, image in images.items()}
                predicted = _predict_images(images, generator, device, transform)[
                    :, :actual_h, :actual_w
                ]
                accumulator[y : y + actual_h, x : x + actual_w] += predicted.permute(
                    1, 2, 0
                ).numpy()

        output = np.memmap(output_path, mode="w+", dtype=np.uint8, shape=(image_h, image_w, 3))
        try:
            for y in range(0, image_h, tile_h):
                bottom = min(y + tile_h, image_h)
                weights = y_weights[y:bottom, None] * x_weights[None, :]
                output[y:bottom] = np.clip(
                    accumulator[y:bottom] / weights[:, :, None] * 255.0 + 0.5,
                    0,
                    255,
                ).astype(np.uint8)
            output.flush()
        finally:
            del output
    finally:
        del accumulator
        accumulator_path.unlink(missing_ok=True)


def _save_pyramidal_tiff(raw_path: Path, output_path: Path, metadata: ImageMetadata) -> None:
    """Encode a raw RGB buffer as an OpenSlide-readable pyramidal BigTIFF."""
    try:
        import pyvips  # pyright: ignore[reportMissingImports]
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "pyvips and libvips are required; install the 'wsi' extra and run inside 'nix develop'"
        ) from exc

    width, height = metadata.width, metadata.height
    generated_path = raw_path.with_suffix(".tif")
    resolution = {
        **({"xres": 1000.0 / metadata.mpp_x} if metadata.mpp_x is not None else {}),
        **({"yres": 1000.0 / metadata.mpp_y} if metadata.mpp_y is not None else {}),
    }
    try:
        image = pyvips.Image.rawload(
            str(raw_path),
            width,
            height,
            3,
            format="uchar",
            interpretation="srgb",
        )
        image.tiffsave(
            str(generated_path),
            tile=True,
            tile_width=256,
            tile_height=256,
            pyramid=True,
            bigtiff=True,
            compression="lzw",
            **resolution,
        )
    except pyvips.Error as exc:
        raise RuntimeError(f"Could not write pyramidal TIFF: {exc}") from exc

    generated = OpenSlideRegionImageReader(generated_path)
    raw: np.memmap | None = None
    try:
        raw = np.memmap(raw_path, mode="r", dtype=np.uint8, shape=(height, width, 3))
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
            if expected_mpp is None:
                continue
            actual_mpp = getattr(generated_metadata, f"mpp_{axis}")
            if actual_mpp is None or not math.isclose(actual_mpp, expected_mpp, rel_tol=1e-3):
                raise RuntimeError(
                    f"Generated mpp_{axis} differs: expected {expected_mpp}, got {actual_mpp}"
                )

        coordinates = {
            (0, 0),
            (width // 2, height // 2),
            (width - 1, height - 1),
        }
        for x, y in coordinates:
            actual = generated.read_region(x, y, 1, 1)[0, 0, ::-1]
            if not np.array_equal(actual, raw[y, x]):
                raise RuntimeError(
                    f"Generated pixel differs at ({x}, {y}): "
                    f"expected {raw[y, x].tolist()}, got {actual.tolist()}"
                )
    finally:
        if raw is not None:
            del raw
        generated.close()
    generated_path.replace(output_path)


def _run_wsi_prediction(
    readers: dict[str, OpenSlideRegionImageReader],
    output_path: Path,
    generator: torch.nn.Module,
    device: torch.device,
    image_size: tuple[int, int],
    tile_overlap: int,
) -> None:
    if output_path.suffix.lower() not in {".tif", ".tiff"}:
        raise ValueError("Full-resolution WSI output must use .tif or .tiff")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    first_reader = readers[_generator_input_names(generator)[0]]
    with tempfile.TemporaryDirectory(prefix=f".{output_path.stem}.", dir=output_path.parent) as tmp:
        raw_path = Path(tmp) / "generated.rgb"
        _write_tiled_rgb(readers, raw_path, generator, device, image_size, tile_overlap)
        _save_pyramidal_tiff(raw_path, output_path, first_reader.metadata)


def _build_runtime(config: RunConfig) -> _InferenceRuntime:
    paths = RunPaths(config.project.run_root)
    paths.create_directories()

    device = resolve_inference_device()
    logger.info("Image inference device: %s", device)
    generator, checkpoint_path = load_inference_generator(config, paths, device)
    return _InferenceRuntime(
        paths=paths,
        generator=generator,
        checkpoint_path=checkpoint_path,
        image_size=config.project.image_size,
        device=device,
    )


def _default_output_dir(config: RunConfig, paths: RunPaths, dirname: str) -> Path:
    if config.inference is None:
        raise ValueError("RunConfig.inference is required to run image inference.")
    return config.inference.output_dir or paths.artifacts_dir / dirname


def _resolve_output_path(
    config: RunConfig,
    paths: RunPaths,
    input_path: Path,
    output_image: Path | None,
    *,
    output_format: str = "same",
    default_dirname: str = "output_single",
) -> Path:
    if output_image is not None:
        return output_image

    output_dir = _default_output_dir(config, paths, default_dirname)
    output_suffix = _output_suffix_for_input(input_path, output_format)
    return output_dir / _generated_filename_for_input(input_path, output_suffix)


def _run_one_image(
    runtime: _InferenceRuntime,
    input_images: dict[str, Path],
    *,
    output_path: Path,
    mode: SingleInferenceMode = "auto",
    tile_overlap: int = DEFAULT_TILE_OVERLAP,
) -> SingleInferenceResult:
    input_names = _generator_input_names(runtime.generator)
    if tuple(input_images) != input_names:
        raise ValueError(
            f"Input paths must match generator input order {input_names}, got {tuple(input_images)}"
        )
    for name, input_path in input_images.items():
        if not input_path.is_file():
            raise FileNotFoundError(f"Input image {name} not found: {input_path}")
        _validate_supported_image_path(input_path, label=f"input_image[{name}]")
    _validate_supported_image_path(output_path, label="output_image")

    requested_mode = _validate_mode(mode)
    readers: dict[str, RegionImageReader] = {}
    try:
        for name, input_path in input_images.items():
            readers[name] = open_image_reader(input_path)
        sizes = {reader.size for reader in readers.values()}
        if len(sizes) != 1:
            details = ", ".join(f"{name}={reader.size}" for name, reader in readers.items())
            raise ValueError(f"Input image dimensions must match; got {details}")

        first_reader = readers[input_names[0]]
        resolved_mode = _resolve_mode(requested_mode, first_reader.size, runtime.image_size)
        pillow_limit = Image.MAX_IMAGE_PIXELS
        if (
            resolved_mode == "tile"
            and not all(
                isinstance(reader, OpenSlideRegionImageReader) for reader in readers.values()
            )
            and pillow_limit is not None
            and first_reader.size[0] * first_reader.size[1] > 2 * pillow_limit
        ):
            raise RuntimeError(
                "Large tiled inference requires OpenSlide; install the 'wsi' extra and "
                "native OpenSlide, then run inside 'nix develop'"
            )
        if resolved_mode == "tile" and any(
            isinstance(reader, OpenSlideRegionImageReader) for reader in readers.values()
        ):
            if not all(
                isinstance(reader, OpenSlideRegionImageReader) for reader in readers.values()
            ):
                raise ValueError(
                    "Full-resolution multi-input WSI inference requires every input "
                    "to use OpenSlide"
                )
            _run_wsi_prediction(
                readers,  # type: ignore[arg-type]
                output_path,
                runtime.generator,
                runtime.device,
                runtime.image_size,
                tile_overlap,
            )
            output = None
        else:
            images = {name: open_rgb(input_path) for name, input_path in input_images.items()}
            if resolved_mode == "resize":
                output = _run_resized_prediction(
                    images, runtime.generator, runtime.device, runtime.image_size
                )
            else:
                output = _run_tiled_prediction(
                    images,
                    runtime.generator,
                    runtime.device,
                    runtime.image_size,
                    tile_overlap,
                )
    finally:
        for reader in readers.values():
            reader.close()

    if output is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_image(output, output_path)

    logger.info(
        "Single-image inference complete: %s -> %s (mode=%s)",
        input_images,
        output_path,
        resolved_mode,
    )
    return SingleInferenceResult(
        input_paths=dict(input_images),
        output_path=output_path,
        checkpoint_path=runtime.checkpoint_path,
        image_size=runtime.image_size,
        mode=resolved_mode,
        device=str(runtime.device),
    )


def collect_input_images(input_dir: Path, *, recursive: bool = False) -> tuple[Path, ...]:
    """Return supported image files from a directory in deterministic order."""
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory not found: {input_dir}")

    iterator = input_dir.rglob("*") if recursive else input_dir.iterdir()
    return tuple(
        sorted(
            path
            for path in iterator
            if path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTENSIONS
        )
    )


def run_single_image_inference(
    config: RunConfig,
    input_images: dict[str, Path],
    output_image: Path | None = None,
    *,
    mode: SingleInferenceMode = "auto",
    tile_overlap: int = DEFAULT_TILE_OVERLAP,
    output_format: str = "same",
) -> SingleInferenceResult:
    """Load a checkpoint, run the generator on one image, and save the generated output."""
    return _run_image_file_inference(
        config,
        input_images,
        output_image,
        mode=mode,
        tile_overlap=tile_overlap,
        output_format=output_format,
        default_dirname="output_single",
    )


def _run_image_file_inference(
    config: RunConfig,
    input_images: dict[str, Path],
    output_image: Path | None = None,
    *,
    mode: SingleInferenceMode = "auto",
    tile_overlap: int = DEFAULT_TILE_OVERLAP,
    output_format: str = "same",
    default_dirname: str,
) -> SingleInferenceResult:
    if config.inference is None:
        raise ValueError("RunConfig.inference is required to run image inference.")
    input_names = tuple(config.model.inputs)
    if set(input_images) != set(input_names):
        raise ValueError(
            f"Input modalities must match configured names {input_names}, got {tuple(input_images)}"
        )

    input_paths = {name: Path(input_images[name]) for name in input_names}
    first_input_path = input_paths[input_names[0]]
    runtime = _build_runtime(config)
    output_path = _resolve_output_path(
        config,
        runtime.paths,
        first_input_path,
        output_image,
        output_format=output_format,
        default_dirname=default_dirname,
    )
    return _run_one_image(
        runtime,
        input_paths,
        output_path=output_path,
        mode=mode,
        tile_overlap=tile_overlap,
    )


def run_image_directory_inference(
    config: RunConfig,
    input_dirs: dict[str, Path],
    output_dir: Path | None = None,
    *,
    recursive: bool = False,
    mode: SingleInferenceMode = "auto",
    tile_overlap: int = DEFAULT_TILE_OVERLAP,
    output_format: str = "same",
) -> DirectoryInferenceResult:
    """Run image inference for all supported image files in named directories."""
    if config.inference is None:
        raise ValueError("RunConfig.inference is required to run image inference.")
    input_names = tuple(config.model.inputs)
    if set(input_dirs) != set(input_names):
        raise ValueError(
            f"Input modalities must match configured names {input_names}, got {tuple(input_dirs)}"
        )

    roots = {name: Path(input_dirs[name]) for name in input_names}
    first_name = input_names[0]
    first_root = roots[first_name]
    images_by_name = {
        name: collect_input_images(root, recursive=recursive) for name, root in roots.items()
    }
    if not images_by_name[first_name]:
        raise FileNotFoundError(
            f"No supported images found in {first_root}. "
            f"Supported extensions: {sorted(VALID_IMAGE_EXTENSIONS)}"
        )

    first_relative = {path.relative_to(first_root) for path in images_by_name[first_name]}
    for name, root in roots.items():
        if name == first_name:
            continue
        relative = {path.relative_to(root) for path in images_by_name[name]}
        missing = sorted(first_relative - relative)
        extra = sorted(relative - first_relative)
        if missing or extra:
            raise ValueError(
                f"Input modality {name} relative paths differ: missing={missing}, extra={extra}"
            )
    runtime = _build_runtime(config)
    if output_dir is not None and output_dir.suffix:
        raise NotADirectoryError(
            f"Output path for directory inference must be a directory: {output_dir}"
        )
    resolved_output_dir = output_dir or _default_output_dir(
        config,
        runtime.paths,
        "output_images",
    )
    if resolved_output_dir.exists() and not resolved_output_dir.is_dir():
        raise NotADirectoryError(
            f"Output path for directory inference must be a directory: {resolved_output_dir}"
        )
    ordered_names = _generator_input_names(runtime.generator)
    results: list[SingleInferenceResult] = []
    for relative_path in sorted(first_relative):
        source_paths = {name: roots[name] / relative_path for name in ordered_names}
        relative_parent = relative_path.parent if recursive else Path()
        output_suffix = _output_suffix_for_input(source_paths[ordered_names[0]], output_format)
        output_path = (
            resolved_output_dir
            / relative_parent
            / _generated_filename_for_input(source_paths[ordered_names[0]], output_suffix)
        )
        results.append(
            _run_one_image(
                runtime,
                source_paths,
                output_path=output_path,
                mode=mode,
                tile_overlap=tile_overlap,
            )
        )

    return DirectoryInferenceResult(
        input_dirs={name: roots[name] for name in ordered_names},
        output_dir=resolved_output_dir,
        checkpoint_path=runtime.checkpoint_path,
        image_size=runtime.image_size,
        device=str(runtime.device),
        results=tuple(results),
    )


def run_image_path_inference(
    config: RunConfig,
    input_paths: dict[str, Path],
    output_path: Path | None = None,
    *,
    recursive: bool = False,
    mode: SingleInferenceMode = "auto",
    tile_overlap: int = DEFAULT_TILE_OVERLAP,
    output_format: str = "same",
) -> SingleInferenceResult | DirectoryInferenceResult:
    """Run image inference on named files or named directories."""
    paths = {name: Path(path) for name, path in input_paths.items()}
    if not paths:
        raise ValueError("At least one input path is required.")
    kinds: set[str] = set()
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Input path {name} not found: {path}")
        if path.is_file():
            kinds.add("file")
        elif path.is_dir():
            kinds.add("directory")
        else:
            raise ValueError(f"Input path {name} is neither a file nor a directory: {path}")
    if len(kinds) != 1:
        raise ValueError("All input paths must be files or all input paths must be directories.")
    if "directory" in kinds:
        return run_image_directory_inference(
            config,
            paths,
            output_path,
            recursive=recursive,
            mode=mode,
            tile_overlap=tile_overlap,
            output_format=output_format,
        )
    return run_single_image_inference(
        config,
        paths,
        output_path,
        mode=mode,
        tile_overlap=tile_overlap,
        output_format=output_format,
    )
