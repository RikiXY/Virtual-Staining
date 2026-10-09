from __future__ import annotations

import logging
import math
import shutil
import tempfile
from collections.abc import Callable, Mapping
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias, cast

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from virtual_staining.inference.outputs import save_rgb
from virtual_staining.inference.runner import (
    Predictor,
    build_inference_transform,
    predict_batch,
)
from virtual_staining.models.io_contract import MODEL_INPUT_RANGE, build_model_input_transform
from virtual_staining.utils.artifacts import generated_path, require_output_name
from virtual_staining.utils.image_io import (
    VALID_IMAGE_EXTENSIONS,
    ImageMetadata,
    OpenSlideRegionImageReader,
    RegionImageReader,
    open_image_reader,
    open_rgb,
    write_pyramidal_tiff_from_raw_rgb,
)

logger = logging.getLogger(__name__)
DEFAULT_TILE_OVERLAP = 16
SingleInferenceMode = Literal["auto", "resize", "tile"]
SUPPORTED_OUTPUT_FORMATS: frozenset[str] = frozenset(
    {"same", *(extension.removeprefix(".") for extension in VALID_IMAGE_EXTENSIONS)}
)
#: Named RGB outputs whose pixel grids map one-to-one onto the predictor input grid.
SAME_GRID_RGB = "same_grid_rgb"
#: Relative slack when comparing known source MPP values (metadata representation noise).
MPP_REL_TOLERANCE = 1e-4


@dataclass(frozen=True)
class PredictionContract:
    """What transport knows about a predictor: named RGB inputs -> named same-grid outputs.

    ``image_size`` is the predictor/tile input ``(width, height)``. Inputs are fed in
    ``input_names`` order as NCHW tensors in ``value_range``; the predictor returns
    ``{output_name: (N, 3, H, W) tensor}`` with exactly ``output_names`` in order, in the
    same range on the same grid. One output is a one-item mapping.
    """

    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    image_size: tuple[int, int]
    output_semantics: str = SAME_GRID_RGB
    value_range: tuple[int, int] = MODEL_INPUT_RANGE

    def __post_init__(self) -> None:
        for field_name, names in (
            ("input_names", self.input_names),
            ("output_names", self.output_names),
        ):
            if (
                not isinstance(names, tuple)
                or not names
                or not all(isinstance(name, str) and name.strip() for name in names)
            ):
                raise ValueError(f"{field_name} must be a non-empty tuple of names, got {names!r}")
            if len(set(names)) != len(names):
                raise ValueError(f"{field_name} must be unique, got {names!r}")
        # Output names become artifact directories; the identifier rule keeps them there.
        for name in self.output_names:
            require_output_name(name)
        size = self.image_size
        if (
            not isinstance(size, tuple)
            or len(size) != 2
            or not all(isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in size)
        ):
            raise ValueError(f"image_size must be (width, height) positive integers, got {size!r}")
        if self.output_semantics != SAME_GRID_RGB:
            raise ValueError(
                f"Unsupported output_semantics {self.output_semantics!r}; only "
                f"{SAME_GRID_RGB!r} (named RGB outputs on the input pixel grid) is supported"
            )
        if tuple(self.value_range) != MODEL_INPUT_RANGE:
            raise ValueError(
                f"Unsupported value_range {self.value_range!r}; "
                f"only {MODEL_INPUT_RANGE} is supported"
            )


@dataclass(frozen=True)
class InferenceRuntime:
    """A caller-owned, already prepared predictor plus its contract.

    Transport only calls ``predictor`` under ``torch.no_grad`` with inputs moved to
    ``device``; it never moves, rebuilds, switches the mode of, or closes the predictor.
    Put it on ``device`` and in eval mode before running inference. ``checkpoint_path``
    and ``predictor_identity`` are provenance only and may be ``None``. Without default
    output directories every call needs an explicit output path.
    """

    predictor: Predictor
    contract: PredictionContract
    device: torch.device
    checkpoint_path: Path | None = None
    predictor_identity: str | None = None
    default_single_output_dir: Path | None = None
    default_directory_output_dir: Path | None = None


RuntimeFactory: TypeAlias = Callable[[], InferenceRuntime]


@dataclass(frozen=True)
class SingleInferenceResult:
    input_paths: dict[str, Path]
    #: One published file per output name, in contract order.
    output_paths: dict[str, Path]
    image_size: tuple[int, int]
    mode: str
    device: str
    checkpoint_path: Path | None = None
    predictor_identity: str | None = None


@dataclass(frozen=True)
class DirectoryInferenceResult:
    input_dirs: dict[str, Path]
    output_dir: Path
    image_size: tuple[int, int]
    device: str
    results: tuple[SingleInferenceResult, ...]
    checkpoint_path: Path | None = None
    predictor_identity: str | None = None


def _check_input_names(runtime: InferenceRuntime, names: Mapping[str, object]) -> None:
    expected = runtime.contract.input_names
    missing = [name for name in expected if name not in names]
    extra = [name for name in names if name not in expected]
    if missing or extra:
        raise ValueError(
            f"Input names must match the prediction contract {expected}: "
            f"missing={missing}, extra={extra}"
        )


def _sample_id_from_input_path(input_path: Path) -> str:
    stem = input_path.stem
    if stem.endswith("_source"):
        return stem[: -len("_source")]
    return stem


def _generated_paths_for_input(
    runtime: InferenceRuntime, output_dir: Path, input_path: Path, output_suffix: str
) -> dict[str, Path]:
    sample_id = _sample_id_from_input_path(input_path)
    return {
        name: generated_path(output_dir, sample_id, name, output_suffix)
        for name in runtime.contract.output_names
    }


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
    return build_model_input_transform(None)


def _predict_images(
    images: dict[str, Image.Image],
    runtime: InferenceRuntime,
    transform: transforms.Compose,
) -> dict[str, torch.Tensor]:
    """One predictor call; every named output of the first batch item, on the CPU."""
    inputs: dict[str, torch.Tensor] = {}
    for name in runtime.contract.input_names:
        source_tensor = transform(images[name])
        if not isinstance(source_tensor, torch.Tensor):
            raise TypeError("Inference transform must return a torch.Tensor")
        inputs[name] = source_tensor.unsqueeze(0)
    outputs = predict_batch(
        runtime.predictor, inputs, runtime.device, runtime.contract.output_names
    )
    return {name: output[0].cpu() for name, output in outputs.items()}


def _run_resized_prediction(
    images: dict[str, Image.Image], runtime: InferenceRuntime
) -> dict[str, torch.Tensor]:
    transform = build_inference_transform(runtime.contract.image_size)
    return _predict_images(images, runtime, transform)


def _run_tiled_prediction(
    images: dict[str, Image.Image],
    runtime: InferenceRuntime,
    tile_overlap: int,
) -> dict[str, torch.Tensor]:
    """Traverse the input tiles once; each tile's prediction feeds every output."""
    image_size = runtime.contract.image_size
    _validate_tile_overlap(image_size, tile_overlap)

    image_w, image_h = images[runtime.contract.input_names[0]].size
    tile_w, tile_h = image_size
    stride_w = tile_w - tile_overlap
    stride_h = tile_h - tile_overlap
    x_starts = _tile_starts(image_w, tile_w, stride_w)
    y_starts = _tile_starts(image_h, tile_h, stride_h)

    transform = _build_no_resize_transform()
    accumulators = {
        name: torch.zeros((3, image_h, image_w), dtype=torch.float32)
        for name in runtime.contract.output_names
    }
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
            # The same-grid contract was validated, so this only drops the padding.
            predicted = _predict_images(tiles, runtime, transform)
            for name, accumulator in accumulators.items():
                accumulator[:, y : y + actual_h, x : x + actual_w] += predicted[name][
                    :, :actual_h, :actual_w
                ]
            weights[:, y : y + actual_h, x : x + actual_w] += 1.0

    return {
        name: (accumulator / weights.clamp_min(1.0)).clamp(0, 1)
        for name, accumulator in accumulators.items()
    }


def _write_tiled_rgb(
    readers: Mapping[str, RegionImageReader],
    output_paths: Mapping[str, Path],
    runtime: InferenceRuntime,
    tile_overlap: int,
) -> None:
    """Write one raw RGB file per output from a single tile traversal of the inputs.

    Each output has its own bounded float32 memmap accumulator next to its raw file.
    """
    image_size = runtime.contract.image_size
    _validate_tile_overlap(image_size, tile_overlap)

    image_w, image_h = readers[runtime.contract.input_names[0]].size
    tile_w, tile_h = image_size
    x_starts = _tile_starts(image_w, tile_w, tile_w - tile_overlap)
    y_starts = _tile_starts(image_h, tile_h, tile_h - tile_overlap)
    x_weights = np.zeros(image_w, dtype=np.uint32)
    y_weights = np.zeros(image_h, dtype=np.uint32)
    for x in x_starts:
        x_weights[x : min(x + tile_w, image_w)] += 1
    for y in y_starts:
        y_weights[y : min(y + tile_h, image_h)] += 1

    accumulator_paths = {name: path.with_suffix(".float32") for name, path in output_paths.items()}
    accumulators: dict[str, np.memmap] = {}
    transform = _build_no_resize_transform()
    try:
        for name, path in accumulator_paths.items():
            accumulators[name] = np.memmap(
                path, mode="w+", dtype=np.float32, shape=(image_h, image_w, 3)
            )
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
                predicted = _predict_images(images, runtime, transform)
                for name, accumulator in accumulators.items():
                    accumulator[y : y + actual_h, x : x + actual_w] += (
                        predicted[name][:, :actual_h, :actual_w].permute(1, 2, 0).numpy()
                    )

        for name, accumulator in accumulators.items():
            output = np.memmap(
                output_paths[name], mode="w+", dtype=np.uint8, shape=(image_h, image_w, 3)
            )
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
        accumulators.clear()
        for path in accumulator_paths.values():
            path.unlink(missing_ok=True)


def _shared_wsi_metadata(readers: Mapping[str, RegionImageReader]) -> ImageMetadata:
    """Output geometry is the shared input grid; MPP is carried per axis only when every
    input provides it and all values agree. Known conflicts fail even if another input
    is uncalibrated; any missing value leaves that output axis unknown."""
    metadata = {name: reader.metadata for name, reader in readers.items()}
    width, height = next(iter(readers.values())).size
    mpp: dict[str, float | None] = {}
    for axis in ("x", "y"):
        known = {
            name: value
            for name, item in metadata.items()
            if (value := getattr(item, f"mpp_{axis}")) is not None
        }
        values = list(known.values())
        if any(not math.isclose(v, values[0], rel_tol=MPP_REL_TOLERANCE) for v in values):
            raise ValueError(f"Input mpp_{axis} values conflict: {known}")
        mpp[axis] = values[0] if len(values) == len(metadata) else None
    return ImageMetadata(width=width, height=height, mpp_x=mpp["x"], mpp_y=mpp["y"])


def _require_scratch_space(directory: Path, width: int, height: int, outputs: int = 1) -> None:
    # Lower bound only: every output's float32 accumulator and uint8 raw RGB coexist
    # while finalizing; the compressed pyramids written next to them are extra.
    required = outputs * width * height * 3 * (np.dtype(np.float32).itemsize + 1)
    available = shutil.disk_usage(directory).free
    if available < required:
        raise OSError(
            f"WSI inference of {outputs} output(s) needs at least {required} bytes "
            f"({required / 2**30:.2f} GiB) of scratch space on the filesystem of "
            f"{directory}, but only {available} bytes "
            f"({available / 2**30:.2f} GiB) are free"
        )


def _run_wsi_prediction(
    readers: dict[str, OpenSlideRegionImageReader],
    output_paths: Mapping[str, Path],
    runtime: InferenceRuntime,
    tile_overlap: int,
) -> None:
    """Predict every named output of a WSI from one tile traversal of the inputs."""
    if any(path.suffix.lower() not in {".tif", ".tiff"} for path in output_paths.values()):
        raise ValueError("Full-resolution WSI output must use .tif or .tiff")

    metadata = _shared_wsi_metadata(readers)
    for path in output_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    for directory in {path.parent for path in output_paths.values()}:
        _require_scratch_space(directory, metadata.width, metadata.height, len(output_paths))
    # Scratch lives next to each output so publication is an atomic same-filesystem
    # replace; the directories (raw RGB, accumulators, unpublished TIFFs) are removed.
    with ExitStack() as stack:
        raw_paths = {
            name: Path(
                stack.enter_context(
                    tempfile.TemporaryDirectory(prefix=f".{path.stem}.", dir=path.parent)
                )
            )
            / "generated.rgb"
            for name, path in output_paths.items()
        }
        _write_tiled_rgb(readers, raw_paths, runtime, tile_overlap)
        for name, path in output_paths.items():
            write_pyramidal_tiff_from_raw_rgb(raw_paths[name], path, metadata)


def _default_output_dir(directory: Path | None, kind: str) -> Path:
    if directory is None:
        raise ValueError(
            f"No output path was given and the inference runtime has no default {kind} "
            "output directory; pass an explicit output path"
        )
    return directory


def _resolve_output_paths(
    runtime: InferenceRuntime,
    input_path: Path,
    output: Path | None,
    *,
    output_format: str = "same",
) -> dict[str, Path]:
    """Destinations of every output: an explicit file only for one output, else a directory.

    Several outputs never share one file: an explicit ``output`` must then be a directory
    that receives the canonical ``<output>/<sample>_generated`` layout.
    """
    names = runtime.contract.output_names
    if output is not None and len(names) == 1:
        return {names[0]: output}
    if output is not None and (output.is_file() or output.suffix.lower() in VALID_IMAGE_EXTENSIONS):
        raise ValueError(
            f"A single output file {output} cannot hold the {len(names)} outputs {list(names)}; "
            "pass an output directory"
        )
    output_dir = output or _default_output_dir(runtime.default_single_output_dir, "single-image")
    return _generated_paths_for_input(
        runtime, output_dir, input_path, _output_suffix_for_input(input_path, output_format)
    )


def _run_one_image(
    runtime: InferenceRuntime,
    input_images: dict[str, Path],
    *,
    output_paths: dict[str, Path],
    mode: SingleInferenceMode = "auto",
    tile_overlap: int = DEFAULT_TILE_OVERLAP,
) -> SingleInferenceResult:
    input_names = runtime.contract.input_names
    if tuple(input_images) != input_names:
        raise ValueError(
            f"Input paths must match contract input order {input_names}, got {tuple(input_images)}"
        )
    if tuple(output_paths) != runtime.contract.output_names:
        raise ValueError(
            f"Output paths must match contract output order {runtime.contract.output_names}"
        )
    if len({path.resolve() for path in output_paths.values()}) != len(output_paths):
        raise ValueError(f"Outputs must be written to distinct paths: {output_paths}")
    for name, input_path in input_images.items():
        if not input_path.is_file():
            raise FileNotFoundError(f"Input image {name} not found: {input_path}")
        _validate_supported_image_path(input_path, label=f"input_image[{name}]")
        for output_path in output_paths.values():
            if input_path.resolve() == output_path.resolve():
                raise ValueError(f"Output path would overwrite input {name}: {output_path}")
    for name, output_path in output_paths.items():
        _validate_supported_image_path(output_path, label=f"output_image[{name}]")

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
        resolved_mode = _resolve_mode(
            requested_mode, first_reader.size, runtime.contract.image_size
        )
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
                "Full-resolution large-image tiled inference requires every input to be "
                "OpenSlide-compatible and use the OpenSlide backend"
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
                output_paths,
                runtime,
                tile_overlap,
            )
            outputs = None
        else:
            images = {name: open_rgb(input_path) for name, input_path in input_images.items()}
            if resolved_mode == "resize":
                outputs = _run_resized_prediction(images, runtime)
            else:
                outputs = _run_tiled_prediction(images, runtime, tile_overlap)
    finally:
        for reader in readers.values():
            reader.close()

    if outputs is not None:
        for name, output in outputs.items():
            save_rgb(output, output_paths[name])

    logger.info(
        "Single-image inference complete: %s -> %s (mode=%s)",
        input_images,
        output_paths,
        resolved_mode,
    )
    return SingleInferenceResult(
        input_paths=dict(input_images),
        output_paths=dict(output_paths),
        image_size=runtime.contract.image_size,
        mode=resolved_mode,
        device=str(runtime.device),
        checkpoint_path=runtime.checkpoint_path,
        predictor_identity=runtime.predictor_identity,
    )


def _collect_input_images(input_dir: Path, *, recursive: bool = False) -> tuple[Path, ...]:
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


def _run_single_image_inference(
    runtime: InferenceRuntime,
    input_images: dict[str, Path],
    output_image: Path | None = None,
    *,
    mode: SingleInferenceMode = "auto",
    tile_overlap: int = DEFAULT_TILE_OVERLAP,
    output_format: str = "same",
) -> SingleInferenceResult:
    _check_input_names(runtime, input_images)
    input_names = runtime.contract.input_names
    input_paths = {name: Path(input_images[name]) for name in input_names}
    output_paths = _resolve_output_paths(
        runtime,
        input_paths[input_names[0]],
        output_image,
        output_format=output_format,
    )
    return _run_one_image(
        runtime,
        input_paths,
        output_paths=output_paths,
        mode=mode,
        tile_overlap=tile_overlap,
    )


def _resolve_runtime(runtime: InferenceRuntime | RuntimeFactory) -> InferenceRuntime:
    return runtime if isinstance(runtime, InferenceRuntime) else runtime()


def _run_image_directory_inference(
    runtime: InferenceRuntime | RuntimeFactory,
    input_dirs: dict[str, Path],
    output_dir: Path | None = None,
    *,
    recursive: bool = False,
    mode: SingleInferenceMode = "auto",
    tile_overlap: int = DEFAULT_TILE_OVERLAP,
    output_format: str = "same",
) -> DirectoryInferenceResult:
    if not input_dirs:
        raise ValueError("At least one input directory is required.")
    roots = {name: Path(path) for name, path in input_dirs.items()}
    first_name = next(iter(roots))
    first_root = roots[first_name]
    images_by_name = {
        name: _collect_input_images(root, recursive=recursive) for name, root in roots.items()
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
    if output_dir is not None and output_dir.suffix:
        raise NotADirectoryError(
            f"Output path for directory inference must be a directory: {output_dir}"
        )
    runtime = _resolve_runtime(runtime)
    _check_input_names(runtime, roots)
    ordered_names = runtime.contract.input_names
    resolved_output_dir = output_dir or _default_output_dir(
        runtime.default_directory_output_dir, "directory"
    )
    if resolved_output_dir.exists() and not resolved_output_dir.is_dir():
        raise NotADirectoryError(
            f"Output path for directory inference must be a directory: {resolved_output_dir}"
        )
    planned: list[tuple[dict[str, Path], dict[str, Path]]] = []
    claimed: dict[Path, Path] = {}
    for relative_path in sorted(first_relative):
        source_paths = {name: roots[name] / relative_path for name in ordered_names}
        first_source = source_paths[ordered_names[0]]
        relative_parent = relative_path.parent if recursive else Path()
        output_suffix = _output_suffix_for_input(first_source, output_format)
        output_paths = _generated_paths_for_input(
            runtime, resolved_output_dir / relative_parent, first_source, output_suffix
        )
        for output_path in output_paths.values():
            if output_path in claimed:
                raise ValueError(
                    f"Inputs {claimed[output_path]} and {first_source} would both be written "
                    f"to {output_path}"
                )
            claimed[output_path] = first_source
        planned.append((source_paths, output_paths))
    results = [
        _run_one_image(
            runtime,
            source_paths,
            output_paths=output_paths,
            mode=mode,
            tile_overlap=tile_overlap,
        )
        for source_paths, output_paths in planned
    ]
    return DirectoryInferenceResult(
        input_dirs={name: roots[name] for name in ordered_names},
        output_dir=resolved_output_dir,
        image_size=runtime.contract.image_size,
        device=str(runtime.device),
        results=tuple(results),
        checkpoint_path=runtime.checkpoint_path,
        predictor_identity=runtime.predictor_identity,
    )


def run_image_path_inference(
    runtime: InferenceRuntime | RuntimeFactory,
    input_paths: Mapping[str, Path],
    output_path: Path | None = None,
    *,
    recursive: bool = False,
    mode: SingleInferenceMode = "auto",
    tile_overlap: int = DEFAULT_TILE_OVERLAP,
    output_format: str = "same",
) -> SingleInferenceResult | DirectoryInferenceResult:
    """Translate named input files, or directories of paired files, with one predictor.

    ``runtime`` is an already constructed ``InferenceRuntime`` or a factory called
    lazily once the inputs have been validated (directory pairing is checked first).
    """
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
        return _run_image_directory_inference(
            runtime,
            paths,
            output_path,
            recursive=recursive,
            mode=mode,
            tile_overlap=tile_overlap,
            output_format=output_format,
        )
    return _run_single_image_inference(
        _resolve_runtime(runtime),
        paths,
        output_path,
        mode=mode,
        tile_overlap=tile_overlap,
        output_format=output_format,
    )
