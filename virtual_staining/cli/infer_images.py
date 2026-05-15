from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from virtual_staining.applications.infer_images import infer_images
from virtual_staining.config.run import RunConfig
from virtual_staining.inference.single import (
    DEFAULT_TILE_OVERLAP,
    SUPPORTED_OUTPUT_FORMATS,
    DirectoryInferenceResult,
    SingleInferenceResult,
)
from virtual_staining.utils.console import print_info, print_section, style


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vs-infer-images",
        description="Run Pix2Pix inference on one image file or a directory of images.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the run config YAML. The inference checkpoint is read from this file.",
    )
    parser.add_argument(
        "--input",
        "--input-image",
        dest="input_path",
        required=True,
        help="Path to an image file or a directory containing images.",
    )
    parser.add_argument(
        "--output",
        "--output-image",
        dest="output_path",
        default=None,
        help=(
            "Output image path for file input, or output directory for directory input. "
            "If omitted, writes to inference.output_dir or artifacts/output_images."
        ),
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="When --input is a directory, process supported images recursively.",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "resize", "tile"],
        default="auto",
        help=(
            "Inference mode. auto preserves full size with tiled inference unless an input "
            "already matches image_size; resize forces one resized patch per image."
        ),
    )
    parser.add_argument(
        "--tile-overlap",
        type=int,
        default=DEFAULT_TILE_OVERLAP,
        help="Overlap in pixels between adjacent tiles for tiled inference.",
    )
    parser.add_argument(
        "--output-format",
        choices=sorted(SUPPORTED_OUTPUT_FORMATS),
        default="same",
        help="Output extension to use when an output file is not provided.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser


def _print_file_result(result: SingleInferenceResult) -> None:
    print_info("Input", str(result.input_path))
    print_info("Checkpoint", str(result.checkpoint_path))
    print_info("Mode", result.mode)
    print_info("Generated", style(str(result.output_path), "bold", "magenta"))


def _print_directory_result(result: DirectoryInferenceResult) -> None:
    print_info("Input dir", str(result.input_dir))
    print_info("Checkpoint", str(result.checkpoint_path))
    print_info("Images", str(len(result.results)))
    print_info("Output dir", style(str(result.output_dir), "bold", "magenta"))


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        stream=sys.stdout,
    )

    config_path = Path(args.config).resolve()
    config = RunConfig.from_yaml(config_path)
    output_path = Path(args.output_path) if args.output_path is not None else None
    result = infer_images(
        config,
        Path(args.input_path),
        output_path,
        recursive=args.recursive,
        mode=args.mode,
        tile_overlap=args.tile_overlap,
        output_format=args.output_format,
    )

    print_section("Image inference")
    if isinstance(result, DirectoryInferenceResult):
        _print_directory_result(result)
    else:
        _print_file_result(result)


if __name__ == "__main__":
    main()
