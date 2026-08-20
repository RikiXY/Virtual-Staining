from __future__ import annotations

import argparse
from pathlib import Path

from virtual_staining.applications.convert import convert_images
from virtual_staining.cli._output import print_info


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vs convert",
        description=(
            "Recursively convert TIFF files or directories to lossless "
            "OpenSlide-compatible pyramidal BigTIFFs."
        ),
    )
    parser.add_argument(
        "inputs", nargs="+", type=Path, metavar="INPUT", help="TIFF file or directory."
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    outputs = convert_images(
        tuple(path.resolve() for path in args.inputs),
        args.output_dir.resolve(),
    )
    for output in outputs:
        print_info("Converted", str(output))


if __name__ == "__main__":
    main()
