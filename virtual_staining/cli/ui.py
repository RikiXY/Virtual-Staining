from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from virtual_staining.cli._common import add_log_level_argument, configure_logging
from virtual_staining.ui.app import run_ui

CHECKPOINT_DIRECTORY_ENV = "VIRTUAL_STAINING_CHECKPOINT_DIR"
OUTPUT_DIRECTORY_ENV = "VIRTUAL_STAINING_OUTPUT_DIR"
RESULTS_DIRECTORY_ENV = "VIRTUAL_STAINING_RESULTS_DIR"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vs-ui",
        description="Launch the Virtual-Staining inference and experiments interface.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=os.environ.get(CHECKPOINT_DIRECTORY_ENV, "checkpoints"),
        help=(
            "Directory searched recursively for .pth checkpoints "
            f"(default: ${CHECKPOINT_DIRECTORY_ENV} or ./checkpoints)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=os.environ.get(OUTPUT_DIRECTORY_ENV, "outputs"),
        help=(
            "Initial directory for saved images and provenance "
            f"(default: ${OUTPUT_DIRECTORY_ENV} or ./outputs)."
        ),
    )
    parser.add_argument(
        "--results-dir",
        default=os.environ.get(RESULTS_DIRECTORY_ENV, "results"),
        help=(
            "Directory searched for experiment runs "
            f"(default: ${RESULTS_DIRECTORY_ENV} or ./results)."
        ),
    )
    parser.add_argument("--host", default="0.0.0.0", help="Interface to bind (default: 0.0.0.0).")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind (default: 8080).")
    add_log_level_argument(parser)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    package_logger = logging.getLogger("virtual_staining")
    old_propagate = package_logger.propagate
    configure_logging(args.log_level)
    try:
        run_ui(
            Path(args.checkpoint_dir),
            Path(args.output_dir),
            Path(args.results_dir),
            host=args.host,
            port=args.port,
        )
    finally:
        package_logger.propagate = old_propagate


if __name__ == "__main__":
    main()
