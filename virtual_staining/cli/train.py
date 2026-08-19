from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from virtual_staining.applications.pipeline import run_stage


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vs-train",
        description="Train the Pix2Pix virtual staining model.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the run config YAML.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        stream=sys.stdout,
    )

    config_path = Path(args.config).resolve()
    run_stage(config_path, "train")


if __name__ == "__main__":
    main()
