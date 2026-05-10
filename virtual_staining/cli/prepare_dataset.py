from __future__ import annotations

import argparse
import logging
import sys

from virtual_staining.data.builder import DatasetBuilder
from virtual_staining.data.config import PreprocessingConfig


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vs-prepare",
        description="Prepare the virtual staining dataset from raw paired images.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to run config YAML.",
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

    config = PreprocessingConfig.from_yaml(args.config)
    builder = DatasetBuilder(config)
    builder.run_all()


if __name__ == "__main__":
    main()
