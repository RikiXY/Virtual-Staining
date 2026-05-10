from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from virtual_staining.config.run import RunConfig
from virtual_staining.data.builder import DatasetBuilder


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

    config_path = Path(args.config).resolve()
    config_bundle = RunConfig.from_yaml(config_path)
    if config_bundle.preprocessing is None:
        raise SystemExit("Config has no 'preprocessing' section.")
    config = config_bundle.preprocessing
    builder = DatasetBuilder(config)
    builder.run_all()


if __name__ == "__main__":
    main()
