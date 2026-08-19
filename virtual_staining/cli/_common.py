from __future__ import annotations

import argparse
import logging
import sys

LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")


def add_config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, help="Path to the run config YAML.")


def add_log_level_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--log-level", default="INFO", choices=LOG_LEVELS)


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        stream=sys.stdout,
    )
