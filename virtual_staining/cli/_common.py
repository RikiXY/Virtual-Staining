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
    package_logger = logging.getLogger("virtual_staining")
    package_logger.setLevel(logging.DEBUG)
    package_logger.propagate = False
    for handler in tuple(package_logger.handlers):
        if getattr(handler, "_virtual_staining_console", False):
            package_logger.removeHandler(handler)
            handler.close()

    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(getattr(logging, level))
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s"))
    handler._virtual_staining_console = True  # type: ignore[attr-defined]
    package_logger.addHandler(handler)
