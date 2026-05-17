from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from virtual_staining.applications.run_queue import run_queue


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vs-run-queue",
        description="Execute a local queue of configurable pipeline stages.",
    )
    parser.add_argument(
        "--queue",
        required=True,
        help="Path to the queue YAML file.",
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

    queue_path = Path(args.queue).resolve()
    state = run_queue(queue_path)
    if state.status == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
