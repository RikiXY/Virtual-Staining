from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

from virtual_staining.applications.pipeline import VALID_STAGES, run_stage, run_stages
from virtual_staining.applications.run_queue import run_queue
from virtual_staining.cli import compare, compare_panels, evaluate, infer_images, organize
from virtual_staining.cli._common import (
    add_config_argument,
    add_log_level_argument,
    configure_logging,
)

Command = Callable[[list[str] | None], None]

_COMMAND_HELP = {
    "prepare": "Prepare the paired-image dataset.",
    "run": "Run the complete pipeline or selected stages.",
    "train": "Train the virtual staining model.",
    "infer": "Run inference on the test split.",
    "infer-images": "Run inference on an image or directory.",
    "evaluate": "Evaluate a run or one target/generated pair.",
    "compare": "Compare metric distributions across runs.",
    "panels": "Build source/generated/target comparison panels.",
    "organize": "Organize run outputs by metric ranking.",
    "queue": "Execute pipeline runs from a queue file.",
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vs", description="Virtual staining experiments.")
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    for command, help_text in _COMMAND_HELP.items():
        subparsers.add_parser(command, help=help_text, add_help=False)
    return parser


def _run_stage_command(stage: str, argv: list[str] | None) -> None:
    parser = argparse.ArgumentParser(prog=f"vs {stage}", description=_COMMAND_HELP[stage])
    add_config_argument(parser)
    add_log_level_argument(parser)
    args = parser.parse_args(argv)
    configure_logging(args.log_level)
    run_stage(Path(args.config).resolve(), stage)


def _run_pipeline(argv: list[str] | None) -> None:
    parser = argparse.ArgumentParser(prog="vs run", description=_COMMAND_HELP["run"])
    add_config_argument(parser)
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=VALID_STAGES,
        default=None,
        help="Pipeline stages to execute in the supplied order (default: complete pipeline).",
    )
    add_log_level_argument(parser)
    args = parser.parse_args(argv)
    configure_logging(args.log_level)
    config_path = Path(args.config).resolve()
    if args.stages is None:
        run_stages(config_path)
    else:
        run_stages(config_path, args.stages)


def _run_queue(argv: list[str] | None) -> None:
    parser = argparse.ArgumentParser(prog="vs queue", description=_COMMAND_HELP["queue"])
    parser.add_argument("--queue", required=True, help="Path to the queue YAML file.")
    add_log_level_argument(parser)
    args = parser.parse_args(argv)
    configure_logging(args.log_level)
    if run_queue(Path(args.queue).resolve()).status == "failed":
        raise SystemExit(1)


def _commands() -> dict[str, Command]:
    return {
        "prepare": lambda argv: _run_stage_command("prepare", argv),
        "run": _run_pipeline,
        "train": lambda argv: _run_stage_command("train", argv),
        "infer": lambda argv: _run_stage_command("infer", argv),
        "infer-images": infer_images.main,
        "evaluate": evaluate.main,
        "compare": compare.main,
        "panels": compare_panels.main,
        "organize": organize.main,
        "queue": _run_queue,
    }


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    parser = _build_parser()
    if not args:
        parser.print_help()
        return
    if args[0] in {"-h", "--help"}:
        parser.parse_args(args)
        return
    command = args.pop(0)
    handler = _commands().get(command)
    if handler is None:
        parser.error(f"unknown command: {command}")
    handler(args)


if __name__ == "__main__":
    main()
