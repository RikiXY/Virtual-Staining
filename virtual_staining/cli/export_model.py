from __future__ import annotations

import argparse
from pathlib import Path

from virtual_staining.applications.export_model import (
    ExportCheckpointSelection,
    ModelBundle,
    export_model_bundle,
)
from virtual_staining.cli._output import print_info, print_section, style


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vs export-model",
        description=(
            "Export selected checkpoints of one tracked run, with their tracked training "
            "configs, as a portable local model bundle. Only built-in method definitions are "
            "available here; export external methods from Python with explicit Definitions. "
            "Export does not imply permission to redistribute the bundle."
        ),
        epilog=(
            "Example:\n"
            "  vs export-model \\\n"
            "      --run-path local_workspace/results/RUN_NAME \\\n"
            "      --output local_workspace/bundles/RUN_NAME \\\n"
            "      --best val_ssim --top-k val_ssim 2 --latest\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--run-path", type=Path, required=True, help="Tracked run directory.")
    parser.add_argument(
        "--output", type=Path, required=True, help="Bundle directory to create (must not exist)."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        action="append",
        default=[],
        metavar="PATH",
        help="A checkpoint file in RUN/checkpoints/ (relative to it or absolute). Repeatable.",
    )
    parser.add_argument("--latest", action="store_true", help="The latest epoch checkpoint.")
    parser.add_argument(
        "--best",
        action="append",
        default=[],
        metavar="METRIC",
        help="Rank 1 of METRIC in RUN/checkpoints/best.json. Repeatable.",
    )
    parser.add_argument(
        "--top-k",
        nargs=2,
        action="append",
        default=[],
        metavar=("METRIC", "RANK"),
        help="Rank RANK of METRIC in RUN/checkpoints/best.json. Repeatable.",
    )
    return parser


def _selections(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> list[ExportCheckpointSelection]:
    selections = [
        ExportCheckpointSelection("explicit", checkpoint_path=path) for path in args.checkpoint
    ]
    if args.latest:
        selections.append(ExportCheckpointSelection("latest"))
    selections += [ExportCheckpointSelection("best", metric=metric) for metric in args.best]
    for metric, rank in args.top_k:
        if not rank.isdigit() or int(rank) <= 0:
            parser.error(f"--top-k RANK must be a positive integer, got {rank!r}")
        selections.append(ExportCheckpointSelection("top_k", metric=metric, rank=int(rank)))
    if not selections:
        parser.error("at least one of --checkpoint, --latest, --best or --top-k is required")
    return selections


def _print_result(bundle: ModelBundle) -> None:
    print_section("Model bundle")
    for record in bundle.index["selections"]:
        label = record["policy"] + "".join(
            f" {record[key]}" for key in ("metric", "rank") if record[key] is not None
        )
        print_info(label, f"{record['checkpoint']} (epoch {record['epoch']})")
    print_section("Done")
    print_info("Bundle written to", style(str(bundle.root), "bold", "magenta"))


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    selections = _selections(parser, args)
    try:
        bundle = export_model_bundle(args.run_path, args.output, selections)
    except (FileNotFoundError, FileExistsError, NotADirectoryError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    _print_result(bundle)
