from __future__ import annotations

import argparse
import sys
from pathlib import Path

from virtual_staining.applications.pipeline import VALID_STAGES
from virtual_staining.cli._common import add_config_argument


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vs config",
        description="Resolve or check a run config without running any stage.",
    )
    commands = parser.add_subparsers(dest="action", metavar="ACTION", required=True)
    resolve = commands.add_parser(
        "resolve", help="Print (or write) the deterministic resolved YAML of a run config."
    )
    add_config_argument(resolve)
    resolve.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write the resolved YAML here instead of stdout; never overwrites.",
    )
    check = commands.add_parser(
        "check", help="Validate a run config, optionally with read-only asset preflight."
    )
    add_config_argument(check)
    check.add_argument(
        "--stages",
        nargs="+",
        choices=VALID_STAGES,
        default=(),
        help="Stages to check, in the order they would run.",
    )
    check.add_argument(
        "--assets",
        action="store_true",
        help="Also check stage inputs read-only (paths, schemas, membership, groups).",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    from virtual_staining.applications.config_authoring import (
        inspect_run_yaml,
        preflight,
        write_config_yaml,
    )

    args = _build_parser().parse_args(argv)
    try:
        inspection = inspect_run_yaml(Path(args.config))
    except (OSError, TypeError, ValueError) as exc:
        print(f"invalid config {args.config}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    if args.action == "resolve":
        if args.output is None:
            sys.stdout.write(inspection.resolved_yaml)
        else:
            try:
                write_config_yaml(inspection.resolved_yaml, args.output)
            except FileExistsError as exc:
                print(f"refusing to overwrite existing {args.output}", file=sys.stderr)
                raise SystemExit(1) from exc
        return

    report = preflight(inspection.config, args.stages, depth="assets" if args.assets else "config")
    print(f"config_sha256: {inspection.resolved_sha256}")
    print(f"depth: {report.depth}")
    print(f"stages: {' '.join(report.stages) or '-'}")
    for check in report.checks:
        print(f"[{check.status}] {check.check_id}: {check.message}")
    print(f"content_verified: {str(report.content_verified).lower()}")
    for limitation in report.limitations:
        print(f"limitation: {limitation}")
    print(f"valid: {str(report.valid).lower()}")
    if not report.valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
