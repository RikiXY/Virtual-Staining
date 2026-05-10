from __future__ import annotations

from virtual_staining.applications.compare import compare_paired, compare_unpaired
from virtual_staining.evaluation.statistics import resolve_metric_direction


def main() -> None:
    from tools.compare_distributions import build_parser

    parser = build_parser()
    args = parser.parse_args()

    if args.mode is None:
        parser.print_help()
        return

    try:
        args.resolved_higher_is_better = resolve_metric_direction(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    if args.mode == "unpaired":
        compare_unpaired(args)
    elif args.mode == "paired":
        compare_paired(args)
    else:
        raise SystemExit(f"Unsupported comparison mode: {args.mode}")
