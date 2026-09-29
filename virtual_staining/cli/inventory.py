from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from virtual_staining.applications.inventory_authoring import (
    KEY_RULES,
    InventoryRequest,
    preview_inventory,
    write_inventory,
)

_SPEC_HELP = "a dataset_root-relative directory (recursive) or glob such as 'raw/AF/**/*.svs'"


def _named_spec(text: str) -> tuple[str, str]:
    name, separator, spec = text.partition("=")
    if not separator or not name or not spec:
        raise argparse.ArgumentTypeError(f"expected NAME=SPEC, got {text!r}")
    return name, spec


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vs inventory",
        description=(
            "Preview or write the raw paired slide-set inventory (inputs/slide_sets.csv) "
            "from explicit asset mappings. Assets are matched by key only; no image is "
            "opened and no alignment, grouping, or correspondence is inferred."
        ),
    )
    commands = parser.add_subparsers(dest="action", metavar="ACTION", required=True)
    for action, help_text in (
        ("preview", "Show the matched sets and every issue; writes nothing."),
        ("write", "Publish the inventory if the preview is valid; never overwrites."),
    ):
        command = commands.add_parser(action, help=help_text, description=help_text)
        command.add_argument("--dataset-root", type=Path, required=True)
        command.add_argument(
            "--input",
            dest="inputs",
            type=_named_spec,
            action="append",
            required=True,
            metavar="NAME=SPEC",
            help=f"Input modality mapping, repeatable and order preserving; SPEC is {_SPEC_HELP}.",
        )
        command.add_argument("--target-modality", required=True, help="Target modality name.")
        command.add_argument("--target", required=True, metavar="SPEC", help="Target mapping.")
        command.add_argument("--reference", required=True, help="Reference input name.")
        command.add_argument(
            "--input-mask",
            dest="input_masks",
            type=_named_spec,
            action="append",
            default=[],
            metavar="NAME=SPEC",
            help="Optional mask mapping for one named input, repeatable.",
        )
        command.add_argument("--target-mask", metavar="SPEC", help="Optional target mask mapping.")
        command.add_argument(
            "--key",
            choices=KEY_RULES,
            default="relative-path",
            help="relative-path matches full relative paths; relative-stem ignores the "
            "final extension (default: relative-path).",
        )
        command.add_argument(
            "--metadata",
            type=Path,
            help="Optional CSV joined by its 'key' column; relative to --dataset-root.",
        )
        if action == "write":
            command.add_argument(
                "--output",
                type=Path,
                default=None,
                help="Output inside --dataset-root, relative to it "
                "(default: inputs/slide_sets.csv); never overwrites.",
            )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        request = InventoryRequest(
            dataset_root=args.dataset_root,
            inputs=tuple(args.inputs),
            target_modality=args.target_modality,
            target=args.target,
            reference=args.reference,
            input_masks=tuple(args.input_masks),
            target_mask=args.target_mask,
            metadata=args.metadata,
            key_rule=args.key,
        )
    except ValueError as exc:
        parser.error(str(exc))
    try:
        preview = preview_inventory(request)
    except OSError as exc:
        print(f"inventory preview failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print(f"dataset_root: {preview.dataset_root}")
    print(f"key_rule: {request.key_rule}")
    print(f"inputs: {' '.join(request.modalities)}")
    print(f"reference: {request.reference}")
    print(f"target: {request.target_modality}")
    print(f"metadata: {request.metadata or 'none'}")
    print(f"matched_sets: {preview.matched_count}")
    for match in preview.matches:
        print(f"set: {match.slide_set.set_id} key={match.key}")
    for issue in preview.issues:
        print(f"[{issue.kind}] {issue.message}")
    for limitation in preview.limitations:
        print(f"limitation: {limitation}")
    print(f"valid: {str(preview.valid).lower()}")
    if not preview.valid:
        raise SystemExit(1)
    if args.action == "write":
        try:
            path = write_inventory(preview, args.output)
        except FileExistsError as exc:
            print(str(exc), file=sys.stderr)
            raise SystemExit(1) from exc
        except (OSError, ValueError, csv.Error) as exc:
            print(f"inventory not written: {exc}", file=sys.stderr)
            raise SystemExit(1) from exc
        print(f"wrote: {path}")


if __name__ == "__main__":
    main()
