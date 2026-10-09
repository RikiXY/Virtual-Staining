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
            "Preview or write paired slide sets (inputs/slide_sets.csv) or independent "
            "unpaired domains (inputs/paths.csv). No images are decoded and no biological "
            "identity, alignment, or correspondence is inferred."
        ),
    )
    commands = parser.add_subparsers(dest="action", metavar="ACTION", required=True)
    for action, help_text in (
        ("preview", "Show source membership and authoring issues; writes nothing."),
        ("write", "Publish the inventory if the preview is valid; never overwrites."),
    ):
        command = commands.add_parser(action, help=help_text, description=help_text)
        command.add_argument("--dataset-root", type=Path, required=True)
        command.add_argument("--pairing", choices=("paired", "unpaired"), default="paired")
        command.add_argument(
            "--domain",
            dest="domains",
            type=_named_spec,
            action="append",
            default=[],
            metavar="NAME=SPEC",
            help=f"Unpaired only: exactly two ordered domain mappings; SPEC is {_SPEC_HELP}.",
        )
        command.add_argument(
            "--input",
            dest="inputs",
            type=_named_spec,
            action="append",
            default=[],
            metavar="NAME=SPEC",
            help=f"Input modality mapping, repeatable and order preserving; SPEC is {_SPEC_HELP}.",
        )
        command.add_argument(
            "--target",
            dest="targets",
            type=_named_spec,
            action="append",
            default=[],
            metavar="NAME=SPEC",
            help="Target modality mapping, repeatable and order preserving; every target is "
            "required for every set.",
        )
        command.add_argument("--reference", help="Paired reference input name (required).")
        command.add_argument(
            "--input-mask",
            dest="input_masks",
            type=_named_spec,
            action="append",
            default=[],
            metavar="NAME=SPEC",
            help="Optional mask mapping for one named input, repeatable.",
        )
        command.add_argument(
            "--target-mask",
            dest="target_masks",
            type=_named_spec,
            action="append",
            default=[],
            metavar="NAME=SPEC",
            help="Optional mask mapping for one named target, repeatable.",
        )
        command.add_argument(
            "--key",
            choices=KEY_RULES,
            default=None,
            help="relative-path matches full relative paths; relative-stem ignores the "
            "final extension (default: relative-path).",
        )
        command.add_argument(
            "--metadata",
            type=Path,
            help="Optional CSV joined by 'key' (paired) or exact root-relative 'path' "
            "(unpaired); relative to --dataset-root.",
        )
        if action == "write":
            command.add_argument(
                "--output",
                type=Path,
                default=None,
                help="Output inside --dataset-root, relative to it "
                "(default: inputs/slide_sets.csv paired, inputs/paths.csv unpaired); "
                "never overwrites.",
            )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.pairing == "unpaired" and any(
        (
            args.inputs,
            args.targets,
            args.reference is not None,
            args.input_masks,
            args.target_masks,
            args.key is not None,
        )
    ):
        parser.error(
            "--input, --target, --reference, --input-mask, --target-mask and --key are paired-only"
        )
    if args.pairing == "paired":
        missing = [
            flag
            for flag, value in (
                ("--input", args.inputs),
                ("--target", args.targets),
                ("--reference", args.reference),
            )
            if not value
        ]
        if missing:
            parser.error(f"the following arguments are required: {', '.join(missing)}")
    try:
        request = InventoryRequest(
            dataset_root=args.dataset_root,
            inputs=tuple(args.inputs),
            targets=tuple(args.targets),
            reference=args.reference or "",
            input_masks=tuple(args.input_masks),
            target_masks=tuple(args.target_masks),
            metadata=args.metadata,
            key_rule=args.key or "relative-path",
            pairing=args.pairing,
            domains=tuple(args.domains),
        )
    except ValueError as exc:
        parser.error(str(exc))
    try:
        preview = preview_inventory(request)
    except OSError as exc:
        print(f"inventory preview failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print(f"dataset_root: {preview.dataset_root}")
    print(f"pairing: {request.pairing}")
    if request.pairing == "unpaired":
        print(f"domains: {' '.join(request.domain_names)}")
        for name, spec in request.domains:
            members = [item.path for item in preview.images if item.domain == name]
            print(f"domain: {name} spec={preview.dataset_root / spec} images={len(members)}")
            for path in members:
                print(f"source: {name} {path}")
    else:
        print(f"key_rule: {request.key_rule}")
        print(f"inputs: {' '.join(request.modalities)}")
        print(f"reference: {request.reference}")
        print(f"targets: {' '.join(request.target_modalities)}")
    print(f"metadata: {request.metadata or 'none'}")
    if request.pairing == "paired":
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
