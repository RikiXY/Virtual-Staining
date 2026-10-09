"""Read-only run-configuration authoring, inspection, and preflight.

Everything here resolves through ``RunConfig.from_mapping`` and the existing stage/data
owners; nothing opens a session, runs a stage, decodes an image or checkpoint, builds a
model, probes a device, or writes run/dataset state. The only write is an explicitly
requested config file, published without ever replacing an existing path.
"""

from __future__ import annotations

import csv
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from virtual_staining.applications.evaluate import (
    evaluation_generated_dir,
    evaluation_protocol,
    load_paired_evaluation_manifest,
    paired_samples,
    unpaired_generated_collection,
    unpaired_reference_collection,
)
from virtual_staining.config.loader import dump_yaml_mapping, load_yaml_mapping
from virtual_staining.config.run import RunConfig
from virtual_staining.data.consumption import validate_groups
from virtual_staining.data.layout import DatasetLayout
from virtual_staining.data.manifest import (
    load_manifest_or_raise,
    load_set_groups,
    paired_record_rows,
    prepared_split_unit,
    require_model_modalities,
)
from virtual_staining.data.slide_sets import resolve_slide_sets
from virtual_staining.data.unpaired import resolve_domain_collections
from virtual_staining.definitions import Definitions
from virtual_staining.experiment.run_layout import RunLayout
from virtual_staining.inference.runner import (
    inference_direction,
    inference_input_names,
    inference_output_dir,
    inference_output_names,
    resolve_inference_checkpoint,
)
from virtual_staining.split_contract import TEST_SPLIT, TRAIN_SPLIT, VAL_SPLIT
from virtual_staining.utils.files import publish_file_no_replace
from virtual_staining.utils.hashing import sha256_bytes

FieldOrigin = Literal["supplied", "defaulted"]
PreflightStatus = Literal["valid", "invalid", "planned", "unverified", "not_applicable"]
PreflightDepth = Literal["config", "assets"]

CONFIG_LIMITATION = (
    "depth=config: only the configuration was resolved; no dataset, inventory, manifest, "
    "mask, checkpoint, or generated-image path was inspected."
)
ASSET_LIMITATIONS = (
    "content_verified=false: paths, schemas, membership, supplied group metadata, and "
    "checkpoint selection were checked; no file was hashed, decoded, or deserialized.",
    "No content-level duplicate or leakage check was run; no patient independence or "
    "scientific validity is claimed.",
    "A successful preflight is not a frozen input snapshot: assets may change afterwards, "
    "and tracked execution re-resolves, re-validates, and freezes what it actually consumes.",
    "planned checks were not verified: an earlier selected stage is expected to (re)produce "
    "that artifact.",
)
# What a failing owner raises for a missing, malformed, or inconsistent asset.
_ASSET_ERRORS = (OSError, ValueError, KeyError, csv.Error)


@dataclass(frozen=True)
class RunConfigInspection:
    """One resolved run configuration in its authored and resolved forms.

    ``authored`` is the caller's mapping (plain dicts/lists, order kept) and
    ``authored_yaml`` renders exactly it; ``resolved``/``resolved_yaml`` are
    ``RunConfig.to_dict()`` rendered with the tracked-snapshot serializer, so
    ``resolved_sha256`` equals the hash a tracked stage records. ``origins`` maps every
    resolved leaf path to ``supplied`` or ``defaulted``; it is explanatory only.
    """

    authored: dict[str, Any]
    config: RunConfig
    resolved: dict[str, Any]
    authored_yaml: str
    resolved_yaml: str
    resolved_sha256: str
    origins: dict[str, FieldOrigin]


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(item) for item in value]
    return value


def _leaves(value: Any, path: str = "") -> dict[str, Any]:
    """Leaf paths like ``a.b[0].c``; non-empty mappings and lists of mappings recurse."""
    if isinstance(value, dict) and value:
        return {
            leaf: item
            for key, child in value.items()
            for leaf, item in _leaves(child, f"{path}.{key}" if path else str(key)).items()
        }
    if isinstance(value, list) and value and all(isinstance(item, dict) for item in value):
        return {
            leaf: item
            for index, child in enumerate(value)
            for leaf, item in _leaves(child, f"{path}[{index}]").items()
        }
    return {path: value}


def field_origins(
    authored: Mapping[str, Any], resolved: Mapping[str, Any]
) -> dict[str, FieldOrigin]:
    """Classify each resolved leaf as supplied by ``authored`` or filled by its owner.

    A leaf is ``supplied`` when the authored mapping has the same path, or a non-mapping
    value at an ancestor path that its owner normalized into this leaf.
    """
    supplied = _leaves(_plain(authored))

    def origin(path: str) -> FieldOrigin:
        ancestors = (path[: match.start()] for match in re.finditer(r"[.\[]", path))
        if path in supplied or any(
            key in supplied and not isinstance(supplied[key], dict) for key in ancestors
        ):
            return "supplied"
        return "defaulted"

    return {path: origin(path) for path in sorted(_leaves(dict(resolved)))}


def inspect_run_mapping(
    raw: Mapping[str, Any], definitions: Definitions | None = None
) -> RunConfigInspection:
    """Resolve ``raw`` through ``RunConfig.from_mapping`` and describe both forms."""
    authored = _plain(raw) if isinstance(raw, Mapping) else raw
    config = RunConfig.from_mapping(authored, definitions)
    resolved = config.to_dict()
    resolved_yaml = config.resolved_yaml()
    return RunConfigInspection(
        authored=authored,
        config=config,
        resolved=resolved,
        authored_yaml=dump_yaml_mapping(authored, sort_keys=False),
        resolved_yaml=resolved_yaml,
        resolved_sha256=sha256_bytes(resolved_yaml.encode("utf-8")),
        origins=field_origins(authored, resolved),
    )


def inspect_run_yaml(
    path: str | Path, definitions: Definitions | None = None
) -> RunConfigInspection:
    return inspect_run_mapping(load_yaml_mapping(path), definitions)


def write_config_yaml(text: str, destination: Path) -> Path:
    """Publish already-rendered YAML at ``destination``; ``FileExistsError`` if it exists."""
    return publish_file_no_replace(text.encode("utf-8"), destination)


@dataclass(frozen=True)
class PreflightCheck:
    check_id: str
    stage: str | None
    status: PreflightStatus
    message: str


@dataclass(frozen=True)
class PreflightReport:
    depth: PreflightDepth
    stages: tuple[str, ...]
    checks: tuple[PreflightCheck, ...]
    limitations: tuple[str, ...]
    # Never true here: preflight does not hash or decode anything.
    content_verified: bool = False

    @property
    def valid(self) -> bool:
        """True unless a currently required condition failed; planned/unverified remain."""
        return not any(check.status == "invalid" for check in self.checks)


def preflight(
    config: RunConfig, stages: Sequence[str] = (), *, depth: PreflightDepth = "config"
) -> PreflightReport:
    """Check ``config`` for ``stages`` (in the given order) without executing any of them.

    ``depth="assets"`` adds read-only path/schema/membership/group/checkpoint-selection
    checks through the stage owners. An artifact that an earlier *selected* stage produces
    is reported ``planned`` rather than inspected.
    """
    RunConfig.check_stages(stages)
    if depth not in ("config", "assets"):
        raise ValueError(f"Unsupported preflight depth {depth!r}")
    checks = [PreflightCheck("config.resolve", None, "valid", "configuration resolved")]
    for index, stage in enumerate(stages):
        try:
            config.validate_stages((stage,))
        except (ValueError, TypeError) as exc:
            checks.append(PreflightCheck(f"{stage}.config", stage, "invalid", str(exc)))
            continue
        checks.append(PreflightCheck(f"{stage}.config", stage, "valid", "configuration valid"))
        if depth == "config":
            checks.append(
                PreflightCheck(f"{stage}.assets", stage, "unverified", "asset checks not requested")
            )
        else:
            checks.extend(_STAGE_CHECKS[stage](config, frozenset(stages[:index])))
    limitations = (CONFIG_LIMITATION,) if depth == "config" else ASSET_LIMITATIONS
    return PreflightReport(depth, tuple(stages), tuple(checks), limitations)


def _check(
    check_id: str, stage: str, run: Callable[[], str], *, planned_by: str | None = None
) -> PreflightCheck:
    """Run one owner-backed check; ``planned_by`` names the earlier stage producing its input."""
    if planned_by is not None:
        return PreflightCheck(
            check_id, stage, "planned", f"produced by the earlier selected {planned_by!r} stage"
        )
    try:
        return PreflightCheck(check_id, stage, "valid", run())
    except _ASSET_ERRORS as exc:
        return PreflightCheck(check_id, stage, "invalid", f"{type(exc).__name__}: {exc}")


def _producer(stage: str, preceding: frozenset[str]) -> str | None:
    return stage if stage in preceding else None


def _prepare_checks(config: RunConfig, preceding: frozenset[str]) -> list[PreflightCheck]:
    assert config.preprocessing is not None
    preprocessing = config.preprocessing

    def inventory() -> str:
        return f"{len(resolve_slide_sets(preprocessing))} slide set(s) resolved from inventory"

    return [_check("prepare.inventory", "prepare", inventory)]


def _train_checks(config: RunConfig, preceding: frozenset[str]) -> list[PreflightCheck]:
    if config.data.pairing == "unpaired":
        return [_check("train.domains", "train", lambda: _unpaired_train(config))]
    return [
        _check(
            "train.manifest",
            "train",
            lambda: _paired_train(config),
            planned_by=_producer("prepare", preceding),
        )
    ]


_PAIRED_SPLITS = (TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT)


def _paired_train(config: RunConfig) -> str:
    assert config.model is not None
    assert config.method is not None
    manifest = load_manifest_or_raise(config.project)
    require_model_modalities(manifest, config.model.inputs, config.model.outputs)
    manifest.validate(check_files_exist=True, require_splits={TRAIN_SPLIT, VAL_SPLIT})
    # Train/val rows plus the held-out test rows training checks for group leakage.
    rows = paired_record_rows(
        [record for record in manifest.records if record.split in _PAIRED_SPLITS],
        input_names=config.model.inputs,
        target_names=config.model.outputs,
        include_masks=config.method.definition.requires_foreground_mask(config),
        groups=load_set_groups(config.project),
    )
    result = validate_groups(
        rows,
        config.data.group_validation,
        patch_split=prepared_split_unit(config.project) == "patch",
    )
    counts = {split: len(manifest.filter_split(split)) for split in (TRAIN_SPLIT, VAL_SPLIT)}
    return (
        f"manifest records {counts}; outputs {list(config.model.outputs)}; "
        f"supplied groups: {_groups(result)}"
    )


def _unpaired_train(config: RunConfig) -> str:
    assert config.model is not None
    domain_a, domain_b = config.model.inputs[0], config.model.outputs[0]
    paths, rows = resolve_domain_collections(
        config.data.domains,
        config.project.dataset_root,
        splits=[TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT],
        roles={domain_a: "input", domain_b: "target"},
        group_metadata=config.data.group_metadata,
    )
    result = validate_groups(rows, config.data.group_validation)
    counts = {f"{split}/{domain}": len(items) for (split, domain), items in paths.items()}
    return f"domain collections {counts}; supplied groups: {_groups(result)}"


def _groups(result: Mapping[str, Any]) -> str:
    return f"status={result['status']}, unit={result['unit']} (no content-level check)"


def _infer_checks(config: RunConfig, preceding: frozenset[str]) -> list[PreflightCheck]:
    assert config.inference is not None
    paths = RunLayout.from_project(config.project)

    def manifest() -> str:
        loaded = load_manifest_or_raise(config.project)
        assert config.model is not None
        require_model_modalities(loaded, config.model.inputs, config.model.outputs)
        loaded.validate(check_files_exist=True, require_splits={TEST_SPLIT})
        return (
            f"{len(loaded.filter_split(TEST_SPLIT))} test record(s); direction="
            f"{inference_direction(config)}, prediction inputs="
            f"{list(inference_input_names(config))}, outputs="
            f"{list(inference_output_names(config))}"
        )

    def checkpoint() -> str:
        path = resolve_inference_checkpoint(config, paths)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return f"selected {path} (not deserialized)"

    explicit = config.inference.checkpoint_path
    # Training writes only into this run's checkpoints/; an explicit path elsewhere is not
    # something an earlier train stage produces.
    trained_here = explicit is None or (paths.root / explicit).resolve().is_relative_to(
        paths.checkpoints_dir.resolve()
    )
    return [
        _check("infer.manifest", "infer", manifest, planned_by=_producer("prepare", preceding)),
        _check(
            "infer.checkpoint",
            "infer",
            checkpoint,
            planned_by=_producer("train", preceding) if trained_here else None,
        ),
    ]


def _evaluate_checks(config: RunConfig, preceding: frozenset[str]) -> list[PreflightCheck]:
    paths = RunLayout.from_project(config.project)
    generated_dir = evaluation_generated_dir(config, paths)
    infers_here = "infer" in preceding and generated_dir == inference_output_dir(config, paths)
    generated_by = "infer" if infers_here else None
    protocol = evaluation_protocol(config)
    if protocol == "unpaired":

        def reference() -> str:
            return f"{len(unpaired_reference_collection(config))} reference image(s)"

        def generated() -> str:
            found = unpaired_generated_collection(config, generated_dir)
            return f"{len(found)} generated image(s) under {generated_dir}"

        return [
            _check("evaluate.reference", "evaluate", reference),
            _check("evaluate.generated", "evaluate", generated, planned_by=generated_by),
        ]

    manifest_check = _check(
        "evaluate.manifest",
        "evaluate",
        lambda: _paired_evaluation_manifest(config),
        planned_by=_producer("prepare", preceding),
    )
    if generated_by is None and manifest_check.status != "valid":
        generated_check = PreflightCheck(
            "evaluate.generated",
            "evaluate",
            "unverified",
            "expected generated files derive from the held-out manifest, which was not checked",
        )
    else:
        generated_check = _check(
            "evaluate.generated",
            "evaluate",
            lambda: _paired_generated(config, generated_dir),
            planned_by=generated_by,
        )
    return [manifest_check, generated_check]


def _paired_evaluation_manifest(config: RunConfig) -> str:
    manifest = load_paired_evaluation_manifest(config)
    slide_sets = DatasetLayout.from_project(config.project).slide_sets_path
    if not slide_sets.is_file():
        raise FileNotFoundError(f"Prepared slide-set metadata not found: {slide_sets}")
    return f"{len(manifest.filter_split(TEST_SPLIT))} aligned test record(s)"


def _paired_generated(config: RunConfig, generated_dir: Path) -> str:
    records = load_paired_evaluation_manifest(config).filter_split(TEST_SPLIT).records
    # One expected artifact per (sample_id, output_name) pair.
    expected = [
        sample.generated_path
        for record in records
        for sample in paired_samples(config, record, generated_dir)
    ]
    missing = [path for path in expected if not path.is_file()]
    strict = config.evaluation is None or config.evaluation.input_failures == "strict"
    if missing and (strict or len(missing) == len(expected)):
        raise FileNotFoundError(
            f"{len(missing)} of {len(expected)} expected generated file(s) missing under "
            f"{generated_dir}, e.g. {missing[0].relative_to(generated_dir)}"
        )
    excluded = f"; {len(missing)} missing will be excluded" if missing else ""
    return f"{len(expected) - len(missing)} of {len(expected)} generated file(s) present{excluded}"


_STAGE_CHECKS: dict[str, Callable[[RunConfig, frozenset[str]], list[PreflightCheck]]] = {
    "prepare": _prepare_checks,
    "train": _train_checks,
    "infer": _infer_checks,
    "evaluate": _evaluate_checks,
}
