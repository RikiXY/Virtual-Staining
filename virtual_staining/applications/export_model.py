"""Export selected checkpoints of one tracked run as a portable local model bundle.

A bundle is a versioned directory holding the selected v4 checkpoints, the exact
tracked training input/resolved configs and training environment snapshot, and a
``bundle.json`` index with their SHA-256 hashes, the checkpoints' v4 reconstruction
metadata and the selection records that chose them. It is reconstructed with the
normal owners (``RunConfig.from_yaml`` + ``load_inference_generator``) and the same
explicitly supplied definitions; nothing in it names Python code to import.

A bundle is a local research artifact. Exporting one does not imply permission to
redistribute the weights, configs or environment metadata it contains.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import stat
import tempfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from virtual_staining.checkpoint_contract import (
    CheckpointCompatibilityError,
    read_checkpoint,
    validate_checkpoint,
)
from virtual_staining.checkpoint_selection import (
    load_best_checkpoint_record,
    resolve_checkpoint_path,
)
from virtual_staining.config.run import RunConfig
from virtual_staining.definitions import Definitions
from virtual_staining.experiment.run_layout import RunLayout
from virtual_staining.utils.hashing import sha256_file, sha256_file_verified

logger = logging.getLogger(__name__)

BUNDLE_SCHEMA_VERSION = 1
BUNDLE_INDEX = "bundle.json"
BUNDLE_NOTICE = (
    "Local research artifact. The configs are the exact tracked training configs and may "
    "reference the original, possibly unavailable, dataset and result paths. Export does "
    "not imply permission to redistribute these weights or files."
)
_INPUT_CONFIG = "config/input.yaml"
_RESOLVED_CONFIG = "config/resolved.yaml"
_ENVIRONMENT = "metadata/training_environment.json"
_CHECKPOINTS = "checkpoints"
_CONFIG_ROLES = {"input": "training_input", "resolved": "training_resolved"}
# Existing v4 payload fields embedded verbatim in the index (``state`` never is).
_CHECKPOINT_METADATA = (
    "format_version",
    "epoch",
    "config_hash",
    "method",
    "image_size",
    "normalization",
)
_INDEX_KEYS = {
    "schema_version",
    "notice",
    "config",
    "environment",
    "requirements",
    "checkpoints",
    "selections",
}
_SELECTION_KEYS = {"policy", "metric", "rank", "metric_value", "mode", "epoch", "checkpoint"}

ExportPolicy = Literal["explicit", "latest", "best", "top_k"]


@dataclass(frozen=True)
class ExportCheckpointSelection:
    """One checkpoint request: an explicit file, the latest, or a ranked catalog entry.

    ``explicit`` takes ``checkpoint_path`` (relative paths are relative to the run's
    ``checkpoints/``); ``best`` takes ``metric`` (rank 1); ``top_k`` takes ``metric``
    and ``rank``; ``latest`` takes nothing.
    """

    policy: ExportPolicy
    checkpoint_path: Path | None = None
    metric: str | None = None
    rank: int | None = None

    def __post_init__(self) -> None:
        needs_path = self.policy == "explicit"
        needs_metric = self.policy in {"best", "top_k"}
        needs_rank = self.policy == "top_k"
        if self.policy not in {"explicit", "latest", "best", "top_k"}:
            raise ValueError(f"Unsupported export checkpoint policy: {self.policy!r}")
        if (self.checkpoint_path is not None) != needs_path:
            raise ValueError(f"policy={self.policy!r} {_takes(needs_path)} checkpoint_path")
        if (self.metric is not None) != needs_metric:
            raise ValueError(f"policy={self.policy!r} {_takes(needs_metric)} metric")
        if (self.rank is not None) != needs_rank:
            raise ValueError(f"policy={self.policy!r} {_takes(needs_rank)} rank")
        if needs_metric and (not isinstance(self.metric, str) or not self.metric.strip()):
            raise ValueError("metric must be a non-blank string")
        if needs_rank and (type(self.rank) is not int or self.rank <= 0):
            raise ValueError("rank must be an integer greater than 0")


@dataclass(frozen=True)
class ModelBundle:
    """A verified bundle: its root, the parsed index and the bundled resolved config."""

    root: Path
    index: dict[str, Any]
    config: RunConfig


def _takes(required: bool) -> str:
    return "requires" if required else "does not take"


# --- export --------------------------------------------------------------------------


def export_model_bundle(
    run_path: Path,
    output_dir: Path,
    selections: Sequence[ExportCheckpointSelection],
    definitions: Definitions | None = None,
) -> ModelBundle:
    """Validate the selected checkpoints of ``run_path`` and publish a bundle at ``output_dir``.

    ``definitions`` defaults to the built-ins; external methods must be supplied here.
    The bundle is built and verified in a hidden staging directory beside
    ``output_dir`` and renamed into place only when verification passes. The source run
    is never modified and an existing destination is never touched.
    """
    if not selections:
        raise ValueError("At least one checkpoint selection is required")
    run = RunLayout(Path(run_path).resolve(strict=True))
    if not run.root.is_dir():
        raise NotADirectoryError(f"Run path is not a directory: {run.root}")
    train = run.stage("train")
    input_config = _source_file(run.root, train.input_config)
    resolved_config = _source_file(run.root, train.resolved_config)
    environment = _source_file(run.root, train.environment)
    checkpoints_dir = run.checkpoints_dir
    if checkpoints_dir.is_symlink() or not checkpoints_dir.is_dir():
        raise NotADirectoryError(f"Run checkpoints/ is missing or a symlink: {checkpoints_dir}")

    destination = _destination(Path(output_dir), run.root)
    config = RunConfig.from_yaml(resolved_config, definitions)
    config_hash = sha256_file(resolved_config)

    # Validate every selected physical checkpoint before anything is written.
    unique: dict[tuple[int, int], dict[str, Any]] = {}
    sources: dict[tuple[int, int], Path] = {}
    records: list[tuple[dict[str, Any], tuple[int, int]]] = []
    for selection in selections:
        source, record = _select(checkpoints_dir, selection)
        details = os.stat(source)
        key = (details.st_dev, details.st_ino)  # hard-link aliases share one copy
        if key not in unique:
            unique[key] = _validated_metadata(source, config, config_hash)
            unique[key]["path"] = f"{_CHECKPOINTS}/{source.name}"
            sources[key] = source
        entry = unique[key]
        if record["epoch"] is not None and record["epoch"] != entry["epoch"]:
            raise CheckpointCompatibilityError(
                f"best.json records epoch {record['epoch']} for {source}, but the checkpoint "
                f"holds epoch {entry['epoch']}."
            )
        record["epoch"] = entry["epoch"]
        records.append((record, key))

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", suffix=".staging", dir=destination.parent)
    )
    try:
        _copy(input_config, staging / _INPUT_CONFIG)
        _copy(resolved_config, staging / _RESOLVED_CONFIG)
        _copy(environment, staging / _ENVIRONMENT)
        for key, entry in unique.items():
            _copy(sources[key], staging / entry["path"])
        index = {
            "schema_version": BUNDLE_SCHEMA_VERSION,
            "notice": BUNDLE_NOTICE,
            "config": {
                name: {
                    "path": path,
                    "sha256": sha256_file(source),
                    "role": _CONFIG_ROLES[name],
                }
                for name, path, source in (
                    ("input", _INPUT_CONFIG, input_config),
                    ("resolved", _RESOLVED_CONFIG, resolved_config),
                )
            },
            "environment": {"path": _ENVIRONMENT, "sha256": sha256_file(environment)},
            "requirements": _requirements(unique.values()),
            "checkpoints": sorted(unique.values(), key=lambda entry: entry["path"]),
            "selections": [
                {**record, "checkpoint": unique[key]["path"]} for record, key in records
            ],
        }
        (staging / BUNDLE_INDEX).write_text(
            json.dumps(index, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        bundle = verify_model_bundle(staging, config.definitions)
        # ponytail: check-then-rename; a directory created concurrently in this window
        # could be replaced if empty. renameat2(RENAME_NOREPLACE) closes it if it matters.
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"Export destination already exists: {destination}")
        os.rename(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    logger.info("Model bundle exported: %s", destination)
    return replace(bundle, root=destination)


def _source_file(root: Path, path: Path) -> Path:
    """Return a required tracked regular file under ``root``; no component may be a symlink."""
    current = root
    for part in path.relative_to(root).parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"Tracked run artifacts must not be symlinks: {current}")
    if not current.is_file():
        raise FileNotFoundError(f"Required tracked run artifact is missing: {current}")
    return current


def _destination(output_dir: Path, run_root: Path) -> Path:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"Export destination already exists: {output_dir}")
    destination = output_dir.resolve()
    if destination.is_relative_to(run_root):
        raise ValueError(
            f"Export destination {destination} must not be inside the source run {run_root}"
        )
    return destination


def _select(
    checkpoints_dir: Path, selection: ExportCheckpointSelection
) -> tuple[Path, dict[str, Any]]:
    record: dict[str, Any] = {
        "policy": selection.policy,
        "metric": None,
        "rank": None,
        "metric_value": None,
        "mode": None,
        "epoch": None,
    }
    if selection.policy == "explicit":
        assert selection.checkpoint_path is not None
        path = checkpoints_dir / selection.checkpoint_path
    elif selection.policy == "latest":
        path = resolve_checkpoint_path(checkpoints_dir, policy="latest")
    else:
        ranked = load_best_checkpoint_record(
            checkpoints_dir,
            policy=selection.policy,
            metric=selection.metric,
            rank=selection.rank or 1,
        )
        path = ranked.checkpoint_path
        record.update(
            metric=ranked.metric,
            rank=selection.rank or 1,
            metric_value=ranked.metric_value,
            mode=ranked.mode,
            epoch=ranked.epoch,
        )
    return _run_checkpoint(checkpoints_dir, path), record


def _run_checkpoint(checkpoints_dir: Path, path: Path) -> Path:
    """Require ``path`` to be a regular, non-symlink file directly inside ``checkpoints/``."""
    if ".." in path.parts or not path.is_relative_to(checkpoints_dir):
        raise ValueError(f"Checkpoint {path} is outside the run's {checkpoints_dir}")
    if len(path.relative_to(checkpoints_dir).parts) != 1:
        raise ValueError(f"Checkpoint {path} must be a file directly in {checkpoints_dir}")
    if path.is_symlink():
        raise ValueError(f"Checkpoint {path} is a symlink; only regular files are exported")
    if not path.exists():
        raise FileNotFoundError(f"Selected checkpoint does not exist: {path}")
    if not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError(f"Selected checkpoint is not a regular file: {path}")
    return path


def _validated_metadata(path: Path, config: RunConfig, config_hash: str) -> dict[str, Any]:
    """Validate one checkpoint through the current owners and return its index entry."""
    payload = read_checkpoint(path)
    config.definitions.require_checkpoint(payload, path)
    checkpoint = validate_checkpoint(
        payload, config.method.definition.checkpoint_identity(config), path
    )
    if checkpoint.config_hash is None:
        raise CheckpointCompatibilityError(
            f"Checkpoint '{path}' has no config_hash, so it cannot be bound to the tracked "
            "training config."
        )
    if checkpoint.config_hash != config_hash:
        raise CheckpointCompatibilityError(
            f"Checkpoint '{path}' config_hash {checkpoint.config_hash} does not match the "
            f"tracked training resolved config {config_hash}."
        )
    assert isinstance(payload, dict)
    digest, _ = sha256_file_verified(path)
    return {"sha256": digest, **{key: payload[key] for key in _CHECKPOINT_METADATA}}


def _requirements(checkpoints: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, str]]]:
    """Method/component definitions (name, source, version) a reader must supply."""
    methods: set[tuple[str, str, str]] = set()
    components: set[tuple[str, str, str]] = set()
    for entry in checkpoints:
        method = entry["method"]
        implementation = method["implementation"]
        methods.add((method["name"], implementation["source"], implementation["version"]))
        for identity in method["components"].values():
            components.add((identity["name"], identity["source"], identity["version"]))

    def rows(values: set[tuple[str, str, str]]) -> list[dict[str, str]]:
        return [
            dict(zip(("name", "source", "version"), row, strict=True)) for row in sorted(values)
        ]

    return {"methods": rows(methods), "components": rows(components)}


def _copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


# --- verification --------------------------------------------------------------------


def verify_model_bundle(bundle_dir: Path, definitions: Definitions | None = None) -> ModelBundle:
    """Verify a bundle's index, paths, hashes and checkpoint identities.

    ``definitions`` defaults to the built-ins and must include every method/component
    the bundle requires. Checkpoints are read and validated through the current v4
    owners; no model is built.
    """
    root = Path(bundle_dir).resolve(strict=True)
    index = _read_index(root / BUNDLE_INDEX)
    if set(index) != _INDEX_KEYS:
        raise ValueError(
            f"Bundle index keys {sorted(index)} do not match schema {sorted(_INDEX_KEYS)}"
        )
    schema_version = index["schema_version"]
    if type(schema_version) is not int or schema_version != BUNDLE_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported bundle schema_version {schema_version!r}; "
            f"supported: {BUNDLE_SCHEMA_VERSION}"
        )

    configs = _mapping(index["config"], "config")
    if set(configs) != set(_CONFIG_ROLES):
        raise ValueError(f"Bundle config must record exactly {sorted(_CONFIG_ROLES)}")
    for name, role in _CONFIG_ROLES.items():
        if _mapping(configs[name], f"config.{name}").get("role") != role:
            raise ValueError(f"Bundle config.{name} must have role {role!r}")
    resolved = _hashed_file(root, configs["resolved"], "config.resolved")
    _hashed_file(root, configs["input"], "config.input")
    _hashed_file(root, _mapping(index["environment"], "environment"), "environment")
    config = RunConfig.from_yaml(resolved, definitions)
    config_hash = sha256_file(resolved)

    checkpoints = index["checkpoints"]
    if not isinstance(checkpoints, list) or not checkpoints:
        raise ValueError("Bundle index must list at least one checkpoint")
    epochs: dict[str, int] = {}
    for position, raw in enumerate(checkpoints):
        entry = _mapping(raw, f"checkpoints[{position}]")
        if set(entry) != {"path", "sha256", *_CHECKPOINT_METADATA}:
            raise ValueError(f"Bundle checkpoints[{position}] has unexpected keys")
        relative = entry["path"]
        if not isinstance(relative, str) or not relative.startswith(f"{_CHECKPOINTS}/"):
            raise ValueError(f"Bundle checkpoints[{position}] must be under {_CHECKPOINTS}/")
        if relative in epochs:
            raise ValueError(f"Bundle lists checkpoint {relative} more than once")
        path = _hashed_file(root, entry, f"checkpoints[{position}]")
        metadata = _validated_metadata(path, config, config_hash)
        if {**metadata, "path": relative} != entry:
            raise ValueError(f"Bundle metadata for {relative} does not match the checkpoint")
        epochs[relative] = metadata["epoch"]

    if index["requirements"] != _requirements(checkpoints):
        raise ValueError("Bundle requirements do not match the checkpoints' metadata")

    selections = index["selections"]
    if not isinstance(selections, list) or not selections:
        raise ValueError("Bundle index must list at least one selection")
    for position, raw in enumerate(selections):
        record = _mapping(raw, f"selections[{position}]")
        if set(record) != _SELECTION_KEYS:
            raise ValueError(f"Bundle selections[{position}] has unexpected keys")
        if record["checkpoint"] not in epochs:
            raise ValueError(f"Bundle selections[{position}] names an unlisted checkpoint")
        if record["epoch"] != epochs[record["checkpoint"]]:
            raise ValueError(f"Bundle selections[{position}] epoch does not match its checkpoint")
        if record["policy"] not in {"explicit", "latest", "best", "top_k"}:
            raise ValueError(f"Bundle selections[{position}] has unknown policy")
        ranked = record["policy"] in {"best", "top_k"}
        for key in ("metric", "rank", "metric_value", "mode"):
            if (record[key] is None) == ranked:
                raise ValueError(
                    f"Bundle selections[{position}].{key} must be "
                    f"{'set' if ranked else 'null'} for policy {record['policy']!r}"
                )
    return ModelBundle(root=root, index=index, config=config)


def _read_index(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"Bundle index is missing or a symlink: {path}")

    def reject_constant(value: str) -> None:
        raise ValueError(f"Bundle index {path} is not strict JSON: {value}")

    try:
        index = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Bundle index {path} is not valid JSON") from exc
    return _mapping(index, "bundle index")


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Bundle {name} must be a JSON object")
    return value


def _hashed_file(root: Path, entry: dict[str, Any], name: str) -> Path:
    """Resolve ``entry['path']`` strictly under ``root`` and check its recorded SHA-256."""
    relative = entry.get("path")
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"Bundle {name}.path must be a non-empty relative path")
    parts = PurePosixPath(relative).parts
    if PurePosixPath(relative).is_absolute() or "\\" in relative or ".." in parts:
        raise ValueError(f"Bundle {name}.path {relative!r} escapes the bundle root")
    path = root
    for part in parts:
        path = path / part
        if path.is_symlink():
            raise ValueError(f"Bundle {name}.path {relative!r} traverses a symlink")
    if not path.resolve().is_relative_to(root) or not path.is_file():
        raise FileNotFoundError(f"Bundle {name}.path {relative!r} is not a file in the bundle")
    if sha256_file(path) != entry.get("sha256"):
        raise ValueError(f"Bundle {name}.path {relative!r} does not match its recorded SHA-256")
    return path
