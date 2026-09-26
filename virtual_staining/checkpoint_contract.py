from __future__ import annotations

import pickle
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from virtual_staining.models.io_contract import NORMALIZATION_CONTRACT

CHECKPOINT_FORMAT_VERSION: int = 4


class CheckpointCompatibilityError(ValueError):
    """Raised when a checkpoint is malformed or semantically incompatible."""


def _canonical(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _canonical(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    return value


_MISSING = "<missing>"
_IDENTITY_KEYS = frozenset({"name", "class"})


def first_difference(stored: Any, current: Any, path: str) -> tuple[str, Any, Any] | None:
    if isinstance(stored, dict) and isinstance(current, dict):
        # Identity keys first, so a type change is reported instead of its consequences.
        for key in sorted(set(stored) | set(current), key=lambda k: (k not in _IDENTITY_KEYS, k)):
            difference = first_difference(
                stored.get(key, _MISSING), current.get(key, _MISSING), f"{path}.{key}"
            )
            if difference is not None:
                return difference
        return None
    if isinstance(stored, list) and isinstance(current, list) and len(stored) == len(current):
        for index, (stored_item, current_item) in enumerate(zip(stored, current, strict=True)):
            difference = first_difference(stored_item, current_item, f"{path}[{index}]")
            if difference is not None:
                return difference
        return None
    if type(stored) is not type(current) or stored != current:
        return path, stored, current
    return None


@dataclass(frozen=True)
class CheckpointIdentity:
    """Semantic identity a checkpoint must match before method state is loaded."""

    method: str
    pairing: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    prediction_directions: tuple[str, ...]
    components: Mapping[str, object]
    image_size: tuple[int, int]
    normalization: Mapping[str, object] = field(default_factory=lambda: NORMALIZATION_CONTRACT)

    def method_metadata(self) -> dict[str, Any]:
        return _canonical(
            {
                "name": self.method,
                "pairing": self.pairing,
                "inputs": self.inputs,
                "outputs": self.outputs,
                "prediction_directions": self.prediction_directions,
                "components": self.components,
            }
        )


@dataclass(frozen=True)
class ValidatedCheckpoint:
    epoch: int
    state: Mapping[str, Any]
    config_hash: str | None


def build_checkpoint_payload(
    identity: CheckpointIdentity,
    *,
    epoch: int,
    state: Mapping[str, Any],
    config_hash: str | None = None,
) -> dict[str, Any]:
    return {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "epoch": epoch,
        "method": identity.method_metadata(),
        "image_size": list(identity.image_size),
        "normalization": _canonical(identity.normalization),
        "config_hash": config_hash or None,
        "state": dict(state),
    }


def read_checkpoint(path: Path) -> object:
    """Deserialize ``path`` onto the CPU with PyTorch's restricted weights-only unpickler.

    Only tensors and primitive containers are accepted; no fallback to unrestricted pickle
    exists. Callers move state to the execution device through normal model/optimizer
    restoration. This narrows arbitrary-code deserialization exposure; it is not a resource
    sandbox.
    """
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except (pickle.UnpicklingError, EOFError, RuntimeError) as exc:
        raise CheckpointCompatibilityError(
            f"Checkpoint '{path}' cannot be read as a supported weights-only checkpoint: {exc}"
        ) from exc


def _require_mapping(payload: Mapping[str, Any], key: str, path: Path) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise CheckpointCompatibilityError(
            f"Checkpoint '{path}' has missing or malformed '{key}' metadata."
        )
    return _canonical(value)


def _check_equal(name: str, stored: Any, current: Any, path: Path) -> None:
    difference = first_difference(stored, current, name)
    if difference is not None:
        field_name, stored_value, current_value = difference
        raise CheckpointCompatibilityError(
            f"Checkpoint '{path}' is incompatible: {field_name} is {stored_value!r} in the "
            f"checkpoint but {current_value!r} in the current run."
        )


def validate_checkpoint(
    payload: object,
    expected: CheckpointIdentity,
    path: Path,
) -> ValidatedCheckpoint:
    """Validate a v4 payload against ``expected`` without touching method state."""
    if not isinstance(payload, Mapping):
        raise CheckpointCompatibilityError(
            f"Checkpoint '{path}' is not a mapping; only format version "
            f"{CHECKPOINT_FORMAT_VERSION} checkpoints are supported."
        )
    if "format_version" not in payload:
        raise CheckpointCompatibilityError(
            f"Checkpoint '{path}' is unversioned and unsupported; only format version "
            f"{CHECKPOINT_FORMAT_VERSION} checkpoints are supported. Retrain with current code."
        )
    version = payload["format_version"]
    if type(version) is not int or version != CHECKPOINT_FORMAT_VERSION:
        raise CheckpointCompatibilityError(
            f"Checkpoint '{path}' has unsupported format version {version!r}; only format "
            f"version {CHECKPOINT_FORMAT_VERSION} is supported. Retrain with current code."
        )
    epoch = payload.get("epoch")
    if type(epoch) is not int or epoch < 0:
        raise CheckpointCompatibilityError(f"Checkpoint '{path}' has malformed epoch {epoch!r}.")

    stored_method = _require_mapping(payload, "method", path)
    current_method = expected.method_metadata()
    for key in ("name", "pairing", "inputs", "outputs", "prediction_directions", "components"):
        if key not in stored_method:
            raise CheckpointCompatibilityError(
                f"Checkpoint '{path}' is missing method.{key} metadata."
            )
        _check_equal(f"method.{key}", stored_method[key], current_method[key], path)
    if set(stored_method) != set(current_method):
        raise CheckpointCompatibilityError(
            f"Checkpoint '{path}' has unexpected method metadata keys: "
            f"{sorted(set(stored_method) - set(current_method))}."
        )

    stored_size = payload.get("image_size")
    if not isinstance(stored_size, Sequence) or isinstance(stored_size, str):
        raise CheckpointCompatibilityError(f"Checkpoint '{path}' has malformed image_size.")
    _check_equal("image_size", _canonical(stored_size), list(expected.image_size), path)

    stored_normalization = _require_mapping(payload, "normalization", path)
    _check_equal("normalization", stored_normalization, _canonical(expected.normalization), path)

    config_hash = payload.get("config_hash")
    if config_hash is not None and not isinstance(config_hash, str):
        raise CheckpointCompatibilityError(f"Checkpoint '{path}' has malformed config_hash.")

    if "state" not in payload:
        raise CheckpointCompatibilityError(f"Checkpoint '{path}' has no method state.")
    state = payload["state"]
    if not isinstance(state, Mapping) or not state or not all(isinstance(k, str) for k in state):
        raise CheckpointCompatibilityError(
            f"Checkpoint '{path}' has malformed method state; expected a non-empty mapping "
            "with string keys."
        )
    return ValidatedCheckpoint(epoch=epoch, state=state, config_hash=config_hash)
