from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any

#: Persisted modality/domain identifiers (config, inventory, manifest, artifacts).
MODALITY_NAME_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_-]*\Z")


def reject_unknown_keys(data: Mapping[str, Any], allowed: frozenset[str], context: str) -> None:
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"Unknown key(s) in {context}: {', '.join(sorted(unknown))}")


def reject_superseded_keys(
    data: Mapping[str, Any], replacements: Mapping[str, str], context: str
) -> None:
    """Reject pre-cutover keys with a pointer to their current-schema replacement."""
    for key, replacement in replacements.items():
        if key in data:
            raise ValueError(
                f"{context}.{key} is not part of the current schema; use {replacement}"
            )


def check_modality_names(names: tuple[str, ...], field_name: str) -> tuple[str, ...]:
    """Require a non-empty tuple of unique safe identifiers; nothing is sanitized."""
    if not names:
        raise ValueError(f"{field_name} must contain at least one name")
    invalid = [
        name for name in names if not isinstance(name, str) or not MODALITY_NAME_PATTERN.match(name)
    ]
    if invalid:
        raise ValueError(
            f"{field_name} contains invalid identifiers {invalid}; names must match "
            "[A-Za-z][A-Za-z0-9_-]*"
        )
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"{field_name} contains duplicate names {duplicates}")
    return names


def parse_modality_names(value: object, field_name: str) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, list | tuple):
        raise TypeError(f"{field_name} must be a sequence of names")
    return check_modality_names(tuple(value), field_name)


def parse_bool_strict(value: object, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    raise TypeError(
        f"'{field_name}' must be a YAML boolean (true or false), "
        f"got {value!r}. Use true or false without quotes."
    )


def parse_choice(value: object, field_name: str, choices: set[str]) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string. Supported values: {sorted(choices)}.")
    if value not in choices:
        raise ValueError(f"{field_name} must be one of {sorted(choices)}. Got {value!r}.")
    return value


def require_finite(value: float, field_name: str) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{field_name} must be a finite number, got {value!r}")


def parse_int(value: object, field_name: str) -> int:
    """Parse with ``int()`` but reject non-finite floats with a ``ValueError``."""
    if isinstance(value, float):
        require_finite(value, field_name)
    return int(value)  # type: ignore[call-overload]
