from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def reject_unknown_keys(data: Mapping[str, Any], allowed: frozenset[str], context: str) -> None:
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"Unknown key(s) in {context}: {', '.join(sorted(unknown))}")


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
