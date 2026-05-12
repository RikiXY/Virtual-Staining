from __future__ import annotations

from collections.abc import Mapping
from typing import Any

_TOP_LEVEL_KEYS: frozenset[str] = frozenset(
    {
        "dataset_root",
        "results_path",
        "run_name",
        "image_size",
        "preprocessing",
        "training",
        "inference",
        "evaluation",
        "compare",
        "compare_panels",
        "organize",
        "model",
    }
)


def reject_unknown_keys(data: Mapping[str, Any], allowed: frozenset[str], context: str) -> None:
    """Raise ValueError listing any keys in *data* that are not in *allowed*."""
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"Unknown key(s) in {context}: {', '.join(sorted(unknown))}")


def parse_bool_strict(value: object, field_name: str) -> bool:
    """Return *value* as bool, raising TypeError if it is not already a Python bool.

    Prevents YAML strings like ``"false"`` from silently becoming ``True``.
    """
    if isinstance(value, bool):
        return value
    raise TypeError(
        f"'{field_name}' must be a YAML boolean (true or false), "
        f"got {value!r}. Use true or false without quotes."
    )
