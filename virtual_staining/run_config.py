from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

# All keys allowed at the top level of a run YAML (shared fields + section names).
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
    }
)


def load_yaml_mapping(path: str | Path) -> dict[str, Any]:
    import yaml

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {path}")

    return data


def section_with_shared_fields(
    data: Mapping[str, Any], section_name: str, shared_fields: set[str]
) -> dict[str, Any]:
    section = data.get(section_name)

    if section is None:
        return dict(data)

    if not isinstance(section, dict):
        raise ValueError(f"Config section '{section_name}' must be a mapping.")

    merged = {field: data[field] for field in shared_fields if field in data}
    merged.update(section)
    return merged


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
