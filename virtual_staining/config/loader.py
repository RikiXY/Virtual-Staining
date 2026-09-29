from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any


def load_yaml_mapping(path: str | Path) -> dict[str, Any]:
    import yaml

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {path}")

    return data


def dump_yaml_mapping(mapping: Mapping[str, Any], *, sort_keys: bool) -> str:
    """Render a plain mapping as YAML; resolved configs use ``sort_keys=True``.

    Tracked stage snapshots and config authoring share this function, so the resolved
    bytes (and therefore the resolved-config SHA-256) are identical for one ``RunConfig``.
    """
    import yaml

    return yaml.safe_dump(
        dict(mapping), default_flow_style=False, allow_unicode=True, sort_keys=sort_keys
    )
