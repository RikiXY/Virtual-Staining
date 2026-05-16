from __future__ import annotations

from pathlib import Path
from typing import Any


def load_yaml_mapping(path: str | Path) -> dict[str, Any]:
    import yaml

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {path}")

    return data
