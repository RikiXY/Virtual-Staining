from __future__ import annotations

from collections.abc import Mapping
from typing import Any


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
