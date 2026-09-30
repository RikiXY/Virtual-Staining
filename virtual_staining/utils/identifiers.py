"""The one persisted modality/output identifier rule.

Names that become manifest columns, config keys, checkpoint identity, or generated
artifact directories are machine identifiers; they are validated, never sanitized.
"""

from __future__ import annotations

import re

#: Persisted modality/domain/output identifiers (config, inventory, manifest, artifacts).
MODALITY_NAME_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_-]*\Z")


def is_modality_name(value: object) -> bool:
    """Whether ``value`` is a safe persisted identifier (never ``.``, ``..`` or a path)."""
    return isinstance(value, str) and MODALITY_NAME_PATTERN.match(value) is not None
