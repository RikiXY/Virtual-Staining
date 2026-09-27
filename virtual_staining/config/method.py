from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from virtual_staining.definitions import MethodDefinition

METHOD_KEYS = frozenset({"name"})
# The built-in default kept for the existing public YAML spelling; resolved like any name.
DEFAULT_METHOD_NAME = "pix2pix"


@dataclass(frozen=True)
class MethodConfig:
    """The selected registered method definition and the options it resolved.

    ``definition`` is a live Python reference and is never serialized; the resolved
    config records the stable ``name`` plus the definition's own option spelling.
    """

    definition: MethodDefinition
    options: Any

    @property
    def name(self) -> str:
        return self.definition.name

    def sections(self) -> dict[str, dict[str, Any]]:
        """Method-owned resolved keys per config section."""
        return self.definition.options_to_sections(self.options)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, **self.sections().get("method", {})}
