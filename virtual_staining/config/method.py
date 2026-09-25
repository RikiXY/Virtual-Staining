from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast

from virtual_staining.config.validation import parse_choice, reject_unknown_keys

MethodName = Literal["pix2pix", "cyclegan"]
_METHOD_KEYS = frozenset({"name"})
_BUILTIN_METHODS = {"pix2pix", "cyclegan"}


@dataclass(frozen=True)
class MethodConfig:
    """Select the built-in image-translation method for a run."""

    name: MethodName = "pix2pix"

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> MethodConfig:
        reject_unknown_keys(data, _METHOD_KEYS, "method")
        return cls(
            name=cast(
                MethodName,
                parse_choice(
                    data.get("name", "pix2pix"),
                    "method.name",
                    _BUILTIN_METHODS,
                ),
            )
        )

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name}
