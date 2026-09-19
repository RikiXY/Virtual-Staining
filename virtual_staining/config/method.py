from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from virtual_staining.config.validation import reject_unknown_keys

_METHOD_KEYS = frozenset({"name", "class_path", "params"})


@dataclass(frozen=True)
class MethodConfig:
    """Selects the translation algorithm, independently from its model components."""

    name: str = "pix2pix"
    class_path: str | None = None
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.name not in {"pix2pix", "cyclegan", "custom"}:
            raise ValueError("method.name must be one of ['custom', 'cyclegan', 'pix2pix']")
        if self.name == "custom" and not self.class_path:
            raise ValueError("method.class_path is required when method.name is 'custom'")
        if self.name != "custom" and self.class_path is not None:
            raise ValueError("method.class_path is only valid when method.name is 'custom'")

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> MethodConfig:
        reject_unknown_keys(data, _METHOD_KEYS, "method")
        params = data.get("params", {})
        if not isinstance(params, dict):
            raise TypeError("method.params must be a YAML mapping")
        class_path = data.get("class_path")
        if class_path is not None and not isinstance(class_path, str):
            raise TypeError("method.class_path must be a string")
        return cls(name=str(data.get("name", "pix2pix")), class_path=class_path, params=params)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"name": self.name}
        if self.class_path is not None:
            result["class_path"] = self.class_path
        if self.params:
            result["params"] = dict(self.params)
        return result
