from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from virtual_staining.config.validation import reject_unknown_keys

MODEL_KEYS = frozenset({"inputs", "target"})


@dataclass(frozen=True)
class ModelConfig:
    """Framework-visible translation I/O: ordered named RGB inputs and one RGB output.

    Network topology is method-owned; a method definition may own further ``model.*``
    keys (the built-ins own ``model.generator`` and ``model.discriminator``).
    """

    inputs: tuple[str, ...]
    target: str

    def __post_init__(self) -> None:
        if (
            not self.inputs
            or len(set(self.inputs)) != len(self.inputs)
            or any(not name.strip() for name in self.inputs)
        ):
            raise ValueError("model.inputs must be a non-empty tuple of unique names")
        if not self.target.strip():
            raise ValueError("model.target must not be blank")

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> ModelConfig:
        reject_unknown_keys(data, MODEL_KEYS, "model")
        for required in ("inputs", "target"):
            if required not in data:
                raise ValueError(f"model requires {required}")
        raw_inputs = data["inputs"]
        if isinstance(raw_inputs, str) or not isinstance(raw_inputs, (list, tuple)):
            raise TypeError("model.inputs must be a sequence")
        return cls(inputs=tuple(str(value) for value in raw_inputs), target=str(data["target"]))

    def to_dict(self) -> dict[str, Any]:
        return {"inputs": list(self.inputs), "target": self.target}
