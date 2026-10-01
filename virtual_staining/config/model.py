from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from virtual_staining.config.validation import (
    check_modality_names,
    parse_modality_names,
    reject_superseded_keys,
    reject_unknown_keys,
)

MODEL_KEYS = frozenset({"inputs", "outputs"})
SUPERSEDED_MODEL_KEYS = {"target": "model.outputs (an ordered list of output names)"}


@dataclass(frozen=True)
class ModelConfig:
    """Framework-visible translation I/O: N ordered named RGB inputs -> M named RGB outputs.

    One output is the one-item case of the same plural representation. Network topology
    is method-owned; a method definition may own further ``model.*`` keys (the built-ins
    own ``model.generator`` and ``model.discriminator``).
    """

    inputs: tuple[str, ...]
    outputs: tuple[str, ...]

    def __post_init__(self) -> None:
        check_modality_names(self.inputs, "model.inputs")
        check_modality_names(self.outputs, "model.outputs")
        shared = sorted(set(self.inputs) & set(self.outputs))
        if shared:
            raise ValueError(f"model.inputs and model.outputs must be disjoint; both name {shared}")

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> ModelConfig:
        reject_superseded_keys(data, SUPERSEDED_MODEL_KEYS, "model")
        reject_unknown_keys(data, MODEL_KEYS, "model")
        for required in ("inputs", "outputs"):
            if required not in data:
                raise ValueError(f"model requires {required}")
        return cls(
            inputs=parse_modality_names(data["inputs"], "model.inputs"),
            outputs=parse_modality_names(data["outputs"], "model.outputs"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"inputs": list(self.inputs), "outputs": list(self.outputs)}
