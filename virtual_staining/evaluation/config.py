from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from virtual_staining.config.validation import parse_bool_strict, reject_unknown_keys

_EVALUATION_KEYS: frozenset[str] = frozenset(
    {
        "save_graphs",
        "target_dir",
        "generated_dir",
        "output_dir",
        "bootstrap_iterations",
        "bootstrap_seed",
    }
)


@dataclass(frozen=True)
class EvaluationConfig:
    save_graphs: bool = False
    target_dir: Path | None = None
    generated_dir: Path | None = None
    output_dir: Path | None = None
    bootstrap_iterations: int = 10_000
    bootstrap_seed: int = 0

    def __post_init__(self) -> None:
        if self.bootstrap_iterations < 0:
            raise ValueError("evaluation.bootstrap_iterations must be >= 0")

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> EvaluationConfig:
        reject_unknown_keys(data, _EVALUATION_KEYS, "evaluation")
        return cls(
            save_graphs=parse_bool_strict(data.get("save_graphs", False), "evaluation.save_graphs"),
            target_dir=Path(data["target_dir"]) if data.get("target_dir") else None,
            generated_dir=Path(data["generated_dir"]) if data.get("generated_dir") else None,
            output_dir=Path(data["output_dir"]) if data.get("output_dir") else None,
            bootstrap_iterations=int(data.get("bootstrap_iterations", 10_000)),
            bootstrap_seed=int(data.get("bootstrap_seed", 0)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in {
                "save_graphs": self.save_graphs,
                "target_dir": str(self.target_dir) if self.target_dir else None,
                "generated_dir": str(self.generated_dir) if self.generated_dir else None,
                "output_dir": str(self.output_dir) if self.output_dir else None,
                "bootstrap_iterations": self.bootstrap_iterations,
                "bootstrap_seed": self.bootstrap_seed,
            }.items()
            if value is not None
        }
