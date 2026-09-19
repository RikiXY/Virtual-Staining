from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from virtual_staining.config.validation import parse_bool_strict, parse_choice, reject_unknown_keys

EvaluationProtocol = Literal["auto", "paired", "unpaired"]

_EVALUATION_KEYS: frozenset[str] = frozenset(
    {
        "save_graphs",
        "generated_dir",
        "output_dir",
        "bootstrap_iterations",
        "bootstrap_seed",
        "protocol",
        "real_target",
    }
)


@dataclass(frozen=True)
class EvaluationConfig:
    save_graphs: bool = False
    generated_dir: Path | None = None
    output_dir: Path | None = None
    bootstrap_iterations: int = 10_000
    bootstrap_seed: int = 0
    protocol: EvaluationProtocol = "auto"
    real_target: Path | None = None

    def __post_init__(self) -> None:
        if self.bootstrap_iterations < 0:
            raise ValueError("evaluation.bootstrap_iterations must be >= 0")

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> EvaluationConfig:
        reject_unknown_keys(data, _EVALUATION_KEYS, "evaluation")
        return cls(
            save_graphs=parse_bool_strict(data.get("save_graphs", False), "evaluation.save_graphs"),
            generated_dir=Path(data["generated_dir"]) if data.get("generated_dir") else None,
            output_dir=Path(data["output_dir"]) if data.get("output_dir") else None,
            bootstrap_iterations=int(data.get("bootstrap_iterations", 10_000)),
            bootstrap_seed=int(data.get("bootstrap_seed", 0)),
            protocol=cast(
                EvaluationProtocol,
                parse_choice(
                    data.get("protocol", "auto"),
                    "evaluation.protocol",
                    {"auto", "paired", "unpaired"},
                ),
            ),
            real_target=Path(data["real_target"]) if data.get("real_target") else None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in {
                "save_graphs": self.save_graphs,
                "generated_dir": str(self.generated_dir) if self.generated_dir else None,
                "output_dir": str(self.output_dir) if self.output_dir else None,
                "bootstrap_iterations": self.bootstrap_iterations,
                "bootstrap_seed": self.bootstrap_seed,
                "protocol": self.protocol,
                "real_target": str(self.real_target) if self.real_target else None,
            }.items()
            if value is not None
        }
