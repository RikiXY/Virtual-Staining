from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from virtual_staining.config.validation import (
    parse_bool_strict,
    parse_choice,
    reject_unknown_keys,
)

if TYPE_CHECKING:
    from virtual_staining.metrics import MetricDefinition, ResolvedMetric

EvaluationProtocol = Literal["paired", "unpaired"]
InputFailureMode = Literal["strict", "permissive"]

_EVALUATION_KEYS: frozenset[str] = frozenset(
    {
        "save_graphs",
        "generated_dir",
        "output_dir",
        "bootstrap_iterations",
        "bootstrap_seed",
        "protocol",
        "metrics",
        "input_failures",
        "reference_collection",
    }
)


@dataclass(frozen=True)
class EvaluationConfig:
    save_graphs: bool = False
    generated_dir: Path | None = None
    output_dir: Path | None = None
    bootstrap_iterations: int = 10_000
    bootstrap_seed: int = 0
    # None resolves to the method's training pairing: pix2pix -> paired, cyclegan -> unpaired.
    protocol: EvaluationProtocol | None = None
    # None requests the built-in default metric set of the paired protocol.
    metrics: tuple[ResolvedMetric, ...] | None = None
    input_failures: InputFailureMode = "strict"
    # Unpaired only: independent real reference collection, same spec as a data.domains entry.
    reference_collection: str | None = None

    def __post_init__(self) -> None:
        if self.bootstrap_iterations < 0:
            raise ValueError("evaluation.bootstrap_iterations must be >= 0")

    @classmethod
    def from_mapping(
        cls, data: dict[str, Any], metric_definitions: Mapping[str, MetricDefinition]
    ) -> EvaluationConfig:
        from virtual_staining.metrics import resolve_metrics

        reject_unknown_keys(data, _EVALUATION_KEYS, "evaluation")
        reference_collection = data.get("reference_collection")
        if reference_collection is not None and (
            not isinstance(reference_collection, str) or not reference_collection.strip()
        ):
            raise TypeError(
                "evaluation.reference_collection must be a non-empty path or pattern string"
            )
        return cls(
            save_graphs=parse_bool_strict(data.get("save_graphs", False), "evaluation.save_graphs"),
            generated_dir=Path(data["generated_dir"]) if data.get("generated_dir") else None,
            output_dir=Path(data["output_dir"]) if data.get("output_dir") else None,
            bootstrap_iterations=int(data.get("bootstrap_iterations", 10_000)),
            bootstrap_seed=int(data.get("bootstrap_seed", 0)),
            protocol=(
                cast(
                    EvaluationProtocol,
                    parse_choice(data["protocol"], "evaluation.protocol", {"paired", "unpaired"}),
                )
                if data.get("protocol") is not None
                else None
            ),
            metrics=(
                resolve_metrics(data["metrics"], metric_definitions)
                if data.get("metrics") is not None
                else None
            ),
            input_failures=cast(
                InputFailureMode,
                parse_choice(
                    data.get("input_failures", "strict"),
                    "evaluation.input_failures",
                    {"strict", "permissive"},
                ),
            ),
            reference_collection=reference_collection,
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
                "metrics": (
                    [metric.request() for metric in self.metrics]
                    if self.metrics is not None
                    else None
                ),
                "input_failures": self.input_failures,
                "reference_collection": self.reference_collection,
            }.items()
            if value is not None
        }
