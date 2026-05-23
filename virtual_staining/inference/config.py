from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from virtual_staining.training.config import SUPPORTED_CHECKPOINT_METRICS

SUPPORTED_CHECKPOINT_POLICIES: frozenset[str] = frozenset({"latest", "best", "top_k"})

_INFERENCE_KEYS: frozenset[str] = frozenset(
    {
        "checkpoint_path",
        "checkpoint_policy",
        "checkpoint_metric",
        "checkpoint_rank",
        "output_dir",
    }
)


@dataclass(frozen=True)
class InferenceConfig:
    checkpoint_policy: str | None = None
    checkpoint_path: Path | None = None
    checkpoint_metric: str | None = None
    checkpoint_rank: int | None = None
    output_dir: Path | None = None

    def validate(self) -> None:
        if self.checkpoint_policy is None and self.checkpoint_path is None:
            raise ValueError(
                "Either inference.checkpoint_path or inference.checkpoint_policy must be set."
            )
        if (
            self.checkpoint_policy is not None
            and self.checkpoint_policy not in SUPPORTED_CHECKPOINT_POLICIES
        ):
            raise ValueError(
                f"Unknown checkpoint_policy: {self.checkpoint_policy!r}. "
                f"Supported values: {sorted(SUPPORTED_CHECKPOINT_POLICIES)}."
            )
        if self.checkpoint_path is not None and not str(self.checkpoint_path).strip():
            raise ValueError("checkpoint_path must be a non-empty path")
        if (
            self.checkpoint_metric is not None
            and self.checkpoint_metric not in SUPPORTED_CHECKPOINT_METRICS
        ):
            raise ValueError(
                f"Unknown checkpoint_metric: {self.checkpoint_metric!r}. "
                f"Supported values: {sorted(SUPPORTED_CHECKPOINT_METRICS)}."
            )
        if self.checkpoint_rank is not None and self.checkpoint_rank <= 0:
            raise ValueError("checkpoint_rank must be greater than 0")
        if self.checkpoint_rank is not None and self.checkpoint_policy not in {"best", "top_k"}:
            raise ValueError(
                "checkpoint_rank is supported only with checkpoint_policy 'best' or 'top_k'"
            )
        if self.checkpoint_policy in {"best", "top_k"} and self.checkpoint_metric is None:
            raise ValueError(
                "checkpoint_metric is required with checkpoint_policy 'best' or 'top_k'"
            )
