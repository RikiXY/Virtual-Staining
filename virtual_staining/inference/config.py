from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

SUPPORTED_CHECKPOINT_POLICIES: frozenset[str] = frozenset({"latest", "best_val_loss"})

_INFERENCE_KEYS: frozenset[str] = frozenset(
    {
        "checkpoint_path",
        "checkpoint_policy",
        "output_dir",
    }
)


@dataclass(frozen=True)
class InferenceConfig:
    checkpoint_policy: str | None = None
    checkpoint_path: Path | None = None
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
