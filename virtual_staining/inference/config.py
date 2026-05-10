from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

_INFERENCE_KEYS: frozenset[str] = frozenset(
    {
        # shared fields and size aliases (accepted after section_with_shared_fields injects them)
        "dataset_root",
        "results_path",
        "run_name",
        "image_size",
        "model_image_size",
        # section-specific
        "checkpoint",
        "checkpoint_policy",
        "test_dir",
        "output_dir",
    }
)


@dataclass(frozen=True)
class InferenceConfig:
    checkpoint_policy: str | None = None
    checkpoint_path: Path | None = None
    test_dir: Path | None = None
    output_dir: Path | None = None

    def validate(self) -> None:
        if self.checkpoint_policy is None and self.checkpoint_path is None:
            raise ValueError(
                "Either inference.checkpoint or inference.checkpoint_policy must be set."
            )
        if self.checkpoint_policy is not None and self.checkpoint_policy != "latest":
            raise ValueError(
                f"Unknown checkpoint_policy: {self.checkpoint_policy!r}. "
                "Only 'latest' is supported."
            )
        if self.checkpoint_path is not None and not str(self.checkpoint_path).strip():
            raise ValueError("checkpoint must be a non-empty path")
