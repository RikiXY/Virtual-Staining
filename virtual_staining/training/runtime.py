from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import torch
import torch.optim as optim

from virtual_staining.training.losses import StepLosses
from virtual_staining.training.results import EpochMetrics


class TrainingMethodRuntime(Protocol):
    name: str
    pairing: str
    loss_names: Sequence[str]
    optimizers: Sequence[optim.Optimizer]

    def train_mode(self) -> None: ...

    def step(self, batch: Any, *, epoch: int, global_step: int) -> StepLosses: ...

    def validate(
        self,
        loader: torch.utils.data.DataLoader,
        *,
        epoch: int,
        output_dir: Path,
    ) -> EpochMetrics: ...

    def component_metadata(self) -> Mapping[str, object]: ...

    def state_dict(self) -> dict[str, Any]: ...

    def load_state_dict(self, state: Mapping[str, Any]) -> None: ...

    def load_legacy_v3(self, checkpoint: Mapping[str, Any]) -> None: ...
