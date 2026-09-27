from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import torch

from virtual_staining.checkpoint_contract import CheckpointIdentity
from virtual_staining.training.preview import ValidationPreviewSink


@dataclass(frozen=True)
class MethodMetrics:
    """Named method-owned metrics returned to generic training orchestration.

    ``losses`` holds the method's objective scalars (``metric_names``), written as
    ``<name>_train``/``<name>_val``. ``component_totals`` and the ``raw``/``weighted``/
    ``current_weight`` maps hold optional per-term diagnostics keyed by ``loss_names``.
    ``image`` holds further validation scalars keyed by ``validation_metric_names``.
    """

    losses: dict[str, float]
    component_totals: dict[str, float] = field(default_factory=dict)
    raw: dict[str, float] = field(default_factory=dict)
    weighted: dict[str, float] = field(default_factory=dict)
    current_weight: dict[str, float] = field(default_factory=dict)
    image: dict[str, float] = field(default_factory=dict)


class TrainingMethodRuntime(Protocol):
    """Behavior owned by one concrete translation method during training.

    The method owns its topology, objectives, optimizers and state; ``Trainer`` owns the
    epoch, validation, checkpoint and history lifecycle and never inspects either.
    """

    @property
    def name(self) -> str: ...
    @property
    def default_checkpoint_metric(self) -> str: ...
    @property
    def metric_names(self) -> Sequence[str]: ...
    @property
    def component_total_names(self) -> Sequence[str]: ...
    @property
    def loss_names(self) -> Sequence[str]: ...
    @property
    def validation_metric_names(self) -> Sequence[str]: ...

    def train_mode(self) -> None: ...
    def batch_size(self, batch: object) -> int: ...
    def step(self, batch: object, *, epoch: int, global_step: int) -> MethodMetrics: ...
    def validate(
        self,
        loader: torch.utils.data.DataLoader,
        *,
        epoch: int,
        preview_sink: ValidationPreviewSink | None = None,
    ) -> MethodMetrics: ...
    def validation_metric(self, metrics: MethodMetrics, name: str) -> float | None: ...
    def checkpoint_selection_metrics(self, metrics: MethodMetrics) -> dict[str, float]: ...
    def checkpoint_selection_modes(self) -> dict[str, str]: ...
    def step_schedulers(
        self,
        *,
        epoch: int,
        validation_metrics: MethodMetrics | None,
    ) -> bool: ...
    def learning_rates(self) -> Mapping[str, float]: ...
    def checkpoint_identity(self) -> CheckpointIdentity: ...
    def objective_metadata(self) -> dict[str, Any] | None:
        """Optional JSON-compatible objective provenance recorded in ``best.json``."""
        ...

    def state_dict(self) -> dict[str, Any]: ...
    def load_state_dict(self, state: Mapping[str, Any]) -> None: ...
