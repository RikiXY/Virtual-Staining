"""A tiny external paired reconstruction method written against the public extension API.

It stands in for third-party code: it imports only documented public MEXINA modules and is
made available solely by a caller passing its definitions explicitly. One network, one
optimizer and one L1 objective; no discriminator and no adversarial term. Two component
architectures are registered through ``ComponentDefinition``. ``TinyReconstruction.built``
counts constructed objects so callers can prove inference builds only the network.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import torch
from torch import nn

from virtual_staining.checkpoint_contract import (
    CheckpointCompatibilityError,
    CheckpointIdentity,
    ValidatedCheckpoint,
)
from virtual_staining.config import reject_unknown_keys
from virtual_staining.config.run import RunConfig
from virtual_staining.definitions import (
    Component,
    ComponentContext,
    ComponentDefinition,
    MethodDefinition,
    ResolutionContext,
)
from virtual_staining.training.runtime import MethodMetrics

SOURCE = "tests.external_method.tiny_reconstruction"
ARCHITECTURES = frozenset({"tiny_conv", "tiny_residual"})
_METHOD_KEYS = frozenset({"architecture", "learning_rate", "plateau_monitor"})


class _TinyNetwork(nn.Module):
    """Named RGB inputs in [-1, 1] -> a one-item mapping to one RGB output in [-1, 1]."""

    def __init__(
        self, input_names: tuple[str, ...], output_name: str, width: int, residual: bool
    ) -> None:
        super().__init__()
        self.input_names = input_names
        self.output_name = output_name
        self.head = nn.Conv2d(3 * len(input_names), width, 3, padding=1)
        self.body = nn.Conv2d(width, width, 3, padding=1) if residual else None
        self.tail = nn.Conv2d(width, 3, 3, padding=1)

    def forward(self, inputs: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        features = torch.relu(self.head(torch.cat([inputs[name] for name in self.input_names], 1)))
        if self.body is not None:
            features = features + torch.relu(self.body(features))
        return {self.output_name: torch.tanh(self.tail(features))}


def _parse_width(raw: Mapping[str, Any], context: ComponentContext) -> dict[str, Any]:
    reject_unknown_keys(raw, frozenset({"width"}), context.field)
    width = raw.get("width", 8)
    if isinstance(width, bool) or not isinstance(width, int) or width < 1:
        raise ValueError(f"{context.field}.width must be an integer >= 1")
    return {"width": width}


TINY_CONV = ComponentDefinition(
    name="tiny_conv",
    version="1",
    source=SOURCE,
    parse_options=_parse_width,
    factory=lambda options, *, input_names, output_name: _TinyNetwork(
        input_names, output_name, options["width"], False
    ),
)
TINY_RESIDUAL = ComponentDefinition(
    name="tiny_residual",
    version="1",
    source=SOURCE,
    parse_options=_parse_width,
    factory=lambda options, *, input_names, output_name: _TinyNetwork(
        input_names, output_name, options["width"], True
    ),
)


@dataclass(frozen=True)
class TinyOptions:
    network: Component
    learning_rate: float
    plateau_monitor: str | None = None


class TinyReconstruction(MethodDefinition):
    name = "tiny_reconstruction"
    version = "1"
    source = SOURCE
    pairing = "paired"
    checkpoint_metrics = MappingProxyType({"val_abs_bias": "min", "loss_recon_val": "min"})
    default_monitor = "val_abs_bias"

    def __init__(self) -> None:
        self.built: Counter[str] = Counter()

    def parse_options(
        self, sections: Mapping[str, Mapping[str, Any]], context: ResolutionContext
    ) -> TinyOptions:
        raw = sections["method"].get("options", {})
        if not isinstance(raw, Mapping):
            raise TypeError("method.options must be a YAML mapping")
        component_raw = {key: value for key, value in raw.items() if key not in _METHOD_KEYS}
        architecture = raw.get("architecture", "tiny_conv")
        component = context.component(architecture, "method.options.architecture")
        if component.name not in ARCHITECTURES:
            raise ValueError(f"method.options.architecture must be one of {sorted(ARCHITECTURES)}")
        learning_rate = raw.get("learning_rate", 1e-3)
        if isinstance(learning_rate, bool) or not isinstance(learning_rate, int | float):
            raise TypeError("method.options.learning_rate must be a number")
        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("method.options.learning_rate must be a finite number > 0")
        monitor = raw.get("plateau_monitor")
        if monitor is not None:
            self.monitor_mode(monitor, "method.options.plateau_monitor")
        return TinyOptions(
            network=component.resolve(component_raw, context.component_context("method.options")),
            learning_rate=float(learning_rate),
            plateau_monitor=monitor,
        )

    def options_to_sections(self, options: TinyOptions) -> dict[str, dict[str, Any]]:
        resolved: dict[str, Any] = {
            "architecture": options.network.name,
            **options.network.options,
            "learning_rate": options.learning_rate,
        }
        if options.plateau_monitor is not None:
            resolved["plateau_monitor"] = options.plateau_monitor
        return {"method": {"options": resolved}}

    def validate(self, config: RunConfig) -> None:
        assert config.model is not None
        if len(config.model.outputs) != 1:
            raise ValueError("tiny_reconstruction predicts exactly one model.outputs entry")

    def component_identities(self, options: TinyOptions) -> Mapping[str, Mapping[str, Any]]:
        return {"network": options.network.identity()}

    def build_network(self, config: RunConfig, device: torch.device) -> nn.Module:
        assert config.method is not None
        assert config.model is not None
        self.built["network"] += 1
        options: TinyOptions = config.method.options
        return options.network.build(
            input_names=tuple(config.model.inputs), output_name=config.model.outputs[0]
        ).to(device)

    def build_training_runtime(
        self,
        config: RunConfig,
        device: torch.device,
        *,
        seed: int,
        benchmark_recorder: object = None,
    ) -> TinyRuntime:
        del seed, benchmark_recorder
        return TinyRuntime(self, config, device)

    def build_inference_model(
        self,
        config: RunConfig,
        checkpoint: ValidatedCheckpoint,
        *,
        direction: str | None,
        device: torch.device,
    ) -> nn.Module:
        del direction
        network = self.build_network(config, device)
        network.load_state_dict(_network_state(checkpoint.state, network))
        return network


def _network_state(state: Mapping[str, Any], network: nn.Module) -> Mapping[str, Any]:
    stored = state.get("network")
    expected = network.state_dict()
    if not isinstance(stored, Mapping) or set(stored) != set(expected):
        raise CheckpointCompatibilityError("tiny_reconstruction state.network keys differ")
    for key, tensor in expected.items():
        value = stored[key]
        if not isinstance(value, torch.Tensor) or value.shape != tensor.shape:
            raise CheckpointCompatibilityError(f"tiny_reconstruction state.network.{key} differs")
    return stored


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


class TinyRuntime:
    """Training runtime: one network, one Adam optimizer, one L1 objective."""

    metric_names = ("loss_recon",)
    component_total_names: tuple[str, ...] = ()
    loss_names: tuple[str, ...] = ()
    validation_metric_names = ("val_abs_bias",)

    def __init__(self, definition: TinyReconstruction, config: RunConfig, device: torch.device):
        assert config.method is not None
        assert config.model is not None
        options: TinyOptions = config.method.options
        self.name = definition.name
        self.default_checkpoint_metric = next(iter(definition.checkpoint_metrics))
        self._definition = definition
        self._identity = definition.checkpoint_identity(config)
        self._device = device
        self._input_names = tuple(config.model.inputs)
        self._output_name = config.model.outputs[0]
        self._plateau_monitor = options.plateau_monitor
        self.network = definition.build_network(config, device)
        definition.built["optimizer"] += 1
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=options.learning_rate)
        definition.built["objective"] += 1
        self.objective = nn.L1Loss()
        self.scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None = None
        if options.plateau_monitor is not None:
            definition.built["scheduler"] += 1
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", patience=0, factor=0.5
            )

    def _unpack(self, batch: Any) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        inputs = {name: batch["inputs"][name].to(self._device) for name in self._input_names}
        return inputs, batch["targets"][self._output_name].to(self._device)

    def train_mode(self) -> None:
        self.network.train()

    def batch_size(self, batch: Any) -> int:
        return int(batch["targets"][self._output_name].shape[0])

    def step(self, batch: object, *, epoch: int, global_step: int) -> MethodMetrics:
        del epoch, global_step
        inputs, target = self._unpack(batch)
        self.optimizer.zero_grad()
        loss = self.objective(self.network(inputs)[self._output_name], target)
        loss.backward()
        self.optimizer.step()
        return MethodMetrics(losses={"loss_recon": float(loss.detach())})

    def validate(self, loader: Any, *, epoch: int, preview_sink: object = None) -> MethodMetrics:
        del epoch, preview_sink
        self.network.eval()
        losses: list[float] = []
        biases: list[float] = []
        with torch.no_grad():
            for batch in loader:
                inputs, target = self._unpack(batch)
                prediction = self.network(inputs)[self._output_name]
                losses.append(float(self.objective(prediction, target)))
                biases.append(float((prediction - target).mean().abs()))
        self.network.train()
        return MethodMetrics(
            losses={"loss_recon": _mean(losses)}, image={"val_abs_bias": _mean(biases)}
        )

    def validation_metric(self, metrics: MethodMetrics, name: str) -> float | None:
        if name == "loss_recon_val":
            return metrics.losses.get("loss_recon")
        return metrics.image.get(name)

    def checkpoint_selection_metrics(self, metrics: MethodMetrics) -> dict[str, float]:
        values = {
            name: self.validation_metric(metrics, name)
            for name in self.checkpoint_selection_modes()
        }
        return {
            name: value
            for name, value in values.items()
            if value is not None and math.isfinite(value)
        }

    def checkpoint_selection_modes(self) -> dict[str, str]:
        return dict(self._definition.checkpoint_metrics)

    def step_schedulers(self, *, epoch: int, validation_metrics: MethodMetrics | None) -> bool:
        del epoch
        if self.scheduler is None or validation_metrics is None or self._plateau_monitor is None:
            return False
        value = self.validation_metric(validation_metrics, self._plateau_monitor)
        if value is None or not math.isfinite(value):
            return False
        self.scheduler.step(value)
        return True

    def learning_rates(self) -> Mapping[str, float]:
        return {"lr": float(self.optimizer.param_groups[0]["lr"])}

    def checkpoint_identity(self) -> CheckpointIdentity:
        return self._identity

    def objective_metadata(self) -> dict[str, Any]:
        return {"objective": "l1_reconstruction", "prediction": "network(inputs)"}

    def state_dict(self) -> dict[str, Any]:
        return {
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": None if self.scheduler is None else self.scheduler.state_dict(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if set(state) != {"network", "optimizer", "scheduler"}:
            raise CheckpointCompatibilityError("tiny_reconstruction state keys differ")
        network = _network_state(state, self.network)
        if (state["scheduler"] is None) != (self.scheduler is None):
            raise CheckpointCompatibilityError("tiny_reconstruction scheduler presence differs")
        self.network.load_state_dict(network)
        self.optimizer.load_state_dict(state["optimizer"])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(state["scheduler"])
