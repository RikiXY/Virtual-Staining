"""One network, one optimizer, one paired reconstruction objective."""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import torch
from torch import nn

from virtual_staining.checkpoint_contract import (
    CheckpointCompatibilityError,
    CheckpointIdentity,
    ValidatedCheckpoint,
)
from virtual_staining.config import reject_unknown_keys
from virtual_staining.definitions import (
    Component,
    ComponentContext,
    ComponentDefinition,
    MethodDefinition,
    ResolutionContext,
)
from virtual_staining.training.runtime import MethodMetrics

SOURCE = "mexina_external_example"


class TinyNetwork(nn.Module):
    def __init__(self, input_names, output_name, width, residual):
        super().__init__()
        self.input_names, self.output_name = input_names, output_name
        self.head = nn.Conv2d(3 * len(input_names), width, 3, padding=1)
        self.body = nn.Conv2d(width, width, 3, padding=1) if residual else None
        self.tail = nn.Conv2d(width, 3, 1)

    def forward(self, inputs):
        features = self.head(torch.cat([inputs[name] for name in self.input_names], dim=1)).relu()
        if self.body is not None:
            features = features + self.body(features).relu()
        return {self.output_name: self.tail(features).tanh()}


def parse_width(raw, context: ComponentContext):
    reject_unknown_keys(raw, frozenset({"width"}), context.field)
    width = raw.get("width", 4)
    if type(width) is not int or width < 1:
        raise ValueError(f"{context.field}.width must be a positive integer")
    return {"width": width}


TINY_CONV = ComponentDefinition(
    "tiny_conv",
    "1",
    SOURCE,
    parse_width,
    lambda options, **names: TinyNetwork(**names, width=options["width"], residual=False),
)
TINY_RESIDUAL = ComponentDefinition(
    "tiny_residual",
    "1",
    SOURCE,
    parse_width,
    lambda options, **names: TinyNetwork(**names, width=options["width"], residual=True),
)


@dataclass(frozen=True)
class Options:
    network: Component
    learning_rate: float


class Reconstruction(MethodDefinition):
    name, version, source = "external_reconstruction", "1", SOURCE
    pairing = "paired"
    checkpoint_metrics = MappingProxyType({"loss_reconstruction_val": "min"})
    default_monitor = "loss_reconstruction_val"

    def parse_options(self, sections, context: ResolutionContext):
        raw = sections["method"].get("options", {})
        if not isinstance(raw, Mapping):
            raise TypeError("method.options must be a mapping")
        reject_unknown_keys(
            raw, frozenset({"architecture", "width", "learning_rate"}), "method.options"
        )
        component = context.component(
            raw.get("architecture", "tiny_conv"), "method.options.architecture"
        )
        if component.name not in {TINY_CONV.name, TINY_RESIDUAL.name}:
            raise ValueError("This example requires tiny_conv or tiny_residual")
        lr = raw.get("learning_rate", 0.001)
        if type(lr) not in (int, float) or not math.isfinite(lr) or lr <= 0:
            raise ValueError("method.options.learning_rate must be finite and positive")
        return Options(
            component.resolve(
                {"width": raw["width"]} if "width" in raw else {},
                context.component_context("method.options"),
            ),
            float(lr),
        )

    def options_to_sections(self, options):
        return {
            "method": {
                "options": {
                    "architecture": options.network.name,
                    **options.network.options,
                    "learning_rate": options.learning_rate,
                }
            }
        }

    def validate(self, config):
        if len(config.model.outputs) != 1:
            raise ValueError("This example method requires exactly one output")

    def component_identities(self, options):
        return {"network": options.network.identity()}

    def build_network(self, config, device):
        return config.method.options.network.build(
            input_names=tuple(config.model.inputs), output_name=config.model.outputs[0]
        ).to(device)

    def build_training_runtime(self, config, device, *, seed, benchmark_recorder=None):
        torch.manual_seed(seed)
        return Runtime(self, config, device)

    def build_inference_model(self, config, checkpoint: ValidatedCheckpoint, *, direction, device):
        network = self.build_network(config, device)
        network.load_state_dict(network_state(checkpoint.state, network))
        return network


def network_state(state, network):
    stored, expected = state.get("network"), network.state_dict()
    if not isinstance(stored, Mapping) or set(stored) != set(expected):
        raise CheckpointCompatibilityError("External network state keys differ")
    for key, tensor in expected.items():
        if not isinstance(stored[key], torch.Tensor) or stored[key].shape != tensor.shape:
            raise CheckpointCompatibilityError(f"External network state shape differs: {key}")
    return stored


class Runtime:
    metric_names = ("loss_reconstruction",)
    component_total_names = loss_names = validation_metric_names = ()

    def __init__(self, definition, config, device):
        self.name = definition.name
        self.default_checkpoint_metric = definition.default_monitor
        self.identity = definition.checkpoint_identity(config)
        self.device, self.output_name = device, config.model.outputs[0]
        self.network = definition.build_network(config, device)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(), lr=config.method.options.learning_rate
        )
        self.objective = nn.L1Loss()

    def train_mode(self):
        self.network.train()

    def batch_size(self, batch):
        return batch["targets"][self.output_name].shape[0]

    def loss(self, batch):
        inputs = {name: batch["inputs"][name].to(self.device) for name in self.identity.inputs}
        return self.objective(
            self.network(inputs)[self.output_name],
            batch["targets"][self.output_name].to(self.device),
        )

    def step(self, batch, *, epoch, global_step):
        self.optimizer.zero_grad()
        loss = self.loss(batch)
        loss.backward()
        self.optimizer.step()
        return MethodMetrics(losses={"loss_reconstruction": float(loss.detach())})

    def validate(self, loader, *, epoch, preview_sink=None):
        self.network.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for batch in loader:
                size = self.batch_size(batch)
                total += float(self.loss(batch)) * size
                count += size
        self.network.train()
        return MethodMetrics(losses={"loss_reconstruction": total / count if count else math.nan})

    def validation_metric(self, metrics, name):
        return (
            metrics.losses.get("loss_reconstruction")
            if name == self.default_checkpoint_metric
            else None
        )

    def checkpoint_selection_metrics(self, metrics):
        value = self.validation_metric(metrics, self.default_checkpoint_metric)
        return (
            {self.default_checkpoint_metric: value}
            if value is not None and math.isfinite(value)
            else {}
        )

    def checkpoint_selection_modes(self):
        return {self.default_checkpoint_metric: "min"}

    def step_schedulers(self, *, epoch, validation_metrics):
        return False

    def learning_rates(self):
        return {"lr": self.optimizer.param_groups[0]["lr"]}

    def checkpoint_identity(self) -> CheckpointIdentity:
        return self.identity

    def objective_metadata(self):
        return {"objective": "paired_l1"}

    def state_dict(self):
        return {"network": self.network.state_dict(), "optimizer": self.optimizer.state_dict()}

    def load_state_dict(self, state):
        if set(state) != {"network", "optimizer"}:
            raise CheckpointCompatibilityError("External runtime state keys differ")
        self.network.load_state_dict(network_state(state, self.network))
        self.optimizer.load_state_dict(state["optimizer"])
