"""Explicit method, component and metric definitions: the Python extension seam.

A caller makes an implementation available by importing it and passing its definitions
in a :class:`Definitions` value; configuration then selects registered names. Nothing is
discovered, and neither YAML nor checkpoint metadata ever names Python code to import.

This module is torch-free so configuration resolution stays lightweight; definitions
import their runtime code only when asked to build a runtime or model.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from virtual_staining.checkpoint_selection import CheckpointMode

if TYPE_CHECKING:
    from pathlib import Path

    import torch

    from virtual_staining.checkpoint_contract import CheckpointIdentity, ValidatedCheckpoint
    from virtual_staining.config.run import RunConfig
    from virtual_staining.config.training import TrainingConfig
    from virtual_staining.metrics import MetricDefinition
    from virtual_staining.training.benchmarking import TrainingBenchmarkRecorder
    from virtual_staining.training.runtime import TrainingMethodRuntime

__all__ = [
    "Component",
    "ComponentContext",
    "ComponentDefinition",
    "DefinitionNotAvailableError",
    "Definitions",
    "MethodDefinition",
    "ResolutionContext",
]


class DefinitionNotAvailableError(ValueError):
    """Configuration or a checkpoint names a definition the caller did not supply."""


@dataclass(frozen=True)
class ComponentContext:
    """What a component option parser may check: its config path and the image size."""

    field: str
    image_size: tuple[int, int]


@dataclass(frozen=True)
class ComponentDefinition:
    """One registered network architecture.

    ``parse_options`` validates raw options strictly and returns them normalized, with
    defaults filled, as a JSON-compatible dict; that dict is both the resolved config
    spelling and the checkpoint reconstruction identity. ``factory(options, **context)``
    builds the module; which keyword context is passed is decided by the method that
    composes the component. Bump ``version`` whenever the same options would build a
    different module.
    """

    name: str
    version: str
    source: str
    parse_options: Callable[[Mapping[str, Any], ComponentContext], dict[str, Any]]
    factory: Callable[..., Any]

    def resolve(self, raw: Mapping[str, Any], context: ComponentContext) -> Component:
        if not isinstance(raw, Mapping):
            raise TypeError(f"{context.field} must be a YAML mapping")
        return Component(self, self.parse_options(raw, context))


@dataclass(frozen=True)
class Component:
    """A component definition bound to its validated options."""

    definition: ComponentDefinition
    options: Mapping[str, Any]

    @property
    def name(self) -> str:
        return self.definition.name

    def identity(self) -> dict[str, Any]:
        return {
            "name": self.definition.name,
            "version": self.definition.version,
            "source": self.definition.source,
            "options": dict(self.options),
        }

    def build(self, **context: Any) -> torch.nn.Module:
        return self.definition.factory(self.options, **context)


@dataclass(frozen=True)
class ResolutionContext:
    """Framework-common values a method may consult while resolving its options."""

    definitions: Definitions
    image_size: tuple[int, int]
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    training: TrainingConfig | None

    def component(self, name: object, field: str) -> ComponentDefinition:
        if not isinstance(name, str):
            raise TypeError(f"{field} must be a string")
        return self.definitions.component(name, field=field)

    def component_context(self, field: str) -> ComponentContext:
        return ComponentContext(field=field, image_size=self.image_size)


class MethodDefinition(ABC):
    """One registered image-translation method: named RGB inputs -> named RGB outputs.

    A definition owns everything method-specific: which config keys it reads, their
    validation and defaults, the validation metrics it reports and ranks, training
    runtime construction, inference-only model construction, and the reconstruction
    identity recorded in checkpoints. The shared ``Trainer`` owns the epoch, checkpoint
    and history lifecycle; the shared inference layer owns image transport.
    """

    #: Stable registered name selected by ``method.name``.
    name: str
    #: Implementation version; bump when the same options build a different model.
    version: str
    #: Package/provider identity, e.g. the distribution or module that owns the code.
    source: str
    #: ``data.pairing`` the method trains on: ``paired`` or ``unpaired``.
    pairing: str
    #: Prediction directions; ``inference.direction`` selects one when there are several.
    prediction_directions: tuple[str, ...] = ("forward",)
    #: Validation metrics ranked in ``checkpoints/best.json`` with their ranking mode.
    #: The first entry is the default checkpoint metric.
    checkpoint_metrics: Mapping[str, CheckpointMode] = MappingProxyType({})
    #: Default ``training.early_stopping.monitor``; None makes the monitor required.
    #: ``resolve_default_monitor`` may derive it from the configured outputs instead.
    default_monitor: str | None = None
    #: Raw keys owned per config section; ``method.name`` and the common keys of every
    #: section belong to the framework. External methods default to ``method.options``.
    owned_keys: Mapping[str, frozenset[str]] = MappingProxyType({"method": frozenset({"options"})})

    # --- configuration -----------------------------------------------------------

    @abstractmethod
    def parse_options(
        self, sections: Mapping[str, Mapping[str, Any]], context: ResolutionContext
    ) -> Any:
        """Validate the owned raw keys (``sections[section][key]``) into method options."""

    @abstractmethod
    def options_to_sections(self, options: Any) -> dict[str, dict[str, Any]]:
        """Serialize options back to their owned config keys (deterministic, JSON-compatible)."""

    def validate(self, config: RunConfig) -> None:
        """Check method rules that span several sections of a resolved config."""
        del config

    def checkpoint_metric_mode(self, metric: str, field: str) -> CheckpointMode:
        """Return the ranking mode of checkpoint metric ``metric`` or reject it."""
        if metric not in self.checkpoint_metrics:
            raise ValueError(
                f"{field}={metric!r} is not a checkpoint metric of method.name={self.name!r}; "
                f"supported: {sorted(self.checkpoint_metrics)}"
            )
        return self.checkpoint_metrics[metric]

    def monitor_mode(self, monitor: str, field: str) -> CheckpointMode:
        """Return the natural mode of validation monitor ``monitor`` or reject it."""
        return self.checkpoint_metric_mode(monitor, field)

    def resolve_default_monitor(self, outputs: tuple[str, ...]) -> str | None:
        """Default early-stopping monitor for ``model.outputs``; None makes it required."""
        del outputs
        return self.default_monitor

    def prediction_inputs(self, config: RunConfig, direction: str | None) -> tuple[str, ...]:
        """Named inputs consumed when predicting in ``direction``."""
        del direction
        return tuple(config.model.inputs)

    def prediction_outputs(self, config: RunConfig, direction: str | None) -> tuple[str, ...]:
        """Ordered named outputs produced when predicting in ``direction``."""
        del direction
        return tuple(config.model.outputs)

    def requires_foreground_mask(self, config: RunConfig) -> bool:
        """Whether paired training batches must carry the prepared foreground mask."""
        del config
        return False

    # --- reconstruction identity -------------------------------------------------

    def reconstruction_options(self, options: Any) -> Mapping[str, Any]:
        """Method-level options, beyond components, that change the reconstructed model."""
        del options
        return {}

    @abstractmethod
    def component_identities(self, options: Any) -> Mapping[str, Mapping[str, Any]]:
        """Identity of every persisted component by role, usually ``Component.identity()``."""

    def checkpoint_identity(self, config: RunConfig) -> CheckpointIdentity:
        from virtual_staining.checkpoint_contract import CheckpointIdentity

        options = config.method.options
        return CheckpointIdentity(
            method=self.name,
            implementation={"version": self.version, "source": self.source},
            pairing=self.pairing,
            inputs=tuple(config.model.inputs),
            outputs=tuple(config.model.outputs),
            prediction_directions=tuple(self.prediction_directions),
            options=self.reconstruction_options(options),
            components=self.component_identities(options),
            image_size=config.project.image_size,
        )

    # --- runtime construction ------------------------------------------------------

    @abstractmethod
    def build_training_runtime(
        self,
        config: RunConfig,
        device: torch.device,
        *,
        seed: int,
        benchmark_recorder: TrainingBenchmarkRecorder | None = None,
    ) -> TrainingMethodRuntime:
        """Build the full training topology (models, optimizers, objectives, state)."""

    @abstractmethod
    def build_inference_model(
        self,
        config: RunConfig,
        checkpoint: ValidatedCheckpoint,
        *,
        direction: str | None,
        device: torch.device,
    ) -> torch.nn.Module:
        """Build only the prediction network and restore it from a validated checkpoint.

        The module maps ``{input_name: NCHW tensor in [-1, 1]}`` to
        ``{output_name: RGB NCHW tensor in [-1, 1]}`` on the same pixel grid, with exactly
        the names and order of ``prediction_outputs`` (one output is a one-item mapping);
        the framework supplies the input names from ``prediction_inputs``. No optimizer,
        scheduler, objective or unused network may be constructed.
        """


def _merge(existing: Mapping[str, Any], added: Iterable[Any], kind: str) -> Mapping[str, Any]:
    merged = dict(existing)
    for definition in added:
        name = definition.name
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{kind} definition name must be a non-blank string")
        if name in merged:
            raise ValueError(
                f"Duplicate {kind} definition {name!r}; definitions are never replaced"
            )
        merged[name] = definition
    return MappingProxyType(merged)


@dataclass(frozen=True)
class Definitions:
    """Immutable set of method, component and metric definitions supplied by the caller.

    ``metrics`` are standalone evaluation metrics selected by ``evaluation.metrics``. A
    method's training validation and checkpoint metrics are owned by its
    :class:`MethodDefinition` and never need to be registered here.
    """

    methods: Mapping[str, MethodDefinition] = field(default_factory=lambda: MappingProxyType({}))
    components: Mapping[str, ComponentDefinition] = field(
        default_factory=lambda: MappingProxyType({})
    )
    metrics: Mapping[str, MetricDefinition] = field(default_factory=lambda: MappingProxyType({}))

    def extend(
        self,
        *,
        methods: Iterable[MethodDefinition] = (),
        components: Iterable[ComponentDefinition] = (),
        metrics: Iterable[MetricDefinition] = (),
    ) -> Definitions:
        """Return a new set with the given definitions added; duplicate names are rejected."""
        return Definitions(
            methods=_merge(self.methods, methods, "method"),
            components=_merge(self.components, components, "component"),
            metrics=_merge(self.metrics, metrics, "metric"),
        )

    def method(self, name: str, *, field: str = "method.name") -> MethodDefinition:
        if name not in self.methods:
            raise DefinitionNotAvailableError(
                f"{field}={name!r} is not a registered method definition; registered: "
                f"{sorted(self.methods)}. Definitions are supplied explicitly in Python."
            )
        return self.methods[name]

    def component(self, name: str, *, field: str = "component") -> ComponentDefinition:
        if name not in self.components:
            raise DefinitionNotAvailableError(
                f"{field}={name!r} is not a registered component definition; registered: "
                f"{sorted(self.components)}. Definitions are supplied explicitly in Python."
            )
        return self.components[name]

    def require_checkpoint(self, payload: object, path: Path) -> None:
        """Reject a checkpoint whose method or components are not registered here.

        Only names are read; nothing in checkpoint metadata is ever imported.
        """
        method = payload.get("method") if isinstance(payload, Mapping) else None
        if not isinstance(method, Mapping):
            return  # malformed metadata is reported by checkpoint validation
        name = method.get("name")
        if isinstance(name, str) and name not in self.methods:
            raise DefinitionNotAvailableError(
                f"Checkpoint '{path}' was written by method {name!r}, but that registered "
                "definition is not available; supply it explicitly (checkpoint metadata "
                "never imports code)."
            )
        components = method.get("components")
        for role, identity in components.items() if isinstance(components, Mapping) else ():
            component = identity.get("name") if isinstance(identity, Mapping) else None
            if isinstance(component, str) and component not in self.components:
                raise DefinitionNotAvailableError(
                    f"Checkpoint '{path}' component {role!r} uses {component!r}, but that "
                    "registered definition is not available; supply it explicitly."
                )
