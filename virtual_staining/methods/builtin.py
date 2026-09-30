"""Built-in Pix2Pix and CycleGAN definitions: the default definition set.

These use exactly the definition mechanism external callers use. Each definition owns
its public config keys, their validation, its validation-metric policy, and the
construction of its training runtime and inference-only generator. Importing this
module does not import torch; runtimes are imported when built.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from virtual_staining.checkpoint_selection import CheckpointMode
from virtual_staining.config.losses import LossConfig, parse_loss_config
from virtual_staining.config.scheduler import (
    LearningRateSchedulerConfig,
    parse_learning_rate_scheduler_config,
)
from virtual_staining.definitions import (
    Component,
    Definitions,
    MethodDefinition,
    ResolutionContext,
)
from virtual_staining.loss_definitions import method_loss_names
from virtual_staining.metrics import BUILTIN_METRIC_DEFINITIONS, BUILTIN_METRICS, ResolvedMetric
from virtual_staining.models.components import BUILTIN_COMPONENTS, BUILTIN_SOURCE

if TYPE_CHECKING:
    import torch

    from virtual_staining.checkpoint_contract import ValidatedCheckpoint
    from virtual_staining.config.run import RunConfig
    from virtual_staining.training.benchmarking import TrainingBenchmarkRecorder
    from virtual_staining.training.runtime import TrainingMethodRuntime

__all__ = [
    "CycleGANDefinition",
    "GanOptions",
    "GanTrainingOptions",
    "PIX2PIX_IMAGE_METRICS",
    "Pix2PixDefinition",
    "builtin_definitions",
    "builtin_method_definitions",
    "pix2pix_validation_metrics",
]

DEFAULT_CYCLEGAN_REPLAY_BUFFER_SIZE = 50
DEFAULT_GENERATOR_ARCHITECTURE = "concat_unet"
# Validation columns both built-ins report for their configured loss terms; a Pix2Pix
# reconstruction term column ends in ``__<output>``.
_LOSS_MONITOR_PATTERN = re.compile(
    r"^loss_val_(?:total_(?:generator|discriminator)|"
    r"(?:raw|weighted|current_weight)_(?:generator|discriminator)_[A-Za-z0-9_-]+)$"
)
# Pix2Pix deliberately reuses these built-in evaluation metrics for validation, once per
# model output (``val_<metric>__<output>``); no column mixes outputs. The definition
# below, not the evaluation subsystem, owns which exist and how they rank.
PIX2PIX_IMAGE_METRICS: Mapping[str, ResolvedMetric] = MappingProxyType(
    {
        name: BUILTIN_METRIC_DEFINITIONS[name].resolve({}, name)
        for name in ("ssim", "psnr", "mae", "rmse", "pcc_rgb_mean", "pcc_gray")
    }
)
_VALIDATION_COLUMN = re.compile(
    rf"^val_(?P<metric>{'|'.join(PIX2PIX_IMAGE_METRICS)})__(?P<output>[A-Za-z][A-Za-z0-9_-]*)$"
)
_OUTPUT_SUFFIX = re.compile(r"__(?P<output>[A-Za-z][A-Za-z0-9_-]*)$")


def pix2pix_validation_metrics(outputs: tuple[str, ...]) -> dict[str, tuple[str, ResolvedMetric]]:
    """Validation column -> (output, metric), metric-major so one output keeps metric order."""
    return {
        f"val_{name}__{output}": (output, metric)
        for name, metric in PIX2PIX_IMAGE_METRICS.items()
        for output in outputs
    }


def _metric_mode(metric: ResolvedMetric) -> CheckpointMode:
    return "max" if metric.definition.higher_is_better else "min"


@dataclass(frozen=True)
class GanTrainingOptions:
    """Adam optimization, LR scheduler and configured loss terms of a built-in GAN."""

    lr_g: float
    lr_d: float
    beta1: float
    beta2: float
    scheduler: LearningRateSchedulerConfig
    losses: LossConfig

    def __post_init__(self) -> None:
        for field_name, value in (("lr_g", self.lr_g), ("lr_d", self.lr_d)):
            if value <= 0:
                raise ValueError(f"{field_name} must be greater than 0")
        for field_name, value in (("beta1", self.beta1), ("beta2", self.beta2)):
            if not (0.0 <= value < 1.0):
                raise ValueError(f"{field_name} must be in [0, 1)")

    def to_dict(self) -> dict[str, Any]:
        return {
            "lr_g": self.lr_g,
            "lr_d": self.lr_d,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "scheduler": self.scheduler.to_dict(),
            "losses": self.losses.to_dict(),
        }


@dataclass(frozen=True)
class GanOptions:
    """Resolved options of a built-in GAN; ``training`` is None without a training section."""

    generator: Component
    discriminator: Component
    training: GanTrainingOptions | None
    replay_buffer_size: int | None = None


class _GanDefinition(MethodDefinition):
    """Configuration shared by the built-in adversarial methods."""

    version = "1"
    source = BUILTIN_SOURCE
    generator_architecture: str
    owned_keys: Mapping[str, frozenset[str]] = MappingProxyType(
        {
            "model": frozenset({"generator", "discriminator"}),
            "training": frozenset({"lr_g", "lr_d", "beta1", "beta2", "scheduler", "losses"}),
        }
    )

    def parse_options(
        self, sections: Mapping[str, Mapping[str, Any]], context: ResolutionContext
    ) -> GanOptions:
        model = sections.get("model", {})
        return GanOptions(
            generator=self._generator(model.get("generator", {}), context),
            discriminator=context.component("patchgan", "model.discriminator").resolve(
                model.get("discriminator", {}), context.component_context("model.discriminator")
            ),
            training=(
                None
                if context.training is None
                else self._training(sections.get("training", {}), context.training.epochs)
            ),
        )

    def _generator(self, raw: object, context: ResolutionContext) -> Component:
        if not isinstance(raw, Mapping):
            raise TypeError("model.generator must be a YAML mapping")
        name = raw.get("architecture", DEFAULT_GENERATOR_ARCHITECTURE)
        definition = context.component(name, "model.generator.architecture")
        if definition.name != self.generator_architecture:
            raise ValueError(
                f"method.name={self.name!r} requires "
                f"model.generator.architecture={self.generator_architecture!r}"
            )
        options = {key: value for key, value in raw.items() if key != "architecture"}
        return definition.resolve(options, context.component_context("model.generator"))

    def _training(self, raw: Mapping[str, Any], epochs: int) -> GanTrainingOptions:
        if "losses" not in raw:
            raise ValueError("training.losses is required")
        losses = parse_loss_config(raw["losses"])
        supported = method_loss_names(self.name)
        unsupported = sorted(
            {term.name for term in (*losses.generator, *losses.discriminator)} - supported
        )
        if unsupported:
            raise ValueError(
                f"training.losses {unsupported} are not supported by method.name={self.name!r}; "
                f"supported: {sorted(supported)}"
            )
        return GanTrainingOptions(
            lr_g=float(raw.get("lr_g", 2e-4)),
            lr_d=float(raw.get("lr_d", 2e-4)),
            beta1=float(raw.get("beta1", 0.5)),
            beta2=float(raw.get("beta2", 0.999)),
            scheduler=parse_learning_rate_scheduler_config(
                raw.get("scheduler", {}),
                epochs=epochs,
                monitor_mode=self.checkpoint_metric_mode,
                default_monitor=next(iter(self.checkpoint_metrics)),
            ),
            losses=losses,
        )

    def options_to_sections(self, options: GanOptions) -> dict[str, dict[str, Any]]:
        sections: dict[str, dict[str, Any]] = {
            "model": {
                "generator": {"architecture": options.generator.name, **options.generator.options},
                "discriminator": dict(options.discriminator.options),
            }
        }
        if options.training is not None:
            sections["training"] = options.training.to_dict()
        return sections

    def monitor_mode(self, monitor: str, field: str) -> CheckpointMode:
        if monitor in {"loss_G_val", "loss_D_val"} or _LOSS_MONITOR_PATTERN.fullmatch(monitor):
            return "min"
        return self.checkpoint_metric_mode(monitor, field)

    def requires_foreground_mask(self, config: RunConfig) -> bool:
        training = config.method.options.training
        return training is not None and any(
            term.requires_mask for term in training.losses.generator
        )


class Pix2PixDefinition(_GanDefinition):
    """Paired Pix2Pix: ConcatUNet over N named inputs -> M named outputs, one joint
    conditional PatchGAN over all inputs and outputs.

    Validation image metrics are per output (``val_<metric>__<output>``); ``loss_G_val``
    is the default checkpoint metric. One output defaults early stopping to its
    ``val_ssim`` column; several outputs require an explicit monitor.
    """

    name = "pix2pix"
    pairing = "paired"
    prediction_directions = ("forward",)
    checkpoint_metrics: Mapping[str, CheckpointMode] = MappingProxyType({"loss_G_val": "min"})
    default_monitor = None
    generator_architecture = "concat_unet"

    def checkpoint_metric_mode(self, metric: str, field: str) -> CheckpointMode:
        match = _VALIDATION_COLUMN.fullmatch(metric)
        if match is not None:
            return _metric_mode(PIX2PIX_IMAGE_METRICS[match["metric"]])
        if metric not in self.checkpoint_metrics:
            raise ValueError(
                f"{field}={metric!r} is not a checkpoint metric of method.name='pix2pix'; "
                "supported: loss_G_val or val_<metric>__<output> with <metric> one of "
                f"{list(PIX2PIX_IMAGE_METRICS)}"
            )
        return self.checkpoint_metrics[metric]

    def checkpoint_modes(self, outputs: tuple[str, ...]) -> dict[str, CheckpointMode]:
        """Every ranked checkpoint metric for these outputs; ``loss_G_val`` first."""
        return {
            **self.checkpoint_metrics,
            **{
                column: _metric_mode(metric)
                for column, (_, metric) in pix2pix_validation_metrics(outputs).items()
            },
        }

    def resolve_default_monitor(self, outputs: tuple[str, ...]) -> str | None:
        return f"val_ssim__{outputs[0]}" if len(outputs) == 1 else None

    def validate(self, config: RunConfig) -> None:
        training = config.training
        options = config.method.options.training
        named = {
            "inference.checkpoint_metric": (
                config.inference.checkpoint_metric if config.inference is not None else None
            ),
            "training.early_stopping.monitor": (
                training.early_stopping.monitor
                if training is not None and training.early_stopping is not None
                else None
            ),
            "training.scheduler.monitor": (
                options.scheduler.monitor if options is not None else None
            ),
        }
        for field, name in named.items():
            match = _OUTPUT_SUFFIX.search(name or "")
            if match is not None and match["output"] not in config.model.outputs:
                raise ValueError(
                    f"{field}={name!r} names output {match['output']!r}, which is not one of "
                    f"model.outputs {list(config.model.outputs)}"
                )

    def component_identities(self, options: GanOptions) -> Mapping[str, Mapping[str, Any]]:
        return {
            "generator": options.generator.identity(),
            "discriminator": options.discriminator.identity(),
        }

    def build_training_runtime(
        self,
        config: RunConfig,
        device: torch.device,
        *,
        seed: int,
        benchmark_recorder: TrainingBenchmarkRecorder | None = None,
    ) -> TrainingMethodRuntime:
        del seed
        from virtual_staining.methods.pix2pix import Pix2PixMethod

        return Pix2PixMethod(config, device, benchmark_recorder=benchmark_recorder)

    def build_inference_model(
        self,
        config: RunConfig,
        checkpoint: ValidatedCheckpoint,
        *,
        direction: str | None,
        device: torch.device,
    ) -> torch.nn.Module:
        del direction
        from virtual_staining.methods.pix2pix import build_pix2pix_inference_generator

        return build_pix2pix_inference_generator(config, checkpoint, device)


class CycleGANDefinition(_GanDefinition):
    """Unpaired CycleGAN: two ResNet generators, two PatchGANs, replay pools.

    Exactly one input and one output: domain A is ``model.inputs[0]`` and domain B is
    ``model.outputs[0]``. Its two prediction directions are alternatives, never two
    simultaneous outputs: A_to_B predicts B and B_to_A predicts A.
    """

    name = "cyclegan"
    pairing = "unpaired"
    prediction_directions = ("A_to_B", "B_to_A")
    checkpoint_metrics: Mapping[str, CheckpointMode] = MappingProxyType({"loss_G_val": "min"})
    default_monitor = None
    generator_architecture = "resnet"
    owned_keys: Mapping[str, frozenset[str]] = MappingProxyType(
        {**_GanDefinition.owned_keys, "method": frozenset({"replay_buffer_size"})}
    )

    def parse_options(
        self, sections: Mapping[str, Mapping[str, Any]], context: ResolutionContext
    ) -> GanOptions:
        size = sections.get("method", {}).get(
            "replay_buffer_size", DEFAULT_CYCLEGAN_REPLAY_BUFFER_SIZE
        )
        if isinstance(size, bool) or not isinstance(size, int):
            raise TypeError("method.replay_buffer_size must be an integer")
        if size < 0:
            raise ValueError("method.replay_buffer_size must be >= 0")
        options = super().parse_options(sections, context)
        return GanOptions(
            generator=options.generator,
            discriminator=options.discriminator,
            training=options.training,
            replay_buffer_size=size,
        )

    def options_to_sections(self, options: GanOptions) -> dict[str, dict[str, Any]]:
        return {
            "method": {"replay_buffer_size": options.replay_buffer_size},
            **super().options_to_sections(options),
        }

    def checkpoint_metric_mode(self, metric: str, field: str) -> CheckpointMode:
        self._reject_paired_metric(metric, field)
        return super().checkpoint_metric_mode(metric, field)

    def monitor_mode(self, monitor: str, field: str) -> CheckpointMode:
        self._reject_paired_metric(monitor, field)
        return super().monitor_mode(monitor, field)

    def _reject_paired_metric(self, metric: str, field: str) -> None:
        if metric.startswith("val_"):
            raise ValueError(
                f"{field}={metric!r} is a paired image-fidelity metric that "
                "method.name='cyclegan' does not report; use loss_G_val or a loss_val_* column"
            )

    def validate(self, config: RunConfig) -> None:
        if len(config.model.inputs) != 1:
            raise ValueError(
                "method.name='cyclegan' requires exactly one model.inputs entry (domain A); "
                f"got {list(config.model.inputs)}"
            )
        if len(config.model.outputs) != 1:
            raise ValueError(
                "method.name='cyclegan' requires exactly one model.outputs entry (domain B); "
                f"got {list(config.model.outputs)}. Its two directions are alternatives, "
                "not simultaneous outputs."
            )
        domain_a, domain_b = config.model.inputs[0], config.model.outputs[0]
        missing = sorted({domain_a, domain_b} - set(config.data.domains))
        extra = sorted(set(config.data.domains) - {domain_a, domain_b})
        if missing or extra:
            raise ValueError(
                f"data.domains keys must be exactly [{domain_a!r}, {domain_b!r}] "
                f"(model.inputs[0], model.outputs[0]); missing={missing}, extra={extra}"
            )
        if config.training is None:
            return
        if config.training.augmentation.enabled:
            raise ValueError("method.name='cyclegan' requires training.augmentation.enabled=false")
        losses = config.method.options.training.losses
        generator = {term.name for term in losses.active_generator}
        discriminator = {term.name for term in losses.active_discriminator}
        for role, names, required in (
            ("generator", generator, "adversarial_lsgan"),
            ("generator", generator, "cycle_l1"),
            ("discriminator", discriminator, "adversarial_lsgan"),
        ):
            if required not in names:
                raise ValueError(
                    f"method.name='cyclegan' requires an active training.losses.{role} "
                    f"term {required!r}"
                )

    def prediction_inputs(self, config: RunConfig, direction: str | None) -> tuple[str, ...]:
        return (config.model.inputs[0],) if direction == "A_to_B" else (config.model.outputs[0],)

    def prediction_outputs(self, config: RunConfig, direction: str | None) -> tuple[str, ...]:
        return (config.model.outputs[0],) if direction == "A_to_B" else (config.model.inputs[0],)

    def component_identities(self, options: GanOptions) -> Mapping[str, Mapping[str, Any]]:
        generator = options.generator.identity()
        discriminator = options.discriminator.identity()
        return {
            "G_A_to_B": generator,
            "G_B_to_A": dict(generator),
            "D_A": discriminator,
            "D_B": dict(discriminator),
        }

    def build_training_runtime(
        self,
        config: RunConfig,
        device: torch.device,
        *,
        seed: int,
        benchmark_recorder: TrainingBenchmarkRecorder | None = None,
    ) -> TrainingMethodRuntime:
        del benchmark_recorder
        from virtual_staining.methods.cyclegan import CycleGANMethod

        return CycleGANMethod(config, device, seed=seed)

    def build_inference_model(
        self,
        config: RunConfig,
        checkpoint: ValidatedCheckpoint,
        *,
        direction: str | None,
        device: torch.device,
    ) -> torch.nn.Module:
        from virtual_staining.methods.cyclegan import build_cyclegan_inference_generator

        if direction is None:
            raise ValueError("CycleGAN inference requires a prediction direction")
        return build_cyclegan_inference_generator(config, checkpoint, direction, device)


def builtin_method_definitions() -> tuple[MethodDefinition, ...]:
    return (Pix2PixDefinition(), CycleGANDefinition())


@cache
def builtin_definitions() -> Definitions:
    """The default definition set: the built-in methods, components and metrics."""
    return Definitions().extend(
        methods=builtin_method_definitions(),
        components=BUILTIN_COMPONENTS,
        metrics=BUILTIN_METRICS,
    )
