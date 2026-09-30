from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import torch
import torch.optim as optim
from torch.amp import GradScaler

from virtual_staining.checkpoint_contract import CheckpointIdentity, ValidatedCheckpoint
from virtual_staining.config.losses import configured_loss_names
from virtual_staining.config.run import RunConfig
from virtual_staining.methods.builtin import (
    GanOptions,
    GanTrainingOptions,
    Pix2PixDefinition,
    pix2pix_validation_metrics,
)
from virtual_staining.training.helpers import (
    TRAINING_STATE_KEYS,
    OptimizationRole,
    build_lr_scheduler,
    check_training_state,
    is_amp_enabled,
    load_training_state,
    loss_validation_metric,
    require_state_keys,
    restoring_validated_state,
    step_lr_schedulers,
    training_state_dict,
    unpack_batch,
    validated_model_state,
)
from virtual_staining.training.losses import ConfiguredLossEvaluator
from virtual_staining.training.preview import ValidationPreviewSink
from virtual_staining.training.runtime import MethodMetrics
from virtual_staining.training.steps import Pix2PixTrainingStep
from virtual_staining.training.validator import validate_epoch

if TYPE_CHECKING:
    from virtual_staining.training.benchmarking import TrainingBenchmarkRecorder

logger = logging.getLogger(__name__)


def build_pix2pix_inference_generator(
    config: RunConfig,
    checkpoint: ValidatedCheckpoint,
    device: torch.device,
) -> torch.nn.Module:
    """Build only the forward generator and restore it from a validated checkpoint."""
    options: GanOptions = config.method.options
    generator = options.generator.build(
        input_names=tuple(config.model.inputs), output_names=tuple(config.model.outputs)
    ).to(device)
    generator.load_state_dict(
        validated_model_state(checkpoint.state, "generator", generator, checkpoint.path)
    )
    generator.eval()
    return generator


class Pix2PixMethod:
    """Own the concrete paired Pix2Pix training topology: N named inputs -> M outputs."""

    metric_names: tuple[str, ...] = ("loss_G", "loss_D")
    component_total_names: tuple[str, ...] = ("generator", "discriminator")

    def __init__(
        self,
        config: RunConfig,
        device: torch.device,
        *,
        benchmark_recorder: TrainingBenchmarkRecorder | None = None,
    ) -> None:
        options: GanOptions = config.method.options
        if config.training is None or options.training is None:
            raise ValueError("training config is required to construct Pix2Pix")
        definition = config.method.definition
        if not isinstance(definition, Pix2PixDefinition):
            raise TypeError("Pix2PixMethod requires the Pix2Pix method definition")
        self.config = config
        self.name = definition.name
        self.input_names = tuple(config.model.inputs)
        self.output_names = tuple(config.model.outputs)
        self._image_metrics = pix2pix_validation_metrics(self.output_names)
        self.validation_metric_names = tuple(self._image_metrics)
        self._checkpoint_metrics = definition.checkpoint_modes(self.output_names)
        self.default_checkpoint_metric = next(iter(self._checkpoint_metrics))
        self._identity = definition.checkpoint_identity(config)
        self._epochs = config.training.epochs
        self._optimization: GanTrainingOptions = options.training
        self.device = device
        self._benchmark_recorder = benchmark_recorder
        self._amp_enabled = is_amp_enabled(device)
        self.loss_config = self._optimization.losses
        self.loss_names = tuple(configured_loss_names(self.loss_config, self.output_names))

        names = {"input_names": self.input_names, "output_names": self.output_names}
        self.generator = options.generator.build(**names).to(device)
        # One joint conditional PatchGAN: all inputs plus all real or generated outputs.
        self.discriminator = options.discriminator.build(**names).to(device)
        optimization = self._optimization
        self._opt_G = optim.Adam(
            self.generator.parameters(),
            lr=optimization.lr_g,
            betas=(optimization.beta1, optimization.beta2),
        )
        self._opt_D = optim.Adam(
            self.discriminator.parameters(),
            lr=optimization.lr_d,
            betas=(optimization.beta1, optimization.beta2),
        )

        self._scaler_G = GradScaler(enabled=self._amp_enabled)
        self._scaler_D = GradScaler(enabled=self._amp_enabled)
        scheduler = optimization.scheduler
        self._scheduler_G = build_lr_scheduler(scheduler, self._epochs, self._opt_G)
        self._scheduler_D = (
            build_lr_scheduler(scheduler, self._epochs, self._opt_D)
            if self.loss_config.active_discriminator
            else None
        )
        # One resolved evaluator shared by training steps and validation.
        self._loss_evaluator = ConfiguredLossEvaluator(
            generator_terms=self.loss_config.generator,
            discriminator_terms=self.loss_config.discriminator,
        )
        self._step = Pix2PixTrainingStep(
            generator=self.generator,
            discriminator=self.discriminator,
            opt_G=self._opt_G,
            opt_D=self._opt_D,
            scaler_G=self._scaler_G,
            scaler_D=self._scaler_D,
            device=device,
            amp_enabled=self._amp_enabled,
            loss_evaluator=self._loss_evaluator,
            benchmark_recorder=benchmark_recorder,
        )

    def train_mode(self) -> None:
        self.generator.train()
        self.discriminator.train()

    def batch_size(self, batch: object) -> int:
        targets = batch.get("targets") if isinstance(batch, dict) else None
        target = targets.get(self.output_names[0]) if isinstance(targets, dict) else None
        if not isinstance(target, torch.Tensor) or target.ndim == 0:
            return 0
        return int(target.shape[0])

    def step(self, batch: object, *, epoch: int, global_step: int) -> MethodMetrics:
        recorder = self._benchmark_recorder
        if recorder is None:
            inputs, targets, masks = self._unpack_batch(batch)
        else:
            with recorder.phase("h2d"):
                inputs, targets, masks = self._unpack_batch(batch)
        result = self._step.step(
            inputs,
            targets,
            epoch=epoch,
            global_step=global_step,
            masks=masks,
        )
        has_components = bool(result.raw or result.weighted or result.current_weight)
        return MethodMetrics(
            losses={"loss_G": result.loss_G, "loss_D": result.loss_D},
            component_totals=(
                {"generator": result.loss_G, "discriminator": result.loss_D}
                if has_components
                else {}
            ),
            raw=dict(result.raw or {}),
            weighted=dict(result.weighted or {}),
            current_weight=dict(result.current_weight or {}),
        )

    def _unpack_batch(
        self, batch: object
    ) -> tuple[
        dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, dict[str, torch.Tensor]]
    ]:
        return unpack_batch(batch, self.device, self.input_names, self.output_names)

    def validate(
        self,
        loader: torch.utils.data.DataLoader,
        *,
        epoch: int,
        preview_sink: ValidationPreviewSink | None = None,
    ) -> MethodMetrics:
        result = validate_epoch(
            epoch=epoch,
            generator=self.generator,
            discriminator=self.discriminator,
            val_loader=loader,
            loss_evaluator=self._loss_evaluator,
            losses=self.loss_config,
            device=self.device,
            amp_enabled=self._amp_enabled,
            input_names=self.input_names,
            output_names=self.output_names,
            image_metrics=self._image_metrics,
            preview_sink=preview_sink,
        )
        has_components = bool(result.raw or result.weighted or result.current_weight)
        return MethodMetrics(
            losses={"loss_G": result.loss_G, "loss_D": result.loss_D},
            component_totals=(
                {"generator": result.loss_G, "discriminator": result.loss_D}
                if has_components
                else {}
            ),
            raw=dict(result.raw),
            weighted=dict(result.weighted),
            current_weight=dict(result.current_weight),
            image=dict(result.image),
        )

    def validation_metric(self, metrics: MethodMetrics, name: str) -> float | None:
        if name in metrics.image:
            return metrics.image[name]
        return loss_validation_metric(metrics, name)

    def checkpoint_selection_metrics(self, metrics: MethodMetrics) -> dict[str, float]:
        values = {"loss_G_val": metrics.losses["loss_G"]}
        values.update(
            {
                name: value
                for name, value in metrics.image.items()
                if name in self._checkpoint_metrics
            }
        )
        return {name: value for name, value in values.items() if math.isfinite(value)}

    def checkpoint_selection_modes(self) -> dict[str, str]:
        return dict(self._checkpoint_metrics)

    def step_schedulers(
        self,
        *,
        epoch: int,
        validation_metrics: MethodMetrics | None,
    ) -> bool:
        scheduler = self._optimization.scheduler
        monitor = scheduler.monitor
        return step_lr_schedulers(
            scheduler,
            (self._scheduler_G, self._scheduler_D),
            epoch=epoch,
            monitor_value=(
                None
                if validation_metrics is None or monitor is None
                else lambda: self.validation_metric(validation_metrics, monitor)
            ),
        )

    def learning_rates(self) -> Mapping[str, float]:
        return {
            "lr_g": float(self._opt_G.param_groups[0]["lr"]),
            "lr_d": float(self._opt_D.param_groups[0]["lr"]),
        }

    def checkpoint_identity(self) -> CheckpointIdentity:
        return self._identity

    def objective_metadata(self) -> dict[str, Any]:
        return self.loss_config.to_dict()

    def _models(self) -> dict[str, torch.nn.Module]:
        return {"generator": self.generator, "discriminator": self.discriminator}

    def _roles(self) -> dict[str, OptimizationRole]:
        return {
            "generator": OptimizationRole(self._opt_G, self._scaler_G, self._scheduler_G),
            "discriminator": OptimizationRole(self._opt_D, self._scaler_D, self._scheduler_D),
        }

    def state_dict(self) -> dict[str, Any]:
        return training_state_dict(
            self._optimization.scheduler, self._epochs, self._models(), self._roles()
        )

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Preflight every state group, then restore; nothing is mutated on rejection."""
        require_state_keys(state, TRAINING_STATE_KEYS, "state")
        check_training_state(
            state, self._optimization.scheduler, self._epochs, self._models(), self._roles()
        )
        with restoring_validated_state(self.name):
            load_training_state(state, self._models(), self._roles())
