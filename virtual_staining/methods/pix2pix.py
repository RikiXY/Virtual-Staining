from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch
import torch.optim as optim
from torch.amp import GradScaler

from virtual_staining.checkpoint_selection import (
    SUPPORTED_CHECKPOINT_METRICS,
    default_checkpoint_mode,
)
from virtual_staining.config.run import RunConfig
from virtual_staining.experiment.run_layout import RunLayout
from virtual_staining.models.factory import build_discriminator, build_generator
from virtual_staining.training.checkpoints import CheckpointManager
from virtual_staining.training.helpers import configured_loss_names, is_amp_enabled, unpack_batch
from virtual_staining.training.losses import ConfiguredLossEvaluator
from virtual_staining.training.runtime import MethodMetrics
from virtual_staining.training.steps import Pix2PixTrainingStep
from virtual_staining.training.validator import validate_epoch

if TYPE_CHECKING:
    from virtual_staining.training.benchmarking import TrainingBenchmarkRecorder

logger = logging.getLogger(__name__)

Scheduler = optim.lr_scheduler.LRScheduler | optim.lr_scheduler.ReduceLROnPlateau


class Pix2PixMethod:
    """Own the concrete paired Pix2Pix training topology."""

    name: str = "pix2pix"
    pairing: str = "paired"
    prediction_directions: tuple[str, ...] = ("forward",)
    default_checkpoint_metric: str = "loss_G_val"
    metric_names: tuple[str, ...] = ("loss_G", "loss_D")
    component_total_names: tuple[str, ...] = ("generator", "discriminator")

    def __init__(
        self,
        config: RunConfig,
        run_paths: RunLayout,
        device: torch.device,
        *,
        benchmark_recorder: TrainingBenchmarkRecorder | None = None,
    ) -> None:
        if config.training is None:
            raise ValueError("training config is required to construct Pix2Pix")
        self.config = config
        self.training = config.training
        self.device = device
        self._benchmark_recorder = benchmark_recorder
        self._amp_enabled = is_amp_enabled(device)
        self.loss_config = self.training.losses
        self.loss_names = tuple(configured_loss_names(self.loss_config))

        self.generator = build_generator(config.model).to(device)
        self.discriminator = build_discriminator(config.model).to(device)
        self._opt_G = optim.Adam(
            self.generator.parameters(),
            lr=self.training.lr_g,
            betas=(self.training.beta1, self.training.beta2),
        )
        self._opt_D = optim.Adam(
            self.discriminator.parameters(),
            lr=self.training.lr_d,
            betas=(self.training.beta1, self.training.beta2),
        )

        self._scaler_G = GradScaler(enabled=self._amp_enabled)
        self._scaler_D = GradScaler(enabled=self._amp_enabled)
        self._scheduler_G = self._build_scheduler(self._opt_G)
        self._scheduler_D = (
            self._build_scheduler(self._opt_D) if self.loss_config.active_discriminator else None
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
            generator_loss_terms=self.loss_config.generator,
            discriminator_loss_terms=self.loss_config.discriminator,
            benchmark_recorder=benchmark_recorder,
        )
        self._loss_evaluator = ConfiguredLossEvaluator(
            generator_terms=self.loss_config.generator,
            discriminator_terms=self.loss_config.discriminator,
        )
        self._checkpoint_manager = CheckpointManager(
            checkpoints_dir=run_paths.checkpoints_dir,
            generator=self.generator,
            discriminator=self.discriminator,
            opt_G=self._opt_G,
            opt_D=self._opt_D,
            scaler_G=self._scaler_G,
            scaler_D=self._scaler_D,
            scheduler_G=self._scheduler_G,
            scheduler_D=self._scheduler_D,
            image_size=config.project.image_size,
            device=device,
            lr_g=self.training.lr_g,
            lr_d=self.training.lr_d,
            beta1=self.training.beta1,
            beta2=self.training.beta2,
            batch_size=self.training.batch_size,
            num_workers=self.training.num_workers,
            target_modality=config.model.target,
        )

    def train_mode(self) -> None:
        self.generator.train()
        self.discriminator.train()

    def batch_size(self, batch: object) -> int:
        if not isinstance(batch, dict):
            return 0
        target = batch.get("target")
        if not isinstance(target, torch.Tensor) or target.ndim == 0:
            return 0
        return int(target.shape[0])

    def step(self, batch: object, *, epoch: int, global_step: int) -> MethodMetrics:
        recorder = self._benchmark_recorder
        if recorder is None:
            inputs, target, masks = self._unpack_batch(batch)
        else:
            with recorder.phase("h2d"):
                inputs, target, masks = self._unpack_batch(batch)
        result = self._step.step(
            inputs,
            target,
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
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict[str, torch.Tensor]]:
        input_names = cast(tuple[str, ...], self.generator.input_names)
        return unpack_batch(batch, self.device, input_names)

    def validate(
        self,
        loader: torch.utils.data.DataLoader,
        *,
        epoch: int,
        output_dir: Path,
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
            output_dir=output_dir,
            benchmark_recorder=self._benchmark_recorder,
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
        if name == "loss_G_val":
            return metrics.losses.get("loss_G")
        if name == "loss_D_val":
            return metrics.losses.get("loss_D")
        if name in metrics.image:
            return metrics.image[name]
        if name == "loss_val_total_generator":
            return metrics.component_totals.get("generator")
        if name == "loss_val_total_discriminator":
            return metrics.component_totals.get("discriminator")
        prefix_maps = (
            ("loss_val_raw_", metrics.raw),
            ("loss_val_weighted_", metrics.weighted),
            ("loss_val_current_weight_", metrics.current_weight),
        )
        for prefix, values in prefix_maps:
            if name.startswith(prefix):
                return values.get(name.removeprefix(prefix))
        return None

    def checkpoint_selection_metrics(self, metrics: MethodMetrics) -> dict[str, float]:
        values = {"loss_G_val": metrics.losses["loss_G"]}
        values.update(
            {
                name: value
                for name, value in metrics.image.items()
                if name in SUPPORTED_CHECKPOINT_METRICS
            }
        )
        return {name: value for name, value in values.items() if math.isfinite(value)}

    def checkpoint_selection_modes(self) -> dict[str, str]:
        return {metric: default_checkpoint_mode(metric) for metric in SUPPORTED_CHECKPOINT_METRICS}

    def step_schedulers(
        self,
        *,
        epoch: int,
        validation_metrics: MethodMetrics | None,
    ) -> bool:
        scheduler_config = self.training.scheduler
        if scheduler_config.name == "none":
            return False

        if scheduler_config.name == "linear_decay":
            for scheduler in (self._scheduler_G, self._scheduler_D):
                if scheduler is not None and not isinstance(
                    scheduler,
                    optim.lr_scheduler.ReduceLROnPlateau,
                ):
                    scheduler.step()
            return True

        if scheduler_config.name == "reduce_on_plateau":
            if validation_metrics is None:
                return False
            metric_value = self.validation_metric(
                validation_metrics,
                scheduler_config.monitor,
            )
            if metric_value is None or not math.isfinite(metric_value):
                logger.warning(
                    "Skipping learning-rate scheduler step at epoch %s because %s is unavailable",
                    epoch,
                    scheduler_config.monitor,
                )
                return False
            for scheduler in (self._scheduler_G, self._scheduler_D):
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(metric_value)
            return True

        raise AssertionError(f"Unsupported scheduler {scheduler_config.name!r}")

    def learning_rates(self) -> Mapping[str, float]:
        return {
            "lr_g": float(self._opt_G.param_groups[0]["lr"]),
            "lr_d": float(self._opt_D.param_groups[0]["lr"]),
        }

    def component_metadata(self) -> Mapping[str, object]:
        model_config = self.config.model.to_dict()
        return {
            "inputs": list(self.config.model.inputs),
            "outputs": [self.config.model.target],
            "generator": {
                "architecture": self.config.model.generator.architecture,
                "class": type(self.generator).__name__,
                "resolved": model_config["generator"],
            },
            "discriminator": {
                "architecture": "patchgan",
                "class": type(self.discriminator).__name__,
                "resolved": model_config["discriminator"],
            },
            "directions": list(self.prediction_directions),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "models": {
                "generator": self.generator.state_dict(),
                "discriminator": self.discriminator.state_dict(),
            },
            "optimizers": {
                "generator": self._opt_G.state_dict(),
                "discriminator": self._opt_D.state_dict(),
            },
            "scalers": {
                "generator": self._scaler_G.state_dict(),
                "discriminator": self._scaler_D.state_dict(),
            },
            "schedulers": {
                "generator": (
                    self._scheduler_G.state_dict() if self._scheduler_G is not None else None
                ),
                "discriminator": (
                    self._scheduler_D.state_dict() if self._scheduler_D is not None else None
                ),
            },
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        models = state["models"]
        optimizers = state["optimizers"]
        scalers = state.get("scalers", {})
        schedulers = state.get("schedulers", {})

        self.generator.load_state_dict(models["generator"])
        self.discriminator.load_state_dict(models["discriminator"])
        self._opt_G.load_state_dict(optimizers["generator"])
        self._opt_D.load_state_dict(optimizers["discriminator"])

        if "generator" in scalers:
            self._scaler_G.load_state_dict(scalers["generator"])
        if "discriminator" in scalers:
            self._scaler_D.load_state_dict(scalers["discriminator"])
        if self._scheduler_G is not None and schedulers.get("generator") is not None:
            self._scheduler_G.load_state_dict(schedulers["generator"])
        if self._scheduler_D is not None and schedulers.get("discriminator") is not None:
            self._scheduler_D.load_state_dict(schedulers["discriminator"])

    def save_checkpoint(self, epoch: int) -> Path:
        return self._checkpoint_manager.save(epoch)

    def load_checkpoint(self, path: Path) -> int:
        return self._checkpoint_manager.load(path)

    def latest_checkpoint(self) -> Path | None:
        return self._checkpoint_manager.latest()

    def _build_scheduler(self, optimizer: optim.Optimizer) -> Scheduler | None:
        scheduler_config = self.training.scheduler
        if scheduler_config.name == "none":
            return None
        if scheduler_config.name == "linear_decay":
            assert scheduler_config.decay_start_epoch is not None
            decay_start_epoch = scheduler_config.decay_start_epoch
            decay_span = max(1, self.training.epochs - decay_start_epoch)

            def lr_lambda(epoch: int) -> float:
                if epoch <= decay_start_epoch:
                    return 1.0
                return max(0.0, 1.0 - (epoch - decay_start_epoch) / decay_span)

            return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        if scheduler_config.name == "reduce_on_plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=scheduler_config.mode,
                factor=scheduler_config.factor,
                patience=scheduler_config.patience,
                min_lr=scheduler_config.min_lr,
            )
        raise AssertionError(f"Unsupported scheduler {scheduler_config.name!r}")
