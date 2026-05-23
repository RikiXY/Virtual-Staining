from __future__ import annotations

import csv
import datetime
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, TextIO

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast

from virtual_staining.experiment.run_paths import RunPaths
from virtual_staining.models.config import ModelConfig
from virtual_staining.training.checkpoints import CheckpointManager
from virtual_staining.training.config import (
    SUPPORTED_CHECKPOINT_METRICS,
    LossConfig,
    TrainingConfig,
    default_checkpoint_mode,
)
from virtual_staining.training.helpers import (
    LossComponentAccumulator,
    component_metric_row,
    configured_loss_names,
    dataset_len,
    is_amp_enabled,
    metrics_fieldnames,
    save_images,
    unpack_batch,
)
from virtual_staining.training.logging import (
    ProgressTracker,
    TrainingLogSession,
    emit_progress_update,
    finish_console_progress,
    format_duration,
)
from virtual_staining.training.losses import (
    ConfiguredLossEvaluator,
    LossEvaluationContext,
    StepLosses,
)
from virtual_staining.training.results import EpochMetrics, TrainingResult
from virtual_staining.training.steps import Pix2PixTrainingStep
from virtual_staining.training.validation_metrics import (
    VALIDATION_IMAGE_METRIC_NAMES,
    ValidationImageMetricAccumulator,
)

if TYPE_CHECKING:
    from virtual_staining.reporting.base import TrainingReporter

logger = logging.getLogger(__name__)
checkpoint_logger = logging.getLogger("virtual_staining.training.checkpoints")


@dataclass
class _TrainingStatus:
    last_checkpoint: str
    best_checkpoint: str = "none"
    best_loss_G_val: float | None = None
    latest_eval_losses: StepLosses | None = None
    latest_eval_epoch: int | None = None


@dataclass
class _BestCheckpointState:
    path: Path | None = None


@dataclass
class _EarlyStoppingState:
    best_value: float | None = None
    best_epoch: int | None = None
    stale_count: int = 0
    stopped: bool = False
    stop_epoch: int | None = None
    stop_reason: str | None = None


@dataclass(frozen=True)
class _EpochResult:
    metrics: EpochMetrics
    stopped_early: bool = False


@dataclass(frozen=True)
class _TrainingLoopResult:
    final_epoch: int
    stopped_early: bool
    stop_reason: str | None
    early_stopping_state: _EarlyStoppingState | None


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class Trainer:
    """
    Orchestrates a Pix2Pix training run.

    Owns the training loop, validation, checkpoint save/load, progress
    reporting, and metric logging. Models and data loaders are constructed
    outside and injected - Trainer does not hardcode architecture choices.
    """

    def __init__(
        self,
        config: TrainingConfig,
        model_config: ModelConfig,
        run_paths: RunPaths,
        generator: nn.Module,
        discriminator: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        device: torch.device,
        *,
        image_size: tuple[int, int],
        train_dir: Path,
        val_dir: Path,
        losses: LossConfig | None = None,
    ) -> None:
        self.config = config
        self.model_config = model_config
        self._run_paths = run_paths
        self.generator = generator
        self.discriminator = discriminator
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self._image_size = image_size
        self._train_dir = train_dir
        self._val_dir = val_dir
        self.losses = losses

        self._amp_enabled = is_amp_enabled(device)

        self._opt_G = optim.Adam(
            generator.parameters(),
            lr=config.lr_g,
            betas=(config.beta1, config.beta2),
        )
        self._opt_D = optim.Adam(
            discriminator.parameters(),
            lr=config.lr_d,
            betas=(config.beta1, config.beta2),
        )
        self._scaler_G = GradScaler(enabled=self._amp_enabled)
        self._scaler_D = GradScaler(enabled=self._amp_enabled)
        generator_loss_terms = losses.generator if losses is not None else ()
        discriminator_loss_terms = losses.discriminator if losses is not None else ()
        self._scheduler_G = self._build_scheduler(self._opt_G)
        self._scheduler_D = (
            self._build_scheduler(self._opt_D) if self._has_active_discriminator() else None
        )
        self._loss_evaluator = ConfiguredLossEvaluator(
            generator_terms=generator_loss_terms,
            discriminator_terms=discriminator_loss_terms,
        )
        self._step = Pix2PixTrainingStep(
            generator=generator,
            discriminator=discriminator,
            opt_G=self._opt_G,
            opt_D=self._opt_D,
            scaler_G=self._scaler_G,
            scaler_D=self._scaler_D,
            device=device,
            amp_enabled=self._amp_enabled,
            generator_loss_terms=generator_loss_terms,
            discriminator_loss_terms=discriminator_loss_terms,
        )

        self._logs_dir = run_paths.logs_dir
        self._checkpoints_dir = run_paths.checkpoints_dir
        self._output_val_dir = run_paths.output_val_dir
        self._output_train_dir = run_paths.output_train_dir
        self._checkpoint_manager = CheckpointManager(
            checkpoints_dir=self._checkpoints_dir,
            generator=generator,
            discriminator=discriminator,
            opt_G=self._opt_G,
            opt_D=self._opt_D,
            scaler_G=self._scaler_G,
            scaler_D=self._scaler_D,
            scheduler_G=self._scheduler_G,
            scheduler_D=self._scheduler_D,
            image_size=image_size,
            device=device,
            model_name=model_config.name,
            lr_g=config.lr_g,
            lr_d=config.lr_d,
            beta1=config.beta1,
            beta2=config.beta2,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(
        self,
        seed: int,
        start_epoch: int = 0,
        reporter: TrainingReporter | None = None,
    ) -> TrainingResult:
        """Run the full training loop."""
        start_time = time.time()
        self._prepare_run_directories()

        log_file = self._training_log_path()
        with TrainingLogSession(log_file, logger, checkpoint_logger):
            self._clear_training_outputs()
            self._log_training_start(seed, start_epoch, log_file)

            progress_tracker = self._start_progress_tracker(start_epoch)
            training_status = self._initial_training_status()
            best_state = self._load_resumed_best_checkpoint(start_epoch, training_status)

            loop_result = self._run_training_epochs(
                start_epoch=start_epoch,
                progress_tracker=progress_tracker,
                training_status=training_status,
                best_state=best_state,
                start_time=start_time,
                reporter=reporter,
            )

            if best_state.path is None:
                best_state.path = self._checkpoint_manager.latest()
                if best_state.path is not None:
                    training_status.best_checkpoint = best_state.path.name

            finish_console_progress()
            total_seconds = time.time() - start_time
            logger.info("Execution completed. Total time = %.2f seconds", total_seconds)
            early_state = loop_result.early_stopping_state
            return TrainingResult(
                final_epoch=loop_result.final_epoch,
                best_checkpoint_path=best_state.path,
                stopped_early=loop_result.stopped_early,
                stop_epoch=early_state.stop_epoch if early_state is not None else None,
                stop_reason=loop_result.stop_reason,
                early_stopping_monitor=(
                    self.config.early_stopping.monitor
                    if self.config.early_stopping is not None
                    else None
                ),
                early_stopping_mode=(
                    self.config.early_stopping.mode
                    if self.config.early_stopping is not None
                    else None
                ),
                early_stopping_best_epoch=(
                    early_state.best_epoch if early_state is not None else None
                ),
                early_stopping_best_value=(
                    early_state.best_value if early_state is not None else None
                ),
            )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _prepare_run_directories(self) -> None:
        for directory in [
            self._logs_dir,
            self._checkpoints_dir,
            self._output_val_dir,
            self._output_train_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

    def _training_log_path(self) -> Path:
        return self._logs_dir / "training.log"

    def _clear_training_outputs(self) -> None:
        for output_file in self._output_train_dir.iterdir():
            if output_file.is_file():
                output_file.unlink()

    def _log_training_start(self, seed: int, start_epoch: int, log_file: Path) -> None:
        device_name = (
            torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else "CPU"
        )
        logger.debug("Seed set to %s", seed)
        logger.info("Device: %s (%s)", self.device, device_name)

        if start_epoch > 0:
            logger.info("Training resumed from epoch %s", start_epoch)
        else:
            logger.debug("Training started from scratch")

        logger.info("=== Pix2Pix training ===")
        logger.info("Run root: %s", self._run_paths.root)
        logger.info("Train dir: %s", self._train_dir)
        logger.info("Validation dir: %s", self._val_dir)
        logger.info("Device: %s", self.device)
        logger.info("Epochs: %s", self.config.epochs)
        logger.info("Start epoch: %s", start_epoch)
        logger.info("Train samples: %s", dataset_len(self.train_loader))
        logger.info("Validation samples: %s", dataset_len(self.val_loader))
        logger.info("Train batches/epoch: %s", len(self.train_loader))
        logger.info("Validation batches: %s", len(self.val_loader))
        logger.info("Detailed log: %s", log_file)
        logger.info(
            "Hyperparameters | lr_g=%s | lr_d=%s | beta1=%s | beta2=%s | scheduler=%s",
            self.config.lr_g,
            self.config.lr_d,
            self.config.beta1,
            self.config.beta2,
            self.config.scheduler.to_yaml_dict(),
        )
        logger.info("Training started")

    def _build_scheduler(
        self,
        optimizer: optim.Optimizer,
    ) -> optim.lr_scheduler.LRScheduler | optim.lr_scheduler.ReduceLROnPlateau | None:
        scheduler_config = self.config.scheduler
        if scheduler_config.name == "none":
            return None
        if scheduler_config.name == "linear_decay":
            assert scheduler_config.decay_start_epoch is not None
            decay_start_epoch = scheduler_config.decay_start_epoch
            decay_span = max(1, self.config.epochs - decay_start_epoch)

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

    def _has_active_discriminator(self) -> bool:
        if self.losses is None:
            return True
        return bool(self.losses.active_discriminator)

    def _start_progress_tracker(self, start_epoch: int) -> ProgressTracker:
        progress_tracker = ProgressTracker(
            total_epochs=self.config.epochs,
            total_batches=len(self.train_loader),
            start_epoch=start_epoch,
            warmup_batches=max(10, self.config.log_rate),
        )
        progress_tracker.start()
        return progress_tracker

    def _initial_training_status(self) -> _TrainingStatus:
        last_checkpoint = Path(self.config.resume).name if self.config.resume else "none"
        return _TrainingStatus(last_checkpoint=last_checkpoint)

    def _load_resumed_best_checkpoint(
        self,
        start_epoch: int,
        training_status: _TrainingStatus,
    ) -> _BestCheckpointState:
        del start_epoch, training_status
        return _BestCheckpointState()

    def _run_training_epochs(
        self,
        *,
        start_epoch: int,
        progress_tracker: ProgressTracker,
        training_status: _TrainingStatus,
        best_state: _BestCheckpointState,
        start_time: float,
        reporter: TrainingReporter | None,
    ) -> _TrainingLoopResult:
        loss_names = configured_loss_names(self.losses)
        train_metrics_path = self._run_paths.metrics_dir / "train.csv"
        validation_metrics_path = self._run_paths.metrics_dir / "validation.csv"
        all_metrics_path = self._run_paths.metrics_dir / "all.csv"
        last_epoch_metrics: EpochMetrics | None = None
        final_epoch = max(start_epoch, self.config.epochs) - 1
        early_stopping_state = (
            _EarlyStoppingState() if self.config.early_stopping is not None else None
        )

        with (
            open(train_metrics_path, "w", newline="", encoding="utf-8") as train_metrics_file,
            open(
                validation_metrics_path, "w", newline="", encoding="utf-8"
            ) as validation_metrics_file,
            open(all_metrics_path, "w", newline="", encoding="utf-8") as all_metrics_file,
        ):
            train_metrics_writer = csv.DictWriter(
                train_metrics_file,
                fieldnames=metrics_fieldnames(loss_names, stage="train"),
            )
            validation_metrics_writer = csv.DictWriter(
                validation_metrics_file,
                fieldnames=metrics_fieldnames(loss_names, stage="val")
                + VALIDATION_IMAGE_METRIC_NAMES,
            )
            all_metrics_writer = csv.DictWriter(
                all_metrics_file,
                fieldnames=metrics_fieldnames(loss_names) + VALIDATION_IMAGE_METRIC_NAMES,
            )
            train_metrics_writer.writeheader()
            validation_metrics_writer.writeheader()
            all_metrics_writer.writeheader()

            for epoch in range(start_epoch, self.config.epochs):
                epoch_result = self._run_training_epoch(
                    epoch=epoch,
                    train_metrics_writer=train_metrics_writer,
                    validation_metrics_writer=validation_metrics_writer,
                    all_metrics_writer=all_metrics_writer,
                    train_metrics_file=train_metrics_file,
                    validation_metrics_file=validation_metrics_file,
                    all_metrics_file=all_metrics_file,
                    progress_tracker=progress_tracker,
                    training_status=training_status,
                    best_state=best_state,
                    start_time=start_time,
                    reporter=reporter,
                    early_stopping_state=early_stopping_state,
                )
                last_epoch_metrics = epoch_result.metrics
                final_epoch = epoch
                if epoch_result.stopped_early:
                    break

        self._save_final_checkpoint_if_needed(
            start_epoch=start_epoch,
            final_epoch=final_epoch,
            final_metrics=last_epoch_metrics,
            progress_tracker=progress_tracker,
            training_status=training_status,
            best_state=best_state,
            start_time=start_time,
            reporter=reporter,
        )
        return _TrainingLoopResult(
            final_epoch=final_epoch,
            stopped_early=bool(early_stopping_state and early_stopping_state.stopped),
            stop_reason=early_stopping_state.stop_reason if early_stopping_state else None,
            early_stopping_state=early_stopping_state,
        )

    def _run_training_epoch(
        self,
        *,
        epoch: int,
        train_metrics_writer: csv.DictWriter,
        validation_metrics_writer: csv.DictWriter,
        all_metrics_writer: csv.DictWriter,
        train_metrics_file: TextIO,
        validation_metrics_file: TextIO,
        all_metrics_file: TextIO,
        progress_tracker: ProgressTracker,
        training_status: _TrainingStatus,
        best_state: _BestCheckpointState,
        start_time: float,
        reporter: TrainingReporter | None,
        early_stopping_state: _EarlyStoppingState | None,
    ) -> _EpochResult:
        logger.debug("Starting epoch %s", epoch)
        epoch_metrics = self._train_epoch(epoch, progress_tracker, training_status)
        logger.debug("Finished epoch %s", epoch)

        val_metrics, validation_checkpoint_path = self._validate_and_update_best(
            epoch=epoch,
            epoch_metrics=epoch_metrics,
            progress_tracker=progress_tracker,
            training_status=training_status,
            best_state=best_state,
            start_time=start_time,
            reporter=reporter,
        )
        if val_metrics is None:
            self._step_lr_schedulers(epoch=epoch, val_metrics=None)

        self._save_scheduled_checkpoint(
            epoch=epoch,
            epoch_metrics=epoch_metrics,
            progress_tracker=progress_tracker,
            training_status=training_status,
            start_time=start_time,
            reporter=reporter,
            existing_checkpoint_path=validation_checkpoint_path,
        )

        self._write_train_metrics(train_metrics_writer, epoch, epoch_metrics)
        train_metrics_file.flush()
        if val_metrics is not None:
            self._write_validation_metrics(validation_metrics_writer, epoch, val_metrics)
            validation_metrics_file.flush()
        self._write_all_metrics(all_metrics_writer, epoch, epoch_metrics, val_metrics)
        all_metrics_file.flush()
        if reporter is not None:
            reporter.on_epoch_completed(epoch_metrics)
        stopped_early = False
        if val_metrics is not None and early_stopping_state is not None:
            stopped_early = self._update_early_stopping(
                epoch=epoch,
                val_metrics=val_metrics,
                state=early_stopping_state,
            )
        return _EpochResult(metrics=epoch_metrics, stopped_early=stopped_early)

    def _save_scheduled_checkpoint(
        self,
        *,
        epoch: int,
        epoch_metrics: EpochMetrics,
        progress_tracker: ProgressTracker,
        training_status: _TrainingStatus,
        start_time: float,
        reporter: TrainingReporter | None,
        existing_checkpoint_path: Path | None = None,
    ) -> Path | None:
        if (epoch + 1) % self.config.checkpoint_rate != 0:
            return None

        checkpoint_path = existing_checkpoint_path
        if checkpoint_path is None:
            checkpoint_path = self._checkpoint_manager.save(epoch)
            training_status.last_checkpoint = checkpoint_path.name
            logger.info("Checkpoint saved to %s at epoch %s", checkpoint_path, epoch)
            if reporter is not None:
                reporter.on_checkpoint_saved(checkpoint_path, epoch)
        self._emit_epoch_progress(
            epoch=epoch,
            epoch_metrics=epoch_metrics,
            progress_tracker=progress_tracker,
            training_status=training_status,
            start_time=start_time,
            eta_str="0s" if epoch == self.config.epochs - 1 else "--",
        )
        return checkpoint_path

    def _validate_and_update_best(
        self,
        *,
        epoch: int,
        epoch_metrics: EpochMetrics,
        progress_tracker: ProgressTracker,
        training_status: _TrainingStatus,
        best_state: _BestCheckpointState,
        start_time: float,
        reporter: TrainingReporter | None,
    ) -> tuple[EpochMetrics | None, Path | None]:
        if (epoch + 1) % self.config.validate_rate != 0:
            return None, None

        val_metrics = self._validate(epoch)
        training_status.latest_eval_losses = StepLosses(
            loss_G=val_metrics.loss_G,
            loss_D=val_metrics.loss_D,
        )
        training_status.latest_eval_epoch = epoch
        self._step_lr_schedulers(epoch=epoch, val_metrics=val_metrics)
        checkpoint_metrics = self._checkpoint_selection_metrics(val_metrics)
        ranked_checkpoint_path: Path | None = None
        if not checkpoint_metrics:
            logger.warning("Skipping checkpoint ranking update because all metrics are non-finite")
        else:
            ranked_checkpoint_path = self._ensure_best_checkpoint_path(
                epoch=epoch,
                checkpoint_path=None,
                training_status=training_status,
                reporter=reporter,
            )
            config_hash = self._read_config_hash()
            loss_config = self.losses.to_yaml_dict() if self.losses is not None else None
            checkpoint_modes = self._checkpoint_selection_modes()
            self._checkpoint_manager.update_selection_records(
                metrics=checkpoint_metrics,
                modes=checkpoint_modes,
                top_k=self.config.checkpoint_top_k,
                epoch=epoch,
                checkpoint_path=ranked_checkpoint_path,
                config_hash=config_hash,
                loss_config=loss_config,
            )
        self._emit_epoch_progress(
            epoch=epoch,
            epoch_metrics=epoch_metrics,
            progress_tracker=progress_tracker,
            training_status=training_status,
            start_time=start_time,
            eta_str="0s" if epoch == self.config.epochs - 1 else "--",
        )
        return val_metrics, ranked_checkpoint_path

    def _step_lr_schedulers(
        self,
        *,
        epoch: int,
        val_metrics: EpochMetrics | None,
    ) -> None:
        scheduler_config = self.config.scheduler
        if scheduler_config.name == "none":
            return

        if scheduler_config.name == "linear_decay":
            for scheduler in (self._scheduler_G, self._scheduler_D):
                if scheduler is not None and not isinstance(
                    scheduler,
                    optim.lr_scheduler.ReduceLROnPlateau,
                ):
                    scheduler.step()
            self._log_learning_rates(epoch)
            return

        if scheduler_config.name == "reduce_on_plateau":
            if val_metrics is None:
                return
            metric_value = self._scheduler_monitor_value(val_metrics)
            if metric_value is None or not math.isfinite(metric_value):
                logger.warning(
                    "Skipping learning-rate scheduler step at epoch %s because %s is unavailable",
                    epoch,
                    scheduler_config.monitor,
                )
                return
            for scheduler in (self._scheduler_G, self._scheduler_D):
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(metric_value)
            self._log_learning_rates(epoch)
            return

        raise AssertionError(f"Unsupported scheduler {scheduler_config.name!r}")

    def _scheduler_monitor_value(self, val_metrics: EpochMetrics) -> float | None:
        monitor = self.config.scheduler.monitor
        if monitor == "loss_G_val":
            return val_metrics.loss_G
        return val_metrics.image.get(monitor)

    def _update_early_stopping(
        self,
        *,
        epoch: int,
        val_metrics: EpochMetrics,
        state: _EarlyStoppingState,
    ) -> bool:
        early_config = self.config.early_stopping
        if early_config is None:
            return False

        metric_value = self._early_stopping_monitor_value(val_metrics)
        if metric_value is None or not math.isfinite(metric_value):
            logger.warning(
                "Skipping early-stopping update at epoch %s because %s is unavailable",
                epoch,
                early_config.monitor,
            )
            return False

        if self._is_early_stopping_improvement(metric_value, state.best_value):
            state.best_value = metric_value
            state.best_epoch = epoch
            state.stale_count = 0
            return False

        state.stale_count += 1
        if state.stale_count >= early_config.patience:
            state.stopped = True
            state.stop_epoch = epoch
            state.stop_reason = (
                f"early_stopping: {early_config.monitor} did not improve by at least "
                f"{early_config.min_delta:g} for {early_config.patience} validation event(s); "
                f"best epoch {state.best_epoch} value {state.best_value:.6g}"
            )
            logger.info("Stopping early at epoch %s: %s", epoch, state.stop_reason)
            return True
        return False

    def _early_stopping_monitor_value(self, val_metrics: EpochMetrics) -> float | None:
        early_config = self.config.early_stopping
        if early_config is None:
            return None
        monitor = early_config.monitor
        if monitor == "loss_G_val":
            return val_metrics.loss_G
        if monitor == "loss_D_val":
            return val_metrics.loss_D
        if monitor in val_metrics.image:
            return val_metrics.image[monitor]
        if monitor == "loss_val_total_generator":
            return val_metrics.loss_G
        if monitor == "loss_val_total_discriminator":
            return val_metrics.loss_D
        prefix_maps = (
            ("loss_val_raw_", val_metrics.raw),
            ("loss_val_weighted_", val_metrics.weighted),
            ("loss_val_current_weight_", val_metrics.current_weight),
        )
        for prefix, values in prefix_maps:
            if monitor.startswith(prefix):
                return values.get(monitor.removeprefix(prefix))
        return None

    def _is_early_stopping_improvement(
        self,
        value: float,
        best_value: float | None,
    ) -> bool:
        early_config = self.config.early_stopping
        if early_config is None or best_value is None:
            return True
        if early_config.mode == "max":
            return value > best_value + early_config.min_delta
        return value < best_value - early_config.min_delta

    def _log_learning_rates(self, epoch: int) -> None:
        logger.info(
            "Learning rates | epoch=%s | lr_g=%s | lr_d=%s",
            epoch,
            self._optimizer_lr(self._opt_G),
            self._optimizer_lr(self._opt_D),
        )

    @staticmethod
    def _optimizer_lr(optimizer: optim.Optimizer) -> float:
        return float(optimizer.param_groups[0]["lr"])

    def _checkpoint_selection_metrics(self, val_metrics: EpochMetrics) -> dict[str, float]:
        metrics = {"loss_G_val": val_metrics.loss_G}
        metrics.update(
            {
                metric: value
                for metric, value in val_metrics.image.items()
                if metric in SUPPORTED_CHECKPOINT_METRICS
            }
        )
        return {metric: value for metric, value in metrics.items() if math.isfinite(value)}

    def _checkpoint_selection_modes(self) -> dict[str, str]:
        return {metric: default_checkpoint_mode(metric) for metric in SUPPORTED_CHECKPOINT_METRICS}

    def _read_config_hash(self) -> str | None:
        if not self._run_paths.config_hash.exists():
            return None
        value = self._run_paths.config_hash.read_text(encoding="utf-8").strip()
        return value or None

    def _ensure_best_checkpoint_path(
        self,
        *,
        epoch: int,
        checkpoint_path: Path | None,
        training_status: _TrainingStatus,
        reporter: TrainingReporter | None,
    ) -> Path:
        if checkpoint_path is not None:
            return checkpoint_path

        checkpoint_path = self._checkpoint_manager.save(epoch)
        training_status.last_checkpoint = checkpoint_path.name
        logger.info(
            "Checkpoint saved to %s at epoch %s for checkpoint selection",
            checkpoint_path,
            epoch,
        )
        if reporter is not None:
            reporter.on_checkpoint_saved(checkpoint_path, epoch)
        return checkpoint_path

    def _write_train_metrics(
        self,
        metrics_writer: csv.DictWriter,
        epoch: int,
        epoch_metrics: EpochMetrics,
    ) -> None:
        row = {
            "epoch": epoch,
            "loss_G_train": f"{epoch_metrics.loss_G:.6f}",
            "loss_D_train": f"{epoch_metrics.loss_D:.6f}",
        }
        row.update(component_metric_row("train", epoch_metrics))
        metrics_writer.writerow(row)

    def _write_validation_metrics(
        self,
        metrics_writer: csv.DictWriter,
        epoch: int,
        val_metrics: EpochMetrics,
    ) -> None:
        row = {
            "epoch": epoch,
            "loss_G_val": f"{val_metrics.loss_G:.6f}",
            "loss_D_val": f"{val_metrics.loss_D:.6f}",
        }
        row.update(component_metric_row("val", val_metrics))
        for name in VALIDATION_IMAGE_METRIC_NAMES:
            value = val_metrics.image.get(name, float("nan"))
            row[name] = f"{value:.6f}" if math.isfinite(value) else ""
        metrics_writer.writerow(row)

    def _write_all_metrics(
        self,
        metrics_writer: csv.DictWriter,
        epoch: int,
        epoch_metrics: EpochMetrics,
        val_metrics: EpochMetrics | None,
    ) -> None:
        row = {
            "epoch": epoch,
            "loss_G_train": f"{epoch_metrics.loss_G:.6f}",
            "loss_D_train": f"{epoch_metrics.loss_D:.6f}",
            "loss_G_val": f"{val_metrics.loss_G:.6f}" if val_metrics else "",
            "loss_D_val": f"{val_metrics.loss_D:.6f}" if val_metrics else "",
        }
        row.update(component_metric_row("train", epoch_metrics))
        row.update(component_metric_row("val", val_metrics))
        if val_metrics is not None:
            for name in VALIDATION_IMAGE_METRIC_NAMES:
                value = val_metrics.image.get(name, float("nan"))
                row[name] = f"{value:.6f}" if math.isfinite(value) else ""
        metrics_writer.writerow(row)

    def _save_final_checkpoint_if_needed(
        self,
        *,
        start_epoch: int,
        final_epoch: int,
        final_metrics: EpochMetrics | None,
        progress_tracker: ProgressTracker,
        training_status: _TrainingStatus,
        best_state: _BestCheckpointState,
        start_time: float,
        reporter: TrainingReporter | None,
    ) -> None:
        if start_epoch >= self.config.epochs or final_metrics is None:
            return

        if (final_epoch + 1) % self.config.checkpoint_rate == 0:
            return

        checkpoint_path = self._checkpoint_manager.save(final_epoch)
        training_status.last_checkpoint = checkpoint_path.name
        logger.info("Final checkpoint saved to %s (epoch %s)", checkpoint_path, final_epoch)
        if reporter is not None:
            reporter.on_checkpoint_saved(checkpoint_path, final_epoch)
        if best_state.path is None:
            best_state.path = checkpoint_path
            training_status.best_checkpoint = checkpoint_path.name
        self._emit_epoch_progress(
            epoch=final_epoch,
            epoch_metrics=final_metrics,
            progress_tracker=progress_tracker,
            training_status=training_status,
            start_time=start_time,
            progress=1.0,
            eta_str="0s",
        )

    def _emit_epoch_progress(
        self,
        *,
        epoch: int,
        epoch_metrics: EpochMetrics,
        progress_tracker: ProgressTracker,
        training_status: _TrainingStatus,
        start_time: float,
        eta_str: str,
        progress: float | None = None,
    ) -> None:
        emit_progress_update(
            progress=(
                progress if progress is not None else (epoch + 1) / progress_tracker.total_epochs
            ),
            epoch_progress=1.0,
            epoch=epoch,
            batch_index=len(self.train_loader) - 1,
            progress_tracker=progress_tracker,
            step_losses=StepLosses(
                loss_G=epoch_metrics.loss_G,
                loss_D=epoch_metrics.loss_D,
            ),
            elapsed_str=format_duration(time.time() - start_time),
            eta_str=eta_str,
            end_time_str=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            last_checkpoint_name=training_status.last_checkpoint,
            best_checkpoint_name=training_status.best_checkpoint,
            best_checkpoint_loss_G_val=training_status.best_loss_G_val,
            eval_losses=training_status.latest_eval_losses,
            eval_epoch=training_status.latest_eval_epoch,
        )

    def _train_epoch(
        self,
        epoch: int,
        progress_tracker: ProgressTracker,
        training_status: _TrainingStatus,
    ) -> EpochMetrics:
        self.generator.train()
        self.discriminator.train()

        total_loss_G = 0.0
        total_loss_D = 0.0
        component_totals = LossComponentAccumulator(configured_loss_names(self.losses))
        num_batches = 0

        for i, batch in enumerate(self.train_loader):
            x, y, masks = unpack_batch(batch, self.device)
            step_losses = self._step.step(
                x,
                y,
                epoch=epoch,
                global_step=epoch * len(self.train_loader) + i,
                masks=masks,
            )
            total_loss_G += step_losses.loss_G
            total_loss_D += step_losses.loss_D
            component_totals.add(
                raw=step_losses.raw,
                weighted=step_losses.weighted,
                current_weight=step_losses.current_weight,
            )
            num_batches += 1

            progress, elapsed, eta, end_time = progress_tracker.calculate_progress(epoch, i)
            elapsed_str = format_duration(elapsed)
            eta_str = format_duration(eta)
            epoch_progress = (i + 1) / progress_tracker.total_batches
            end_time_str = (
                "warming up"
                if end_time is None
                else datetime.datetime.fromtimestamp(end_time).strftime("%Y-%m-%d %H:%M:%S")
            )

            should_update_progress = (
                i % self.config.log_rate == 0 or i == len(self.train_loader) - 1
            )
            if should_update_progress:
                emit_progress_update(
                    progress=progress,
                    epoch_progress=epoch_progress,
                    epoch=epoch,
                    batch_index=i,
                    progress_tracker=progress_tracker,
                    step_losses=step_losses,
                    elapsed_str=elapsed_str,
                    eta_str=eta_str,
                    end_time_str=end_time_str,
                    last_checkpoint_name=training_status.last_checkpoint,
                    best_checkpoint_name=training_status.best_checkpoint,
                    best_checkpoint_loss_G_val=training_status.best_loss_G_val,
                    eval_losses=training_status.latest_eval_losses,
                    eval_epoch=training_status.latest_eval_epoch,
                )

        if num_batches == 0:
            raise RuntimeError("Training loader was empty; cannot compute epoch metrics.")
        component_averages = component_totals.average(num_batches)
        return EpochMetrics(
            loss_G=total_loss_G / num_batches,
            loss_D=total_loss_D / num_batches,
            raw=component_averages.raw,
            weighted=component_averages.weighted,
            current_weight=component_averages.current_weight,
        )

    def _validate(self, epoch: int) -> EpochMetrics:
        generator_was_training = self.generator.training
        discriminator_was_training = self.discriminator.training
        self.generator.eval()
        self.discriminator.eval()

        try:
            self._output_val_dir.mkdir(parents=True, exist_ok=True)

            total_loss_G = 0.0
            total_loss_D = 0.0
            component_totals = LossComponentAccumulator(configured_loss_names(self.losses))
            image_metric_totals = ValidationImageMetricAccumulator()
            count = 0

            with torch.no_grad():
                for i, batch in enumerate(self.val_loader):
                    x, y, masks = unpack_batch(batch, self.device)

                    with autocast(device_type=self.device.type, enabled=self._amp_enabled):
                        fake = self.generator(x)
                        context = LossEvaluationContext(epoch=epoch, masks=masks)
                        D_fake: torch.Tensor | None = None
                        if self._validation_needs_discriminator_logits():
                            D_real = self.discriminator(x, y)
                            D_fake_logits = self.discriminator(x, fake)
                            D_fake = D_fake_logits
                            discriminator_loss = self._loss_evaluator.discriminator_total(
                                discriminator_real=D_real,
                                discriminator_fake=D_fake_logits,
                                context=context,
                            )
                        else:
                            discriminator_loss = None
                        generator_loss = self._loss_evaluator.generator_total(
                            prediction=fake,
                            target=y,
                            discriminator_fake=D_fake,
                            context=context,
                        )
                        loss_D = (
                            discriminator_loss.total
                            if discriminator_loss is not None
                            else fake.sum() * 0.0
                        )
                        loss_G = generator_loss.total
                        if discriminator_loss is not None:
                            component_totals.add(
                                raw=discriminator_loss.raw,
                                weighted=discriminator_loss.weighted,
                                current_weight=discriminator_loss.current_weight,
                            )
                        component_totals.add(
                            raw=generator_loss.raw,
                            weighted=generator_loss.weighted,
                            current_weight=generator_loss.current_weight,
                        )

                    total_loss_D += loss_D.item()
                    total_loss_G += loss_G.item()
                    image_metric_totals.add_batch(fake, y)
                    count += 1

                    if i < 5:
                        save_images(self._output_val_dir, x[0], fake[0], y[0], epoch, i)

            avg_loss_G = total_loss_G / count if count > 0 else 0.0
            avg_loss_D = total_loss_D / count if count > 0 else 0.0
            component_averages = component_totals.average(count)

            logger.info(
                "[Epoch %s] Validation: loss_G=%.4f loss_D=%.4f",
                epoch,
                avg_loss_G,
                avg_loss_D,
            )

            return EpochMetrics(
                loss_G=avg_loss_G,
                loss_D=avg_loss_D,
                raw=component_averages.raw,
                weighted=component_averages.weighted,
                current_weight=component_averages.current_weight,
                image=image_metric_totals.mean(),
            )
        finally:
            if generator_was_training:
                self.generator.train()
            if discriminator_was_training:
                self.discriminator.train()

    def _validation_needs_discriminator_logits(self) -> bool:
        if self.losses is None:
            return False
        terms = (*self.losses.generator, *self.losses.discriminator)
        return any(term.name == "adversarial_bce" for term in terms)
