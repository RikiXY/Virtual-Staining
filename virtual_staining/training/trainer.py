from __future__ import annotations

import datetime
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler

from virtual_staining.experiment.run_paths import RunPaths
from virtual_staining.training.checkpoint_selection import (
    load_best_checkpoint_record,
    update_checkpoint_selection,
)
from virtual_staining.training.checkpoints import CheckpointManager
from virtual_staining.training.config import (
    SUPPORTED_CHECKPOINT_METRICS,
    TrainingConfig,
    default_checkpoint_mode,
)
from virtual_staining.training.helpers import (
    LossComponentAccumulator,
    configured_loss_names,
    dataset_len,
    is_amp_enabled,
    unpack_batch,
)
from virtual_staining.training.history import TrainingHistory
from virtual_staining.training.loss_config import LossConfig
from virtual_staining.training.losses import ConfiguredLossEvaluator, StepLosses
from virtual_staining.training.progress import (
    ProgressTracker,
    TrainingLogSession,
    emit_progress_update,
    finish_console_progress,
    format_duration,
)
from virtual_staining.training.results import EpochMetrics, TrainingResult
from virtual_staining.training.steps import Pix2PixTrainingStep
from virtual_staining.training.validator import validate_epoch

logger = logging.getLogger(__name__)
checkpoint_logger = logging.getLogger("virtual_staining.training.checkpoints")


@dataclass
class _TrainingSession:
    start_epoch: int
    start_time: float
    progress_tracker: ProgressTracker
    history: TrainingHistory
    last_checkpoint: str
    best_checkpoint: str = "none"
    best_checkpoint_path: Path | None = None
    best_loss_G_val: float | None = None
    latest_eval_losses: StepLosses | None = None
    latest_eval_epoch: int | None = None
    early_stopping_best_value: float | None = None
    early_stopping_best_epoch: int | None = None
    early_stopping_stale_count: int = 0
    stopped: bool = False
    stop_epoch: int | None = None
    stop_reason: str | None = None
    final_epoch: int = -1
    final_metrics: EpochMetrics | None = None


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class Trainer:
    """
    Orchestrates a Pix2Pix training run.

    Owns the training loop, validation, checkpoint save/load, progress
    progress display and metric logging. Models and data loaders are constructed
    outside and injected - Trainer does not hardcode architecture choices.
    """

    def __init__(
        self,
        config: TrainingConfig,
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
        target_modality: str | None = None,
    ) -> None:
        self.config = config
        self._run_paths = run_paths
        self.generator = generator
        self.discriminator = discriminator
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self._train_dir = train_dir
        self._val_dir = val_dir
        self.losses = losses
        self._target_modality = target_modality
        self._input_names = cast(tuple[str, ...], generator.input_names)

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
            lr_g=config.lr_g,
            lr_d=config.lr_d,
            beta1=config.beta1,
            beta2=config.beta2,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            target_modality=target_modality,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resume(self, checkpoint: str | Path) -> int:
        """Load a checkpoint and return the next epoch to train."""
        if checkpoint == "latest":
            checkpoint_path = self._checkpoint_manager.latest()
            if checkpoint_path is None:
                raise FileNotFoundError(
                    f"resume='latest' but no checkpoints found in {self._checkpoints_dir}"
                )
        else:
            checkpoint_path = Path(checkpoint)
            if not checkpoint_path.is_absolute():
                checkpoint_path = self._checkpoints_dir / checkpoint_path
            checkpoint_path = checkpoint_path.resolve()
            if checkpoint_path.suffix != ".pth":
                raise ValueError(
                    f"resume checkpoint path must end with '.pth'; got {checkpoint_path}"
                )
            if not checkpoint_path.is_file():
                raise FileNotFoundError(f"resume checkpoint not found: {checkpoint_path}")

        return self._checkpoint_manager.load(checkpoint_path)

    def train(
        self,
        seed: int,
        start_epoch: int = 0,
    ) -> TrainingResult:
        """Run the full training loop."""
        start_time = time.time()
        self._prepare_run_directories()

        log_file = self._training_log_path()
        with TrainingLogSession(log_file, logger, checkpoint_logger):
            self._clear_training_outputs()
            self._log_training_start(seed, start_epoch, log_file)

            session = self._run_training_epochs(start_epoch=start_epoch, start_time=start_time)

            if session.best_checkpoint_path is None:
                session.best_checkpoint_path = self._checkpoint_manager.latest()
                if session.best_checkpoint_path is not None:
                    session.best_checkpoint = session.best_checkpoint_path.name

            finish_console_progress()
            total_seconds = time.time() - start_time
            logger.info("Execution completed. Total time = %.2f seconds", total_seconds)
            return TrainingResult(
                final_epoch=session.final_epoch,
                best_checkpoint_path=session.best_checkpoint_path,
                stopped_early=session.stopped,
                stop_epoch=session.stop_epoch,
                stop_reason=session.stop_reason,
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
                early_stopping_best_epoch=session.early_stopping_best_epoch,
                early_stopping_best_value=session.early_stopping_best_value,
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
            self.config.scheduler.to_dict(),
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

    def _run_training_epochs(
        self,
        *,
        start_epoch: int,
        start_time: float,
    ) -> _TrainingSession:
        loss_names = configured_loss_names(self.losses)
        progress_tracker = self._start_progress_tracker(start_epoch)

        with TrainingHistory(self._run_paths.metrics_dir, loss_names) as history:
            session = _TrainingSession(
                start_epoch=start_epoch,
                start_time=start_time,
                progress_tracker=progress_tracker,
                history=history,
                last_checkpoint=(Path(self.config.resume).name if self.config.resume else "none"),
                final_epoch=max(start_epoch, self.config.epochs) - 1,
            )
            if start_epoch > 0:
                self._sync_best_checkpoint(session)

            for epoch in range(start_epoch, self.config.epochs):
                session.final_metrics = self._run_training_epoch(epoch=epoch, session=session)
                session.final_epoch = epoch
                if session.stopped:
                    break

        self._save_final_checkpoint_if_needed(session)
        return session

    def _run_training_epoch(
        self,
        *,
        epoch: int,
        session: _TrainingSession,
    ) -> EpochMetrics:
        logger.debug("Starting epoch %s", epoch)
        epoch_metrics = self._train_epoch(epoch, session)
        logger.debug("Finished epoch %s", epoch)

        val_metrics, validation_checkpoint_path = self._validate_and_update_best(
            epoch=epoch,
            epoch_metrics=epoch_metrics,
            session=session,
        )
        if val_metrics is None:
            self._step_lr_schedulers(epoch=epoch, val_metrics=None)

        self._save_scheduled_checkpoint(
            epoch=epoch,
            epoch_metrics=epoch_metrics,
            session=session,
            existing_checkpoint_path=validation_checkpoint_path,
        )

        session.history.write_epoch(epoch, epoch_metrics, val_metrics)
        if val_metrics is not None and self.config.early_stopping is not None:
            self._update_early_stopping(epoch=epoch, val_metrics=val_metrics, session=session)
        return epoch_metrics

    def _save_scheduled_checkpoint(
        self,
        *,
        epoch: int,
        epoch_metrics: EpochMetrics,
        session: _TrainingSession,
        existing_checkpoint_path: Path | None = None,
    ) -> Path | None:
        if (epoch + 1) % self.config.checkpoint_rate != 0:
            return None

        checkpoint_path = existing_checkpoint_path
        if checkpoint_path is None:
            checkpoint_path = self._checkpoint_manager.save(epoch)
            session.last_checkpoint = checkpoint_path.name
            logger.info("Checkpoint saved to %s at epoch %s", checkpoint_path, epoch)
        self._emit_epoch_progress(
            epoch=epoch,
            epoch_metrics=epoch_metrics,
            session=session,
            eta_str="0s" if epoch == self.config.epochs - 1 else "--",
        )
        return checkpoint_path

    def _validate_and_update_best(
        self,
        *,
        epoch: int,
        epoch_metrics: EpochMetrics,
        session: _TrainingSession,
    ) -> tuple[EpochMetrics | None, Path | None]:
        if (epoch + 1) % self.config.validate_rate != 0:
            return None, None

        val_metrics = self._validate(epoch)
        session.latest_eval_losses = StepLosses(
            loss_G=val_metrics.loss_G,
            loss_D=val_metrics.loss_D,
        )
        session.latest_eval_epoch = epoch
        self._step_lr_schedulers(epoch=epoch, val_metrics=val_metrics)
        checkpoint_metrics = self._checkpoint_selection_metrics(val_metrics)
        ranked_checkpoint_path: Path | None = None
        if not checkpoint_metrics:
            logger.warning("Skipping checkpoint ranking update because all metrics are non-finite")
        else:
            ranked_checkpoint_path = self._ensure_best_checkpoint_path(
                epoch=epoch,
                checkpoint_path=None,
                session=session,
            )
            config_hash = self._read_config_hash()
            loss_config = self.losses.to_dict() if self.losses is not None else None
            checkpoint_modes = self._checkpoint_selection_modes()
            update_checkpoint_selection(
                self._checkpoints_dir,
                metrics=checkpoint_metrics,
                modes=checkpoint_modes,
                top_k=self.config.checkpoint_top_k,
                epoch=epoch,
                checkpoint_path=ranked_checkpoint_path,
                config_hash=config_hash,
                loss_config=loss_config,
            )
            self._sync_best_checkpoint(session)
        self._emit_epoch_progress(
            epoch=epoch,
            epoch_metrics=epoch_metrics,
            session=session,
            eta_str="0s" if epoch == self.config.epochs - 1 else "--",
        )
        return val_metrics, ranked_checkpoint_path

    def _validate(self, epoch: int) -> EpochMetrics:
        return validate_epoch(
            epoch=epoch,
            generator=self.generator,
            discriminator=self.discriminator,
            val_loader=self.val_loader,
            loss_evaluator=self._loss_evaluator,
            losses=self.losses,
            device=self.device,
            amp_enabled=self._amp_enabled,
            output_dir=self._output_val_dir,
        )

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
        session: _TrainingSession,
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

        if self._is_early_stopping_improvement(metric_value, session.early_stopping_best_value):
            session.early_stopping_best_value = metric_value
            session.early_stopping_best_epoch = epoch
            session.early_stopping_stale_count = 0
            return False

        session.early_stopping_stale_count += 1
        if session.early_stopping_stale_count >= early_config.patience:
            session.stopped = True
            session.stop_epoch = epoch
            session.stop_reason = (
                f"early_stopping: {early_config.monitor} did not improve by at least "
                f"{early_config.min_delta:g} for {early_config.patience} validation event(s); "
                f"best epoch {session.early_stopping_best_epoch} "
                f"value {session.early_stopping_best_value:.6g}"
            )
            logger.info("Stopping early at epoch %s: %s", epoch, session.stop_reason)
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

    def _sync_best_checkpoint(self, session: _TrainingSession) -> None:
        try:
            record = load_best_checkpoint_record(
                self._checkpoints_dir,
                policy="best",
                metric="loss_G_val",
            )
        except FileNotFoundError:
            return
        session.best_checkpoint_path = record.checkpoint_path
        session.best_checkpoint = record.checkpoint_path.name
        session.best_loss_G_val = record.metric_value

    def _ensure_best_checkpoint_path(
        self,
        *,
        epoch: int,
        checkpoint_path: Path | None,
        session: _TrainingSession,
    ) -> Path:
        if checkpoint_path is not None:
            return checkpoint_path

        checkpoint_path = self._checkpoint_manager.save(epoch)
        session.last_checkpoint = checkpoint_path.name
        logger.info(
            "Checkpoint saved to %s at epoch %s for checkpoint selection",
            checkpoint_path,
            epoch,
        )
        return checkpoint_path

    def _save_final_checkpoint_if_needed(self, session: _TrainingSession) -> None:
        if session.start_epoch >= self.config.epochs or session.final_metrics is None:
            return

        if (session.final_epoch + 1) % self.config.checkpoint_rate == 0:
            return

        checkpoint_path = self._checkpoint_manager.save(session.final_epoch)
        session.last_checkpoint = checkpoint_path.name
        logger.info("Final checkpoint saved to %s (epoch %s)", checkpoint_path, session.final_epoch)
        if session.best_checkpoint_path is None:
            session.best_checkpoint_path = checkpoint_path
            session.best_checkpoint = checkpoint_path.name
        self._emit_epoch_progress(
            epoch=session.final_epoch,
            epoch_metrics=session.final_metrics,
            session=session,
            progress=1.0,
            eta_str="0s",
        )

    def _emit_epoch_progress(
        self,
        *,
        epoch: int,
        epoch_metrics: EpochMetrics,
        session: _TrainingSession,
        eta_str: str,
        progress: float | None = None,
    ) -> None:
        emit_progress_update(
            progress=(
                progress
                if progress is not None
                else (epoch + 1) / session.progress_tracker.total_epochs
            ),
            epoch_progress=1.0,
            epoch=epoch,
            batch_index=len(self.train_loader) - 1,
            progress_tracker=session.progress_tracker,
            step_losses=StepLosses(
                loss_G=epoch_metrics.loss_G,
                loss_D=epoch_metrics.loss_D,
            ),
            elapsed_str=format_duration(time.time() - session.start_time),
            eta_str=eta_str,
            end_time_str=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            last_checkpoint_name=session.last_checkpoint,
            best_checkpoint_name=session.best_checkpoint,
            best_checkpoint_loss_G_val=session.best_loss_G_val,
            eval_losses=session.latest_eval_losses,
            eval_epoch=session.latest_eval_epoch,
        )

    def _train_epoch(
        self,
        epoch: int,
        session: _TrainingSession,
    ) -> EpochMetrics:
        self.generator.train()
        self.discriminator.train()

        total_loss_G = 0.0
        total_loss_D = 0.0
        component_totals = LossComponentAccumulator(configured_loss_names(self.losses))
        num_batches = 0

        for i, batch in enumerate(self.train_loader):
            inputs, target, masks = unpack_batch(batch, self.device, self._input_names)
            step_losses = self._step.step(
                inputs,
                target,
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

            progress, elapsed, eta, end_time = session.progress_tracker.calculate_progress(epoch, i)
            elapsed_str = format_duration(elapsed)
            eta_str = format_duration(eta)
            epoch_progress = (i + 1) / session.progress_tracker.total_batches
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
                    progress_tracker=session.progress_tracker,
                    step_losses=step_losses,
                    elapsed_str=elapsed_str,
                    eta_str=eta_str,
                    end_time_str=end_time_str,
                    last_checkpoint_name=session.last_checkpoint,
                    best_checkpoint_name=session.best_checkpoint,
                    best_checkpoint_loss_G_val=session.best_loss_G_val,
                    eval_losses=session.latest_eval_losses,
                    eval_epoch=session.latest_eval_epoch,
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
