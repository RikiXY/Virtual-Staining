from __future__ import annotations

import csv
import datetime
import logging
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
from virtual_staining.training.checkpoints import (
    BEST_CHECKPOINT_POLICY,
    CheckpointManager,
    load_best_checkpoint_record,
)
from virtual_staining.training.config import LossConfig, TrainingConfig
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

if TYPE_CHECKING:
    from virtual_staining.reporting.base import TrainingReporter

logger = logging.getLogger(__name__)
checkpoint_logger = logging.getLogger("virtual_staining.training.checkpoints")


@dataclass
class _TrainingStatus:
    last_checkpoint: str
    best_checkpoint: str = "none"
    latest_eval_losses: StepLosses | None = None
    latest_eval_epoch: int | None = None


@dataclass
class _BestCheckpointState:
    path: Path | None = None
    val_loss: float | None = None


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

            self._run_training_epochs(
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
            final_epoch = max(start_epoch, self.config.epochs) - 1
            return TrainingResult(
                final_epoch=final_epoch,
                best_checkpoint_path=best_state.path,
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
            "Hyperparameters | lr_g=%s | lr_d=%s | beta1=%s | beta2=%s",
            self.config.lr_g,
            self.config.lr_d,
            self.config.beta1,
            self.config.beta2,
        )
        logger.info("Training started")

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
        if start_epoch == 0:
            return _BestCheckpointState()

        try:
            best_record = load_best_checkpoint_record(
                self._checkpoints_dir,
                policy=BEST_CHECKPOINT_POLICY,
            )
        except FileNotFoundError:
            logger.info("No existing best checkpoint record found for resumed training")
            return _BestCheckpointState()

        training_status.best_checkpoint = best_record.checkpoint_path.name
        logger.info(
            "Resumed best checkpoint: %s (%s=%.6f at epoch %s)",
            best_record.checkpoint_path,
            best_record.metric,
            best_record.metric_value,
            best_record.epoch,
        )
        return _BestCheckpointState(
            path=best_record.checkpoint_path,
            val_loss=best_record.metric_value,
        )

    def _run_training_epochs(
        self,
        *,
        start_epoch: int,
        progress_tracker: ProgressTracker,
        training_status: _TrainingStatus,
        best_state: _BestCheckpointState,
        start_time: float,
        reporter: TrainingReporter | None,
    ) -> None:
        loss_names = configured_loss_names(self.losses)
        metrics_path = self._run_paths.metrics_dir / "metrics.csv"
        last_epoch_metrics: EpochMetrics | None = None

        with open(metrics_path, "w", newline="", encoding="utf-8") as metrics_file:
            metrics_writer = csv.DictWriter(
                metrics_file,
                fieldnames=metrics_fieldnames(loss_names),
            )
            metrics_writer.writeheader()

            for epoch in range(start_epoch, self.config.epochs):
                last_epoch_metrics = self._run_training_epoch(
                    epoch=epoch,
                    metrics_writer=metrics_writer,
                    metrics_file=metrics_file,
                    progress_tracker=progress_tracker,
                    training_status=training_status,
                    best_state=best_state,
                    start_time=start_time,
                    reporter=reporter,
                )

        self._save_final_checkpoint_if_needed(
            start_epoch=start_epoch,
            final_metrics=last_epoch_metrics,
            progress_tracker=progress_tracker,
            training_status=training_status,
            best_state=best_state,
            start_time=start_time,
            reporter=reporter,
        )

    def _run_training_epoch(
        self,
        *,
        epoch: int,
        metrics_writer: csv.DictWriter,
        metrics_file: TextIO,
        progress_tracker: ProgressTracker,
        training_status: _TrainingStatus,
        best_state: _BestCheckpointState,
        start_time: float,
        reporter: TrainingReporter | None,
    ) -> EpochMetrics:
        logger.debug("Starting epoch %s", epoch)
        epoch_metrics = self._train_epoch(epoch, progress_tracker, training_status)
        logger.debug("Finished epoch %s", epoch)

        checkpoint_path = self._save_scheduled_checkpoint(
            epoch=epoch,
            epoch_metrics=epoch_metrics,
            progress_tracker=progress_tracker,
            training_status=training_status,
            start_time=start_time,
            reporter=reporter,
        )

        val_metrics = self._validate_and_update_best(
            epoch=epoch,
            epoch_metrics=epoch_metrics,
            checkpoint_path=checkpoint_path,
            progress_tracker=progress_tracker,
            training_status=training_status,
            best_state=best_state,
            start_time=start_time,
            reporter=reporter,
        )

        self._write_epoch_metrics(metrics_writer, epoch, epoch_metrics, val_metrics)
        metrics_file.flush()
        if reporter is not None:
            reporter.on_epoch_completed(epoch_metrics)
        return epoch_metrics

    def _save_scheduled_checkpoint(
        self,
        *,
        epoch: int,
        epoch_metrics: EpochMetrics,
        progress_tracker: ProgressTracker,
        training_status: _TrainingStatus,
        start_time: float,
        reporter: TrainingReporter | None,
    ) -> Path | None:
        if (epoch + 1) % self.config.checkpoint_rate != 0:
            return None

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
        checkpoint_path: Path | None,
        progress_tracker: ProgressTracker,
        training_status: _TrainingStatus,
        best_state: _BestCheckpointState,
        start_time: float,
        reporter: TrainingReporter | None,
    ) -> EpochMetrics | None:
        if (epoch + 1) % self.config.validate_rate != 0:
            return None

        val_metrics = self._validate(epoch)
        training_status.latest_eval_losses = StepLosses(
            loss_G=val_metrics.loss_G,
            loss_D=val_metrics.loss_D,
        )
        training_status.latest_eval_epoch = epoch
        if best_state.val_loss is None or val_metrics.loss_G < best_state.val_loss:
            best_checkpoint_path = self._ensure_best_checkpoint_path(
                epoch=epoch,
                checkpoint_path=checkpoint_path,
                training_status=training_status,
                reporter=reporter,
            )
            self._checkpoint_manager.save_best_record(
                policy=BEST_CHECKPOINT_POLICY,
                metric="loss_G_val",
                epoch=epoch,
                checkpoint_path=best_checkpoint_path,
                metric_value=val_metrics.loss_G,
            )
            best_state.val_loss = val_metrics.loss_G
            best_state.path = best_checkpoint_path
            training_status.best_checkpoint = best_checkpoint_path.name
        self._emit_epoch_progress(
            epoch=epoch,
            epoch_metrics=epoch_metrics,
            progress_tracker=progress_tracker,
            training_status=training_status,
            start_time=start_time,
            eta_str="0s" if epoch == self.config.epochs - 1 else "--",
        )
        return val_metrics

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
            "Checkpoint saved to %s at epoch %s for %s",
            checkpoint_path,
            epoch,
            BEST_CHECKPOINT_POLICY,
        )
        if reporter is not None:
            reporter.on_checkpoint_saved(checkpoint_path, epoch)
        return checkpoint_path

    def _write_epoch_metrics(
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
        metrics_writer.writerow(row)

    def _save_final_checkpoint_if_needed(
        self,
        *,
        start_epoch: int,
        final_metrics: EpochMetrics | None,
        progress_tracker: ProgressTracker,
        training_status: _TrainingStatus,
        best_state: _BestCheckpointState,
        start_time: float,
        reporter: TrainingReporter | None,
    ) -> None:
        if start_epoch >= self.config.epochs or final_metrics is None:
            return

        final_epoch = self.config.epochs - 1
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
            count = 0

            with torch.no_grad():
                for i, batch in enumerate(self.val_loader):
                    x, y, masks = unpack_batch(batch, self.device)

                    with autocast(device_type=self.device.type, enabled=self._amp_enabled):
                        fake = self.generator(x)
                        D_real = self.discriminator(x, y)
                        D_fake = self.discriminator(x, fake)
                        context = LossEvaluationContext(epoch=epoch, masks=masks)
                        discriminator_loss = self._loss_evaluator.discriminator_total(
                            discriminator_real=D_real,
                            discriminator_fake=D_fake,
                            context=context,
                        )
                        generator_loss = self._loss_evaluator.generator_total(
                            prediction=fake,
                            target=y,
                            discriminator_fake=D_fake,
                            context=context,
                        )
                        loss_D = discriminator_loss.total
                        loss_G = generator_loss.total
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
            )
        finally:
            if generator_was_training:
                self.generator.train()
            if discriminator_was_training:
                self.discriminator.train()
