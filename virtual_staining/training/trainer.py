from __future__ import annotations

import datetime
import logging
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from virtual_staining.checkpoint_selection import (
    load_best_checkpoint_record,
    update_checkpoint_selection,
)
from virtual_staining.config.training import TrainingConfig
from virtual_staining.experiment.run_layout import RunLayout
from virtual_staining.experiment.session import ExperimentSession
from virtual_staining.training.checkpoints import MethodCheckpointManager
from virtual_staining.training.helpers import LossComponentAccumulator, dataset_len
from virtual_staining.training.history import TrainingHistory
from virtual_staining.training.progress import (
    ProgressReporter,
    ProgressTracker,
    ProgressUpdate,
    format_duration,
    format_progress_log,
)
from virtual_staining.training.results import TrainingResult
from virtual_staining.training.runtime import MethodMetrics, TrainingMethodRuntime

if TYPE_CHECKING:
    from virtual_staining.training.benchmarking import TrainingBenchmarkRecorder

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
    best_checkpoint_metric_value: float | None = None
    latest_eval_metrics: MethodMetrics | None = None
    latest_eval_epoch: int | None = None
    early_stopping_best_value: float | None = None
    early_stopping_best_epoch: int | None = None
    early_stopping_stale_count: int = 0
    stopped: bool = False
    stop_epoch: int | None = None
    stop_reason: str | None = None
    final_epoch: int = -1
    final_metrics: MethodMetrics | None = None


class Trainer:
    """Orchestrate method-independent training lifecycle and run services."""

    def __init__(
        self,
        config: TrainingConfig,
        run_paths: RunLayout,
        method: TrainingMethodRuntime,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        device: torch.device,
        *,
        train_dir: Path,
        val_dir: Path,
        experiment_session: ExperimentSession,
        config_hash: str,
        image_size: tuple[int, int],
        progress_reporter: ProgressReporter | None = None,
        benchmark_recorder: TrainingBenchmarkRecorder | None = None,
    ) -> None:
        self.config = config
        self.method = method
        self.progress_reporter = progress_reporter
        self._benchmark_recorder = benchmark_recorder
        self._run_paths = run_paths
        self._experiment_session = experiment_session
        self._config_hash = config_hash
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self._train_dir = train_dir
        self._val_dir = val_dir
        self.losses = method.loss_config
        self._logs_dir = run_paths.logs_dir
        self._checkpoints_dir = run_paths.checkpoints_dir
        self._output_val_dir = run_paths.output_val_dir
        self._output_train_dir = run_paths.output_train_dir
        self._checkpoints = MethodCheckpointManager(
            method,
            run_paths.checkpoints_dir,
            image_size=image_size,
            device=device,
            config_hash=config_hash,
        )

    def resume(self, checkpoint: str | Path) -> int:
        if checkpoint == "latest":
            checkpoint_path = self._checkpoints.latest()
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

        return self._checkpoints.load(checkpoint_path)

    def train(
        self,
        seed: int,
        start_epoch: int = 0,
    ) -> TrainingResult:
        start_time = time.time()
        self._prepare_run_directories()
        self._clear_training_outputs()
        self._log_training_start(seed, start_epoch)

        session = self._run_training_epochs(start_epoch=start_epoch, start_time=start_time)

        if session.best_checkpoint_path is None:
            session.best_checkpoint_path = self._checkpoints.latest()
            if session.best_checkpoint_path is not None:
                session.best_checkpoint = session.best_checkpoint_path.name

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
                self.config.early_stopping.mode if self.config.early_stopping is not None else None
            ),
            early_stopping_best_epoch=session.early_stopping_best_epoch,
            early_stopping_best_value=session.early_stopping_best_value,
        )

    def _prepare_run_directories(self) -> None:
        for directory in [
            self._logs_dir,
            self._checkpoints_dir,
            self._output_val_dir,
            self._output_train_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

    def _clear_training_outputs(self) -> None:
        for output_file in self._output_train_dir.iterdir():
            if output_file.is_file():
                output_file.unlink()

    def _log_training_start(self, seed: int, start_epoch: int) -> None:
        device_name = (
            torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else "CPU"
        )
        logger.debug("Seed set to %s", seed)
        logger.info("Device: %s (%s)", self.device, device_name)

        if start_epoch > 0:
            logger.info("Training resumed from epoch %s", start_epoch)
        else:
            logger.debug("Training started from scratch")

        logger.info("=== %s training ===", self.method.name)
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
        learning_rates = " | ".join(
            f"{name}={value}" for name, value in self.method.learning_rates().items()
        )
        logger.info(
            "Optimization | %s | scheduler=%s",
            learning_rates or "no optimizer learning rates",
            self.config.scheduler.to_dict(),
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

    def _run_training_epochs(
        self,
        *,
        start_epoch: int,
        start_time: float,
    ) -> _TrainingSession:
        loss_names = list(self.method.loss_names)
        progress_tracker = self._start_progress_tracker(start_epoch)

        with TrainingHistory(
            self._run_paths.epochs_csv,
            loss_names,
            resume_at=start_epoch,
            metric_names=self.method.metric_names,
            component_total_names=self.method.component_total_names,
        ) as history:
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
    ) -> MethodMetrics:
        recorder = self._benchmark_recorder
        if recorder is not None:
            recorder.start_epoch(epoch)
        try:
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

            reported = session.history.write_epoch(epoch, epoch_metrics, val_metrics)
            self._experiment_session.log_metrics(reported, step=epoch)
            if val_metrics is not None and self.config.early_stopping is not None:
                self._update_early_stopping(epoch=epoch, val_metrics=val_metrics, session=session)
            return epoch_metrics
        finally:
            if recorder is not None:
                recorder.finish_epoch(epoch)

    def _save_scheduled_checkpoint(
        self,
        *,
        epoch: int,
        epoch_metrics: MethodMetrics,
        session: _TrainingSession,
        existing_checkpoint_path: Path | None = None,
    ) -> Path | None:
        if (epoch + 1) % self.config.checkpoint_rate != 0:
            return None

        checkpoint_path = existing_checkpoint_path
        if checkpoint_path is None:
            checkpoint_path = self._save_checkpoint(epoch)
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
        epoch_metrics: MethodMetrics,
        session: _TrainingSession,
    ) -> tuple[MethodMetrics | None, Path | None]:
        if (epoch + 1) % self.config.validate_rate != 0:
            return None, None

        val_metrics = self._validate(epoch)
        session.latest_eval_metrics = val_metrics
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
            config_hash = self._config_hash
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

    def _validate(self, epoch: int) -> MethodMetrics:
        recorder = self._benchmark_recorder
        if recorder is None:
            return self._validate_impl(epoch)
        with recorder.phase("validation"):
            return self._validate_impl(epoch)

    def _validate_impl(self, epoch: int) -> MethodMetrics:
        return self.method.validate(
            self.val_loader,
            epoch=epoch,
            output_dir=self._output_val_dir,
        )

    def _step_lr_schedulers(
        self,
        *,
        epoch: int,
        val_metrics: MethodMetrics | None,
    ) -> None:
        if self.method.step_schedulers(
            epoch=epoch,
            validation_metrics=val_metrics,
        ):
            self._log_learning_rates(epoch)

    def _update_early_stopping(
        self,
        *,
        epoch: int,
        val_metrics: MethodMetrics,
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

    def _early_stopping_monitor_value(self, val_metrics: MethodMetrics) -> float | None:
        early_config = self.config.early_stopping
        if early_config is None:
            return None
        return self.method.validation_metric(val_metrics, early_config.monitor)

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
        values = " | ".join(
            f"{name}={value}" for name, value in self.method.learning_rates().items()
        )
        logger.info("Learning rates | epoch=%s | %s", epoch, values or "none")

    def _checkpoint_selection_metrics(self, val_metrics: MethodMetrics) -> dict[str, float]:
        return self.method.checkpoint_selection_metrics(val_metrics)

    def _checkpoint_selection_modes(self) -> dict[str, str]:
        return self.method.checkpoint_selection_modes()

    def _sync_best_checkpoint(self, session: _TrainingSession) -> None:
        try:
            record = load_best_checkpoint_record(
                self._checkpoints_dir,
                policy="best",
                metric=self.method.default_checkpoint_metric,
            )
        except FileNotFoundError:
            return
        session.best_checkpoint_path = record.checkpoint_path
        session.best_checkpoint = record.checkpoint_path.name
        session.best_checkpoint_metric_value = record.metric_value

    def _ensure_best_checkpoint_path(
        self,
        *,
        epoch: int,
        checkpoint_path: Path | None,
        session: _TrainingSession,
    ) -> Path:
        if checkpoint_path is not None:
            return checkpoint_path

        checkpoint_path = self._save_checkpoint(epoch)
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

        checkpoint_path = self._save_checkpoint(session.final_epoch)
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

    def _save_checkpoint(self, epoch: int) -> Path:
        recorder = self._benchmark_recorder
        if recorder is None:
            return self._checkpoints.save(epoch)
        with recorder.phase("checkpoint"):
            return self._checkpoints.save(epoch)

    def _emit_epoch_progress(
        self,
        *,
        epoch: int,
        epoch_metrics: MethodMetrics,
        session: _TrainingSession,
        eta_str: str,
        progress: float | None = None,
    ) -> None:
        update = ProgressUpdate(
            progress=(
                progress
                if progress is not None
                else (epoch + 1) / session.progress_tracker.total_epochs
            ),
            epoch_progress=1.0,
            epoch=epoch,
            batch_index=len(self.train_loader) - 1,
            total_epochs=session.progress_tracker.total_epochs,
            total_batches=session.progress_tracker.total_batches,
            step_metrics=epoch_metrics.losses,
            elapsed_str=format_duration(time.time() - session.start_time),
            eta_str=eta_str,
            end_time_str=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            last_checkpoint_name=session.last_checkpoint,
            best_checkpoint_name=session.best_checkpoint,
            best_checkpoint_metric_name=self.method.default_checkpoint_metric,
            best_checkpoint_metric_value=session.best_checkpoint_metric_value,
            eval_metrics=(
                session.latest_eval_metrics.losses
                if session.latest_eval_metrics is not None
                else None
            ),
            eval_epoch=session.latest_eval_epoch,
        )
        if self.progress_reporter is not None:
            self.progress_reporter(update)
        logger.debug("%s", format_progress_log(update))

    def _train_epoch(
        self,
        epoch: int,
        session: _TrainingSession,
    ) -> MethodMetrics:
        self.method.train_mode()

        loss_totals: dict[str, float] = {}
        method_component_totals: dict[str, float] = {}
        component_totals = LossComponentAccumulator(list(self.method.loss_names))
        num_batches = 0

        recorder = self._benchmark_recorder
        batch_cycle_started = time.perf_counter() if recorder is not None else 0.0
        for i, batch in enumerate(self.train_loader):
            if recorder is not None:
                batch_received_at = time.perf_counter()
                recorder.batch_received(
                    epoch=epoch,
                    batch_index=i,
                    samples=self.method.batch_size(batch),
                    data_wait_seconds=batch_received_at - batch_cycle_started,
                    cycle_started_at=batch_cycle_started,
                )

            step_metrics = self.method.step(
                batch,
                epoch=epoch,
                global_step=epoch * len(self.train_loader) + i,
            )
            self._validate_method_metric_names(step_metrics)
            _accumulate_values(loss_totals, step_metrics.losses)
            _accumulate_values(method_component_totals, step_metrics.component_totals)
            component_totals.add(
                raw=step_metrics.raw,
                weighted=step_metrics.weighted,
                current_weight=step_metrics.current_weight,
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
                update = ProgressUpdate(
                    progress=progress,
                    epoch_progress=epoch_progress,
                    epoch=epoch,
                    batch_index=i,
                    total_epochs=session.progress_tracker.total_epochs,
                    total_batches=session.progress_tracker.total_batches,
                    step_metrics=step_metrics.losses,
                    elapsed_str=elapsed_str,
                    eta_str=eta_str,
                    end_time_str=end_time_str,
                    last_checkpoint_name=session.last_checkpoint,
                    best_checkpoint_name=session.best_checkpoint,
                    best_checkpoint_metric_name=self.method.default_checkpoint_metric,
                    best_checkpoint_metric_value=session.best_checkpoint_metric_value,
                    eval_metrics=(
                        session.latest_eval_metrics.losses
                        if session.latest_eval_metrics is not None
                        else None
                    ),
                    eval_epoch=session.latest_eval_epoch,
                )
                if self.progress_reporter is not None:
                    self.progress_reporter(update)
                logger.debug("%s", format_progress_log(update))

            if recorder is not None:
                recorder.finish_batch()
                batch_cycle_started = time.perf_counter()

        if num_batches == 0:
            raise RuntimeError("Training loader was empty; cannot compute epoch metrics.")

        component_averages = component_totals.average(num_batches)
        return MethodMetrics(
            losses={name: loss_totals[name] / num_batches for name in self.method.metric_names},
            component_totals={
                name: method_component_totals[name] / num_batches
                for name in self.method.component_total_names
                if name in method_component_totals
            },
            raw=component_averages.raw,
            weighted=component_averages.weighted,
            current_weight=component_averages.current_weight,
        )

    def _validate_method_metric_names(self, metrics: MethodMetrics) -> None:
        expected = tuple(self.method.metric_names)
        actual = tuple(metrics.losses)
        if actual != expected:
            raise ValueError(
                f"Method {self.method.name!r} returned training metrics {actual}; "
                f"expected {expected}"
            )


def _accumulate_values(totals: dict[str, float], values: Mapping[str, float]) -> None:
    for name, value in values.items():
        totals[name] = totals.get(name, 0.0) + value
