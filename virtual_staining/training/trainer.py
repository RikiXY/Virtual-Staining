from __future__ import annotations

import csv
import datetime
import io
import logging
import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, TextIO

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.amp import GradScaler, autocast
from torchvision.utils import save_image

from virtual_staining.experiment.run_paths import RunPaths
from virtual_staining.models.config import ModelConfig
from virtual_staining.training.checkpoints import (
    BEST_CHECKPOINT_POLICY,
    CheckpointManager,
    load_best_checkpoint_record,
)
from virtual_staining.training.config import LossConfig, TrainingConfig
from virtual_staining.training.losses import (
    ConfiguredLossEvaluator,
    LossEvaluationContext,
    StepLosses,
)
from virtual_staining.training.results import EpochMetrics, TrainingResult
from virtual_staining.training.steps import Pix2PixTrainingStep
from virtual_staining.utils.console import style

if TYPE_CHECKING:
    from virtual_staining.reporting.base import TrainingReporter

logger = logging.getLogger(__name__)
checkpoint_logger = logging.getLogger("virtual_staining.training.checkpoints")
_PLAIN_TEXT_STREAM = io.StringIO()

# ---------------------------------------------------------------------------
# Module-level helpers (private to this module)
# ---------------------------------------------------------------------------


def _is_amp_enabled(device: torch.device) -> bool:
    return isinstance(device, torch.device) and device.type == "cuda"


def _save_images(
    path: Path,
    source_tensor: torch.Tensor,
    output: torch.Tensor,
    target: torch.Tensor,
    epoch: int,
    batch_index: int,
) -> None:
    # Images are normalised to [-1, 1]; bring back to [0, 1] before saving.
    save_image((source_tensor * 0.5 + 0.5), path / f"epoch{epoch}_batch{batch_index}_input.tif")
    save_image((output * 0.5 + 0.5), path / f"epoch{epoch}_batch{batch_index}_output.tif")
    save_image((target * 0.5 + 0.5), path / f"epoch{epoch}_batch{batch_index}_target.tif")


def _dataset_len(loader: torch.utils.data.DataLoader) -> int:
    assert loader.dataset is not None
    return len(loader.dataset)  # type: ignore[arg-type]  -- Dataset.__len__ exists at runtime but is absent from torch stubs


def _get_first_pair_size(dataset: torch.utils.data.Dataset | None) -> dict | None:
    if dataset is None or len(dataset) == 0:  # type: ignore[arg-type]
        return None
    pairs = getattr(dataset, "pairs", None)
    if pairs is None:
        return None
    source_path, target_path = pairs[0]
    with Image.open(source_path) as src_img:
        source_size = src_img.size
    with Image.open(target_path) as tgt_img:
        target_size = tgt_img.size
    return {
        "source": source_size,
        "target": target_size,
        "source_path": str(source_path),
        "target_path": str(target_path),
    }


def _unpack_batch(
    batch: object,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor] | None]:
    if not isinstance(batch, (tuple, list)) or len(batch) not in {2, 3}:
        raise TypeError("training batches must contain (source, target) or (source, target, masks)")
    x = batch[0]
    y = batch[1]
    if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
        raise TypeError("training batch source and target must be tensors")
    masks = None
    if len(batch) == 3:
        raw_masks = batch[2]
        if not isinstance(raw_masks, dict):
            raise TypeError("training batch masks must be a mapping")
        masks = {}
        for name, value in raw_masks.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"training batch mask {name!r} must be a tensor")
            masks[str(name)] = value.to(device)
    return x.to(device), y.to(device), masks


def _configured_loss_names(losses: LossConfig | None) -> list[str]:
    if losses is None:
        return []
    names = [f"generator_{term.name}" for term in losses.generator]
    names.extend(f"discriminator_{term.name}" for term in losses.discriminator)
    return names


def _metrics_fieldnames(loss_names: list[str]) -> list[str]:
    fields = [
        "epoch",
        "loss_G_train",
        "loss_D_train",
        "loss_G_val",
        "loss_D_val",
    ]
    if loss_names:
        fields.extend(
            [
                "loss_train_total_generator",
                "loss_train_total_discriminator",
                "loss_val_total_generator",
                "loss_val_total_discriminator",
            ]
        )
    for stage in ("train", "val"):
        for term_name in loss_names:
            fields.extend(
                [
                    f"loss_{stage}_raw_{term_name}",
                    f"loss_{stage}_weighted_{term_name}",
                    f"loss_{stage}_current_weight_{term_name}",
                ]
            )
    return fields


def _component_metric_row(stage: str, metrics: EpochMetrics | None) -> dict[str, str]:
    if metrics is None:
        return {}
    if not metrics.raw and not metrics.weighted and not metrics.current_weight:
        return {}
    row = {
        f"loss_{stage}_total_generator": f"{metrics.loss_G:.6f}",
        f"loss_{stage}_total_discriminator": f"{metrics.loss_D:.6f}",
    }
    for term_name in sorted(metrics.raw):
        row[f"loss_{stage}_raw_{term_name}"] = f"{metrics.raw[term_name]:.6f}"
    for term_name in sorted(metrics.weighted):
        row[f"loss_{stage}_weighted_{term_name}"] = f"{metrics.weighted[term_name]:.6f}"
    for term_name in sorted(metrics.current_weight):
        row[f"loss_{stage}_current_weight_{term_name}"] = f"{metrics.current_weight[term_name]:.6f}"
    return row


def _average_components(
    totals: dict[str, float],
    count: int,
    loss_names: list[str],
) -> dict[str, float]:
    if count == 0:
        return {}
    return {name: totals.get(name, 0.0) / count for name in loss_names}


def _accumulate_components(totals: dict[str, float], values: dict[str, float] | None) -> None:
    if values is None:
        return
    for name, value in values.items():
        totals[name] = totals.get(name, 0.0) + value


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "--"
    seconds = int(max(seconds, 0))
    if seconds < 60:
        return f"{seconds}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {sec:02d}s"
    hours, minutes = divmod(minutes, 60)
    if hours < 24:
        return f"{hours}h {minutes:02d}m"
    days, hours = divmod(hours, 24)
    return f"{days}d {hours:02d}h"


def _style_console(text: str, *names: str, stream: TextIO, color: bool) -> str:
    if not color:
        return text
    return style(text, *names, stream=stream)


def _color_progress(progress: float, stream: TextIO = sys.stderr) -> str:
    text = f"{progress:.2%}"
    if progress < 0.33:
        return style(text, "yellow", stream=stream)
    if progress < 0.66:
        return style(text, "cyan", stream=stream)
    if progress < 0.90:
        return style(text, "blue", stream=stream)
    return style(text, "green", stream=stream)


def _render_progress_bar(progress: float, width: int = 40, stream: TextIO = sys.stderr) -> str:
    progress = min(max(progress, 0.0), 1.0)
    filled = int(width * progress)
    if progress > 0 and filled == 0:
        filled = 1
    if progress >= 1:
        filled = width
    empty = width - filled
    bar = "█" * filled + "-" * empty
    return f"[{style(bar, 'green', stream=stream)}]"


def _format_loss_parts(
    step_losses: StepLosses,
    *,
    stream: TextIO,
    color: bool,
) -> str:
    parts = [
        (
            f"{_style_console('loss_G', 'bold', stream=stream, color=color)} "
            f"{_style_console(f'{step_losses.loss_G:.4f}', 'cyan', stream=stream, color=color)}"
        )
    ]
    parts.append(
        f"{_style_console('loss_D', 'bold', stream=stream, color=color)} "
        f"{_style_console(f'{step_losses.loss_D:.4f}', 'orange', stream=stream, color=color)}"
    )
    return " | ".join(parts)


def _format_progress_message(
    *,
    progress: float,
    epoch_progress: float,
    epoch: int,
    batch_index: int,
    progress_tracker: _ProgressTracker,
    step_losses: StepLosses,
    elapsed_str: str,
    eta_str: str,
    end_time_str: str,
    last_checkpoint_name: str,
    best_checkpoint_name: str,
    stream: TextIO = sys.stderr,
    color: bool = True,
) -> str:
    bar = (
        _render_progress_bar(progress, stream=stream)
        if color
        else _render_progress_bar(progress, stream=_PLAIN_TEXT_STREAM)
    )
    progress_text = _color_progress(progress, stream=stream) if color else f"{progress:.2%}"
    return (
        f"{bar} "
        f"ep {epoch + 1}/{progress_tracker.total_epochs} "
        f"({progress_text}) | "
        f"b {batch_index + 1}/{progress_tracker.total_batches} "
        f"({epoch_progress:.0%}) | "
        f"{_format_loss_parts(step_losses, stream=stream, color=color)} | "
        f"elapsed {elapsed_str} | "
        f"ETA {eta_str} | "
        f"end {end_time_str} | "
        f"last ckpt {last_checkpoint_name} | "
        f"best ckpt {best_checkpoint_name}"
    )


def _emit_progress_update(
    *,
    progress: float,
    epoch_progress: float,
    epoch: int,
    batch_index: int,
    progress_tracker: _ProgressTracker,
    step_losses: StepLosses,
    elapsed_str: str,
    eta_str: str,
    end_time_str: str,
    last_checkpoint_name: str,
    best_checkpoint_name: str,
) -> None:
    console_message = _format_progress_message(
        progress=progress,
        epoch_progress=epoch_progress,
        epoch=epoch,
        batch_index=batch_index,
        progress_tracker=progress_tracker,
        step_losses=step_losses,
        elapsed_str=elapsed_str,
        eta_str=eta_str,
        end_time_str=end_time_str,
        last_checkpoint_name=last_checkpoint_name,
        best_checkpoint_name=best_checkpoint_name,
        stream=sys.stderr,
        color=True,
    )
    _update_console_progress(console_message)

    log_message = _format_progress_message(
        progress=progress,
        epoch_progress=epoch_progress,
        epoch=epoch,
        batch_index=batch_index,
        progress_tracker=progress_tracker,
        step_losses=step_losses,
        elapsed_str=elapsed_str,
        eta_str=eta_str,
        end_time_str=end_time_str,
        last_checkpoint_name=last_checkpoint_name,
        best_checkpoint_name=best_checkpoint_name,
        stream=_PLAIN_TEXT_STREAM,
        color=False,
    )
    logger.debug("%s", log_message)


def _terminal_width(stream: TextIO = sys.stderr) -> int:
    try:
        return os.get_terminal_size(stream.fileno()).columns
    except OSError:
        return 140


def _update_console_progress(message: str, stream: TextIO = sys.stderr) -> None:
    if not stream.isatty():
        stream.write(message + "\n")
        stream.flush()
        return

    terminal_width = _terminal_width(stream)
    clean_message = message[: terminal_width - 1]
    stream.write("\r" + clean_message.ljust(terminal_width - 1))
    stream.flush()


def _finish_console_progress(stream: TextIO = sys.stderr) -> None:
    stream.write("\n")
    stream.flush()


# ---------------------------------------------------------------------------
# Progress tracker
# ---------------------------------------------------------------------------


class _ProgressTracker:
    def __init__(
        self,
        total_epochs: int,
        total_batches: int,
        start_epoch: int = 0,
        max_history: int = 300,
        warmup_batches: int = 10,
        min_eta_batches: int = 5,
    ) -> None:
        self.total_epochs = total_epochs
        self.total_batches = total_batches
        self.start_epoch = start_epoch
        self.max_history = max_history
        self.warmup_batches = warmup_batches
        self.min_eta_batches = min_eta_batches

        self.total_steps = total_epochs * total_batches
        self.start_step = start_epoch * total_batches

        self.start_time: float | None = None
        self.last_step_time: float | None = None
        self.step_durations: list[float] = []

    def start(self) -> None:
        now = time.time()
        self.start_time = now
        self.last_step_time = now
        self.step_durations = []

    def calculate_progress(
        self, epoch: int, batch: int
    ) -> tuple[float, float, float | None, float | None]:
        now = time.time()

        current_step = epoch * self.total_batches + batch + 1
        completed_since_start = current_step - self.start_step
        remaining_steps = max(self.total_steps - current_step, 0)

        assert self.last_step_time is not None, "call start() before calculate_progress()"
        step_duration = now - self.last_step_time
        self.last_step_time = now

        if completed_since_start > self.warmup_batches:
            self.step_durations.append(step_duration)
            if len(self.step_durations) > self.max_history:
                self.step_durations.pop(0)

        assert self.start_time is not None, "call start() before calculate_progress()"
        total_elapsed_time = now - self.start_time
        progress = current_step / self.total_steps if self.total_steps > 0 else 1.0

        if len(self.step_durations) < self.min_eta_batches:
            eta = None
            end_time = None
        else:
            avg_step_time = sum(self.step_durations) / len(self.step_durations)
            eta = avg_step_time * remaining_steps
            end_time = now + eta

        return progress, total_elapsed_time, eta, end_time


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

        self._amp_enabled = _is_amp_enabled(device)

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

        for d in [
            self._logs_dir,
            self._checkpoints_dir,
            self._output_val_dir,
            self._output_train_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)

        log_file = self._logs_dir / "training.log"
        file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        old_level = logger.level
        old_propagate = logger.propagate
        old_checkpoint_level = checkpoint_logger.level
        old_checkpoint_propagate = checkpoint_logger.propagate
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        logger.addHandler(file_handler)
        checkpoint_logger.setLevel(logging.DEBUG)
        checkpoint_logger.propagate = False
        checkpoint_logger.addHandler(file_handler)
        try:
            device_name = (
                torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else "CPU"
            )
            logger.debug("Seed set to %s", seed)
            logger.info("Device: %s (%s)", self.device, device_name)

            for f in self._output_train_dir.iterdir():
                if f.is_file():
                    f.unlink()

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
            logger.info("Train samples: %s", _dataset_len(self.train_loader))
            logger.info("Validation samples: %s", _dataset_len(self.val_loader))
            logger.info("Train batches/epoch: %s", len(self.train_loader))
            logger.info("Validation batches: %s", len(self.val_loader))
            logger.info("Detailed log: %s", log_file)
            logger.info("Training started")

            progress_tracker = _ProgressTracker(
                total_epochs=self.config.epochs,
                total_batches=len(self.train_loader),
                start_epoch=start_epoch,
                warmup_batches=max(10, self.config.log_rate),
            )
            progress_tracker.start()

            logger.info(
                "Hyperparameters | lr_g=%s | lr_d=%s | beta1=%s | beta2=%s",
                self.config.lr_g,
                self.config.lr_d,
                self.config.beta1,
                self.config.beta2,
            )

            training_status = {
                "last_checkpoint": (
                    Path(self.config.resume).name if self.config.resume else "none "
                ),
                "best_checkpoint": "none",
            }

            metrics_path = self._run_paths.metrics_dir / "metrics.csv"
            best_checkpoint_path: Path | None = None
            best_val_loss: float | None = None
            if start_epoch > 0:
                try:
                    best_record = load_best_checkpoint_record(
                        self._checkpoints_dir,
                        policy=BEST_CHECKPOINT_POLICY,
                    )
                except FileNotFoundError:
                    logger.info("No existing best checkpoint record found for resumed training")
                else:
                    best_checkpoint_path = best_record.checkpoint_path
                    best_val_loss = best_record.metric_value
                    training_status["best_checkpoint"] = best_record.checkpoint_path.name
                    logger.info(
                        "Resumed best checkpoint: %s (%s=%.6f at epoch %s)",
                        best_record.checkpoint_path,
                        best_record.metric,
                        best_record.metric_value,
                        best_record.epoch,
                    )
            loss_names = _configured_loss_names(self.losses)
            with open(metrics_path, "w", newline="", encoding="utf-8") as metrics_file:
                metrics_writer = csv.DictWriter(
                    metrics_file,
                    fieldnames=_metrics_fieldnames(loss_names),
                )
                metrics_writer.writeheader()

                for epoch in range(start_epoch, self.config.epochs):
                    logger.debug("Starting epoch %s", epoch)

                    epoch_metrics = self._train_epoch(epoch, progress_tracker, training_status)
                    checkpoint_path_for_epoch: Path | None = None

                    logger.debug("Finished epoch %s", epoch)

                    if (epoch + 1) % self.config.checkpoint_rate == 0:
                        checkpoint_path_for_epoch = self._checkpoint_manager.save(epoch)
                        training_status["last_checkpoint"] = checkpoint_path_for_epoch.name
                        logger.info(
                            "Checkpoint saved to %s at epoch %s", checkpoint_path_for_epoch, epoch
                        )
                        if reporter is not None:
                            reporter.on_checkpoint_saved(checkpoint_path_for_epoch, epoch)
                        elapsed_str = _format_duration(time.time() - start_time)
                        end_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        epoch_step_losses = StepLosses(
                            loss_G=epoch_metrics.loss_G,
                            loss_D=epoch_metrics.loss_D,
                        )
                        _emit_progress_update(
                            progress=(epoch + 1) / progress_tracker.total_epochs,
                            epoch_progress=1.0,
                            epoch=epoch,
                            batch_index=len(self.train_loader) - 1,
                            progress_tracker=progress_tracker,
                            step_losses=epoch_step_losses,
                            elapsed_str=elapsed_str,
                            eta_str="0s" if epoch == self.config.epochs - 1 else "--",
                            end_time_str=end_time_str,
                            last_checkpoint_name=training_status["last_checkpoint"],
                            best_checkpoint_name=training_status["best_checkpoint"],
                        )

                    val_metrics = None
                    if (epoch + 1) % self.config.validate_rate == 0:
                        val_metrics = self._validate(epoch)
                        if best_val_loss is None or val_metrics.loss_G < best_val_loss:
                            if checkpoint_path_for_epoch is None:
                                checkpoint_path_for_epoch = self._checkpoint_manager.save(epoch)
                                training_status["last_checkpoint"] = checkpoint_path_for_epoch.name
                                logger.info(
                                    "Checkpoint saved to %s at epoch %s for %s",
                                    checkpoint_path_for_epoch,
                                    epoch,
                                    BEST_CHECKPOINT_POLICY,
                                )
                                if reporter is not None:
                                    reporter.on_checkpoint_saved(checkpoint_path_for_epoch, epoch)
                            self._checkpoint_manager.save_best_record(
                                policy=BEST_CHECKPOINT_POLICY,
                                metric="loss_G_val",
                                epoch=epoch,
                                checkpoint_path=checkpoint_path_for_epoch,
                                metric_value=val_metrics.loss_G,
                            )
                            best_val_loss = val_metrics.loss_G
                            best_checkpoint_path = checkpoint_path_for_epoch
                            training_status["best_checkpoint"] = checkpoint_path_for_epoch.name
                            elapsed_str = _format_duration(time.time() - start_time)
                            end_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            _emit_progress_update(
                                progress=(epoch + 1) / progress_tracker.total_epochs,
                                epoch_progress=1.0,
                                epoch=epoch,
                                batch_index=len(self.train_loader) - 1,
                                progress_tracker=progress_tracker,
                                step_losses=StepLosses(
                                    loss_G=epoch_metrics.loss_G,
                                    loss_D=epoch_metrics.loss_D,
                                ),
                                elapsed_str=elapsed_str,
                                eta_str="0s" if epoch == self.config.epochs - 1 else "--",
                                end_time_str=end_time_str,
                                last_checkpoint_name=training_status["last_checkpoint"],
                                best_checkpoint_name=training_status["best_checkpoint"],
                            )

                    row = {
                        "epoch": epoch,
                        "loss_G_train": f"{epoch_metrics.loss_G:.6f}",
                        "loss_D_train": f"{epoch_metrics.loss_D:.6f}",
                        "loss_G_val": f"{val_metrics.loss_G:.6f}" if val_metrics else "",
                        "loss_D_val": f"{val_metrics.loss_D:.6f}" if val_metrics else "",
                    }
                    row.update(_component_metric_row("train", epoch_metrics))
                    row.update(_component_metric_row("val", val_metrics))
                    metrics_writer.writerow(row)
                    metrics_file.flush()
                    if reporter is not None:
                        reporter.on_epoch_completed(epoch_metrics)

            if start_epoch < self.config.epochs:
                final_epoch = self.config.epochs - 1
                if (final_epoch + 1) % self.config.checkpoint_rate != 0:
                    checkpoint_path = self._checkpoint_manager.save(final_epoch)
                    training_status["last_checkpoint"] = checkpoint_path.name
                    logger.info(
                        "Final checkpoint saved to %s (epoch %s)",
                        checkpoint_path,
                        final_epoch,
                    )
                    if reporter is not None:
                        reporter.on_checkpoint_saved(checkpoint_path, final_epoch)
                    if best_checkpoint_path is None:
                        best_checkpoint_path = checkpoint_path
                        training_status["best_checkpoint"] = checkpoint_path.name
                    elapsed_str = _format_duration(time.time() - start_time)
                    end_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    _emit_progress_update(
                        progress=1.0,
                        epoch_progress=1.0,
                        epoch=final_epoch,
                        batch_index=len(self.train_loader) - 1,
                        progress_tracker=progress_tracker,
                        step_losses=StepLosses(
                            loss_G=epoch_metrics.loss_G,
                            loss_D=epoch_metrics.loss_D,
                        ),
                        elapsed_str=elapsed_str,
                        eta_str="0s",
                        end_time_str=end_time_str,
                        last_checkpoint_name=training_status["last_checkpoint"],
                        best_checkpoint_name=training_status["best_checkpoint"],
                    )
            if best_checkpoint_path is None:
                best_checkpoint_path = self._checkpoint_manager.latest()
                if best_checkpoint_path is not None:
                    training_status["best_checkpoint"] = best_checkpoint_path.name

            _finish_console_progress()
            total_seconds = time.time() - start_time
            logger.info("Execution completed. Total time = %.2f seconds", total_seconds)
            final_epoch = max(start_epoch, self.config.epochs) - 1
            return TrainingResult(
                final_epoch=final_epoch,
                best_checkpoint_path=best_checkpoint_path,
            )
        finally:
            checkpoint_logger.removeHandler(file_handler)
            logger.removeHandler(file_handler)
            file_handler.close()
            checkpoint_logger.propagate = old_checkpoint_propagate
            checkpoint_logger.setLevel(old_checkpoint_level)
            logger.propagate = old_propagate
            logger.setLevel(old_level)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _train_epoch(
        self,
        epoch: int,
        progress_tracker: _ProgressTracker,
        training_status: dict,
    ) -> EpochMetrics:
        self.generator.train()
        self.discriminator.train()

        total_loss_G = 0.0
        total_loss_D = 0.0
        raw_totals: dict[str, float] = {}
        weighted_totals: dict[str, float] = {}
        current_weight_totals: dict[str, float] = {}
        loss_names = _configured_loss_names(self.losses)
        num_batches = 0

        for i, batch in enumerate(self.train_loader):
            x, y, masks = _unpack_batch(batch, self.device)
            step_losses = self._step.step(
                x,
                y,
                epoch=epoch,
                global_step=epoch * len(self.train_loader) + i,
                masks=masks,
            )
            total_loss_G += step_losses.loss_G
            total_loss_D += step_losses.loss_D
            _accumulate_components(raw_totals, step_losses.raw)
            _accumulate_components(weighted_totals, step_losses.weighted)
            _accumulate_components(current_weight_totals, step_losses.current_weight)
            num_batches += 1

            progress, elapsed, eta, end_time = progress_tracker.calculate_progress(epoch, i)
            elapsed_str = _format_duration(elapsed)
            eta_str = _format_duration(eta)
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
                _emit_progress_update(
                    progress=progress,
                    epoch_progress=epoch_progress,
                    epoch=epoch,
                    batch_index=i,
                    progress_tracker=progress_tracker,
                    step_losses=step_losses,
                    elapsed_str=elapsed_str,
                    eta_str=eta_str,
                    end_time_str=end_time_str,
                    last_checkpoint_name=training_status["last_checkpoint"],
                    best_checkpoint_name=training_status["best_checkpoint"],
                )

        if num_batches == 0:
            raise RuntimeError("Training loader was empty; cannot compute epoch metrics.")
        return EpochMetrics(
            loss_G=total_loss_G / num_batches,
            loss_D=total_loss_D / num_batches,
            raw=_average_components(raw_totals, num_batches, loss_names),
            weighted=_average_components(weighted_totals, num_batches, loss_names),
            current_weight=_average_components(current_weight_totals, num_batches, loss_names),
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
            raw_totals: dict[str, float] = {}
            weighted_totals: dict[str, float] = {}
            current_weight_totals: dict[str, float] = {}
            loss_names = _configured_loss_names(self.losses)
            count = 0

            with torch.no_grad():
                for i, batch in enumerate(self.val_loader):
                    x, y, masks = _unpack_batch(batch, self.device)

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
                        _accumulate_components(raw_totals, discriminator_loss.raw)
                        _accumulate_components(weighted_totals, discriminator_loss.weighted)
                        _accumulate_components(
                            current_weight_totals,
                            discriminator_loss.current_weight,
                        )
                        _accumulate_components(raw_totals, generator_loss.raw)
                        _accumulate_components(weighted_totals, generator_loss.weighted)
                        _accumulate_components(current_weight_totals, generator_loss.current_weight)

                    total_loss_D += loss_D.item()
                    total_loss_G += loss_G.item()
                    count += 1

                    if i < 5:
                        _save_images(self._output_val_dir, x[0], fake[0], y[0], epoch, i)

            avg_loss_G = total_loss_G / count if count > 0 else 0.0
            avg_loss_D = total_loss_D / count if count > 0 else 0.0

            logger.info(
                "[Epoch %s] Validation: loss_G=%.4f loss_D=%.4f",
                epoch,
                avg_loss_G,
                avg_loss_D,
            )

            return EpochMetrics(
                loss_G=avg_loss_G,
                loss_D=avg_loss_D,
                raw=_average_components(raw_totals, count, loss_names),
                weighted=_average_components(weighted_totals, count, loss_names),
                current_weight=_average_components(current_weight_totals, count, loss_names),
            )
        finally:
            if generator_was_training:
                self.generator.train()
            if discriminator_was_training:
                self.discriminator.train()
