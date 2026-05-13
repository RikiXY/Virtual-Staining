from __future__ import annotations

import csv
import datetime
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.amp import GradScaler, autocast
from torchvision.utils import save_image

from virtual_staining.experiment.run_paths import RunPaths
from virtual_staining.models.config import ModelConfig
from virtual_staining.training.checkpoints import CheckpointManager
from virtual_staining.training.config import TrainingConfig
from virtual_staining.training.losses import build_gan_loss
from virtual_staining.training.results import EpochMetrics, TrainingResult
from virtual_staining.training.steps import Pix2PixTrainingStep

if TYPE_CHECKING:
    from virtual_staining.reporting.base import TrainingReporter

logger = logging.getLogger(__name__)

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


def _color_progress(progress: float) -> str:
    return f"{progress:.2%}"


def _render_progress_bar(progress: float, width: int = 40) -> str:
    progress = min(max(progress, 0.0), 1.0)
    filled = int(width * progress)
    if progress > 0 and filled == 0:
        filled = 1
    if progress >= 1:
        filled = width
    empty = width - filled
    bar = "█" * filled + "-" * empty
    return f"[{bar}]"


def _update_console_progress(message: str) -> None:
    logger.debug(message)


def _finish_console_progress() -> None:
    logger.debug("Progress finished")


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
        self._loss_fn = build_gan_loss(model_config.gan_loss, l1_weight=config.l1_weight)
        self._step = Pix2PixTrainingStep(
            generator=generator,
            discriminator=discriminator,
            opt_G=self._opt_G,
            opt_D=self._opt_D,
            scaler_G=self._scaler_G,
            scaler_D=self._scaler_D,
            loss_fn=self._loss_fn,
            device=device,
            amp_enabled=self._amp_enabled,
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
            gan_loss=model_config.gan_loss,
            l1_weight=config.l1_weight,
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
        logger.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)
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
                "Hyperparameters | l1_weight=%s | lr_g=%s | lr_d=%s | beta1=%s | beta2=%s",
                self.config.l1_weight,
                self.config.lr_g,
                self.config.lr_d,
                self.config.beta1,
                self.config.beta2,
            )

            training_status = {
                "last_checkpoint": (
                    Path(self.config.resume).name if self.config.resume else "none "
                )
            }

            metrics_path = self._run_paths.root / "metrics.csv"
            best_checkpoint_path: Path | None = None
            with open(metrics_path, "w", newline="", encoding="utf-8") as metrics_file:
                metrics_writer = csv.DictWriter(
                    metrics_file,
                    fieldnames=[
                        "epoch",
                        "loss_G_train",
                        "loss_D_train",
                        "loss_G_val",
                        "loss_D_val",
                    ],
                )
                metrics_writer.writeheader()

                for epoch in range(start_epoch, self.config.epochs):
                    logger.debug("Starting epoch %s", epoch)

                    epoch_metrics = self._train_epoch(
                        epoch, log_file, progress_tracker, training_status
                    )

                    logger.debug("Finished epoch %s", epoch)

                    if (epoch + 1) % self.config.checkpoint_rate == 0:
                        checkpoint_path = self._checkpoint_manager.save(epoch)
                        best_checkpoint_path = checkpoint_path
                        training_status["last_checkpoint"] = checkpoint_path.name
                        logger.info("Checkpoint saved to %s at epoch %s", checkpoint_path, epoch)
                        if reporter is not None:
                            reporter.on_checkpoint_saved(checkpoint_path, epoch)
                        if epoch == self.config.epochs - 1:
                            _update_console_progress(
                                f"{_render_progress_bar(1.0)} "
                                f"global {_color_progress(1.0)} | "
                                f"ep {epoch + 1}/{self.config.epochs} (100%) | "
                                f"b {len(self.train_loader)}/{len(self.train_loader)} | "
                                f"loss_G {epoch_metrics.loss_G:.4f} | "
                                f"loss_D {epoch_metrics.loss_D:.4f} | "
                                f"elapsed {_format_duration(time.time() - start_time)} | "
                                f"ETA 0s | "
                                f"ckpt {training_status['last_checkpoint']}"
                            )

                    val_metrics = None
                    if (epoch + 1) % self.config.validate_rate == 0:
                        val_metrics = self._validate(epoch, log_file)

                    metrics_writer.writerow(
                        {
                            "epoch": epoch,
                            "loss_G_train": f"{epoch_metrics.loss_G:.6f}",
                            "loss_D_train": f"{epoch_metrics.loss_D:.6f}",
                            "loss_G_val": f"{val_metrics.loss_G:.6f}" if val_metrics else "",
                            "loss_D_val": f"{val_metrics.loss_D:.6f}" if val_metrics else "",
                        }
                    )
                    metrics_file.flush()
                    if reporter is not None:
                        reporter.on_epoch_completed(epoch_metrics)

            if start_epoch < self.config.epochs:
                final_epoch = self.config.epochs - 1
                if (final_epoch + 1) % self.config.checkpoint_rate != 0:
                    checkpoint_path = self._checkpoint_manager.save(final_epoch)
                    best_checkpoint_path = checkpoint_path
                    training_status["last_checkpoint"] = checkpoint_path.name
                    logger.info(
                        "Final checkpoint saved to %s (epoch %s)",
                        checkpoint_path,
                        final_epoch,
                    )
                    if reporter is not None:
                        reporter.on_checkpoint_saved(checkpoint_path, final_epoch)

            _finish_console_progress()
            total_seconds = time.time() - start_time
            logger.info("Execution completed. Total time = %.2f seconds", total_seconds)
            final_epoch = max(start_epoch, self.config.epochs) - 1
            return TrainingResult(
                final_epoch=final_epoch,
                best_checkpoint_path=best_checkpoint_path,
            )
        finally:
            logger.removeHandler(file_handler)
            file_handler.close()
            logger.setLevel(old_level)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _train_epoch(
        self,
        epoch: int,
        log_file: Path,
        progress_tracker: _ProgressTracker,
        training_status: dict,
    ) -> EpochMetrics:
        self.generator.train()
        self.discriminator.train()

        total_loss_G = 0.0
        total_loss_D = 0.0
        num_batches = 0

        for i, (x, y) in enumerate(self.train_loader):
            x, y = x.to(self.device), y.to(self.device)
            step_losses = self._step.step(x, y)
            total_loss_G += step_losses.loss_G
            total_loss_D += step_losses.loss_D
            num_batches += 1

            progress, elapsed, eta, end_time = progress_tracker.calculate_progress(epoch, i)
            elapsed_str = _format_duration(elapsed)
            eta_str = _format_duration(eta)
            epoch_progress = (i + 1) / progress_tracker.total_batches

            console_message = (
                f"{_render_progress_bar(progress)} "
                f"global {_color_progress(progress)} | "
                f"ep {epoch + 1}/{progress_tracker.total_epochs} "
                f"({epoch_progress:.0%}) | "
                f"b {i + 1}/{progress_tracker.total_batches} | "
                f"loss_G {step_losses.loss_G:.4f} | "
                f"loss_D {step_losses.loss_D:.4f} | "
                f"elapsed {elapsed_str} | "
                f"ETA {eta_str} | "
                f"ckpt {training_status['last_checkpoint']}"
            )

            if i % self.config.log_rate == 0 or i == len(self.train_loader) - 1:
                _update_console_progress(console_message)

            if i % self.config.log_rate == 0:
                end_time_str = (
                    "warming up"
                    if end_time is None
                    else datetime.datetime.fromtimestamp(end_time).strftime("%Y-%m-%d %H:%M:%S")
                )
                logger.debug(
                    "[ep %s | b %s] loss_G: %.4f loss_D: %.4f - "
                    "%.2f%% | elapsed %s | ETA %s | end %s",
                    epoch,
                    i,
                    step_losses.loss_G,
                    step_losses.loss_D,
                    progress * 100,
                    elapsed_str,
                    eta_str,
                    end_time_str,
                )

        if num_batches == 0:
            raise RuntimeError("Training loader was empty; cannot compute epoch metrics.")
        return EpochMetrics(loss_G=total_loss_G / num_batches, loss_D=total_loss_D / num_batches)

    def _validate(self, epoch: int, log_file: Path) -> EpochMetrics:
        self.generator.eval()
        self.discriminator.eval()

        self._output_val_dir.mkdir(parents=True, exist_ok=True)

        total_loss_G = 0.0
        total_loss_D = 0.0
        count = 0

        with torch.no_grad():
            for i, (x, y) in enumerate(self.val_loader):
                x, y = x.to(self.device), y.to(self.device)

                with autocast(device_type=self.device.type, enabled=self._amp_enabled):
                    fake = self.generator(x)
                    D_real = self.discriminator(x, y)
                    D_fake = self.discriminator(x, fake)
                    loss_D = self._loss_fn.discriminator_loss(D_real, D_fake)
                    loss_G = self._loss_fn.generator_loss(D_fake, fake, y)

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

        return EpochMetrics(loss_G=avg_loss_G, loss_D=avg_loss_D)
