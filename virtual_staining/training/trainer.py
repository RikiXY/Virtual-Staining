from __future__ import annotations

import csv
import datetime
import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.amp import GradScaler, autocast
from torchvision.utils import save_image

from virtual_staining.training.config import TrainingConfig
from virtual_staining.training.results import EpochMetrics
from virtual_staining.utils.cli import print_info, print_section, style
from virtual_staining.utils.env import collect_environment


# ---------------------------------------------------------------------------
# Module-level helpers (private to this module)
# ---------------------------------------------------------------------------

def _is_amp_enabled(device: torch.device) -> bool:
    return isinstance(device, torch.device) and device.type == "cuda"


def _log_message(
    message: str,
    log_file: Path,
    show_time: bool = True,
    use_stdout: bool = True,
) -> None:
    if show_time:
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message = f"[{now_str}] {message}"
    with open(log_file, "a+") as f:
        f.write(message + "\n")
    if use_stdout:
        print(message)


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


def _get_first_pair_size(dataset) -> dict | None:
    if len(dataset) == 0:
        return None
    source_path, target_path = dataset.pairs[0]
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


def _save_run_config(run_config: dict, run_root: Path) -> Path:
    config_path = run_root / "run_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=4, default=str)
    return config_path


def _log_run_header(log_file: Path, run_config: dict) -> None:
    _log_message("=" * 80, log_file, show_time=False, use_stdout=False)
    _log_message("RUN CONFIGURATION", log_file, show_time=False, use_stdout=False)
    _log_message("=" * 80, log_file, show_time=False, use_stdout=False)
    for key, value in run_config.items():
        _log_message(f"{key}: {value}", log_file, show_time=False, use_stdout=False)
    _log_message("=" * 80, log_file, show_time=False, use_stdout=False)


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
    text = f"{progress:.2%}"
    if progress < 0.33:
        return style(text, "yellow")
    if progress < 0.66:
        return style(text, "cyan")
    if progress < 0.90:
        return style(text, "blue")
    return style(text, "green")


def _render_progress_bar(progress: float, width: int = 40) -> str:
    progress = min(max(progress, 0.0), 1.0)
    filled = int(width * progress)
    if progress > 0 and filled == 0:
        filled = 1
    if progress >= 1:
        filled = width
    empty = width - filled
    bar = "█" * filled + "-" * empty
    return f"[{style(bar, 'green')}]"


def _update_console_progress(message: str) -> None:
    try:
        terminal_width = os.get_terminal_size().columns
    except OSError:
        terminal_width = 140
    clean_message = message[: terminal_width - 1]
    print("\r" + clean_message.ljust(terminal_width - 1), end="", flush=True)


def _finish_console_progress() -> None:
    print()


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

        step_duration = now - self.last_step_time  # type: ignore[operator]
        self.last_step_time = now

        if completed_since_start > self.warmup_batches:
            self.step_durations.append(step_duration)
            if len(self.step_durations) > self.max_history:
                self.step_durations.pop(0)

        total_elapsed_time = now - self.start_time  # type: ignore[operator]
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
        generator: nn.Module,
        discriminator: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        device: torch.device,
    ) -> None:
        self.config = config
        self.generator = generator
        self.discriminator = discriminator
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

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
        self._bce_loss: nn.Module = nn.BCEWithLogitsLoss()
        self._l1_loss: nn.Module = nn.L1Loss()

        self._logs_dir = config.run_root / "logs"
        self._checkpoints_dir = config.run_root / "checkpoints"
        self._output_val_dir = config.run_root / "output_val"
        self._output_train_dir = config.run_root / "output_train"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self, seed: int) -> None:
        """Run the full training loop."""
        start_time = time.time()

        for d in [
            self._logs_dir,
            self._checkpoints_dir,
            self._output_val_dir,
            self._output_train_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)

        env = collect_environment()
        self.config.to_yaml(self.config.run_root / "config.yaml")
        with open(self.config.run_root / "environment.json", "w", encoding="utf-8") as f:
            json.dump(env, f, indent=2, default=str)

        timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_file = self._logs_dir / f"Log-{timestamp_str}.txt"
        if log_file.exists():
            log_file.unlink()

        device_name = (
            torch.cuda.get_device_name(self.device)
            if self.device.type == "cuda"
            else "CPU"
        )
        _log_message(f"Seed set to {seed}", log_file, use_stdout=False)
        _log_message(f"Device: {self.device} ({device_name})", log_file)

        run_config = self._build_run_config(seed, timestamp_str, log_file, env)
        config_path = _save_run_config(run_config, self.config.run_root)
        _log_run_header(log_file, run_config)
        _log_message(f"Run config saved to {config_path}", log_file, use_stdout=False)

        for f in self._output_train_dir.iterdir():
            if f.is_file():
                f.unlink()

        start_epoch = 0
        if self.config.resume is not None:
            if Path(self.config.resume).exists():
                start_epoch = self.load_checkpoint(Path(self.config.resume), log_file)
            else:
                _log_message(
                    f"WARNING - Checkpoint not found: {self.config.resume}",
                    log_file,
                    use_stdout=False,
                )
        else:
            _log_message("Training started from scratch", log_file, use_stdout=False)

        print_section("Pix2Pix training")
        print_info("Run root", str(self.config.run_root))
        print_info("Dataset root", str(self.config.dataset_root))
        print_info("Device", str(self.device))
        print_info("Epochs", str(self.config.epochs))
        print_info("Start epoch", str(start_epoch))
        print_info("Train samples", str(len(self.train_loader.dataset)))  # type: ignore[arg-type]
        print_info("Validation samples", str(len(self.val_loader.dataset)))  # type: ignore[arg-type]
        print_info("Train batches/epoch", str(len(self.train_loader)))
        print_info("Validation batches", str(len(self.val_loader)))
        print_info("Detailed log", str(log_file))
        print()
        print(style("Training progress:", "bold", "cyan"))

        progress_tracker = _ProgressTracker(
            total_epochs=self.config.epochs,
            total_batches=len(self.train_loader),
            start_epoch=start_epoch,
            warmup_batches=max(10, self.config.log_rate),
        )
        progress_tracker.start()

        _log_message("Training started", log_file, use_stdout=False)
        _log_message(
            f"Hyperparameters | l1_weight={self.config.l1_weight} | "
            f"lr_g={self.config.lr_g} | lr_d={self.config.lr_d} | "
            f"beta1={self.config.beta1} | beta2={self.config.beta2}",
            log_file,
            use_stdout=False,
        )

        training_status = {
            "last_checkpoint": (
                Path(self.config.resume).name if self.config.resume else "none "
            )
        }

        metrics_path = self.config.run_root / "metrics.csv"
        with open(metrics_path, "w", newline="", encoding="utf-8") as metrics_file:
            metrics_writer = csv.DictWriter(
                metrics_file,
                fieldnames=["epoch", "loss_G_train", "loss_D_train", "loss_G_val", "loss_D_val"],
            )
            metrics_writer.writeheader()

            for epoch in range(start_epoch, self.config.epochs):
                _log_message(f"Starting epoch {epoch}", log_file, use_stdout=False)

                epoch_metrics = self._train_epoch(
                    epoch, log_file, progress_tracker, training_status
                )

                _log_message(f"Finished epoch {epoch}", log_file, use_stdout=False)

                if (epoch + 1) % self.config.checkpoint_rate == 0:
                    checkpoint_path = self._checkpoints_dir / f"ep{epoch:03d}.pth"
                    self.save_checkpoint(checkpoint_path, epoch)
                    training_status["last_checkpoint"] = checkpoint_path.name
                    _log_message(
                        f"Checkpoint saved to {checkpoint_path} at epoch {epoch}",
                        log_file,
                        use_stdout=False,
                    )
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

                metrics_writer.writerow({
                    "epoch": epoch,
                    "loss_G_train": f"{epoch_metrics.loss_G:.6f}",
                    "loss_D_train": f"{epoch_metrics.loss_D:.6f}",
                    "loss_G_val": f"{val_metrics.loss_G:.6f}" if val_metrics else "",
                    "loss_D_val": f"{val_metrics.loss_D:.6f}" if val_metrics else "",
                })
                metrics_file.flush()

        _finish_console_progress()
        total_seconds = time.time() - start_time
        _log_message(
            f"Execution completed. Total time = {total_seconds:.2f} seconds",
            log_file,
            use_stdout=False,
        )

    def save_checkpoint(self, checkpoint_path: Path, epoch: int) -> None:
        checkpoint = {
            "epoch": epoch,
            "generator_state_dict": self.generator.state_dict(),
            "discriminator_state_dict": self.discriminator.state_dict(),
            "optimizerG_state_dict": self._opt_G.state_dict(),
            "optimizerD_state_dict": self._opt_D.state_dict(),
            "scalerG_state_dict": self._scaler_G.state_dict(),
            "scalerD_state_dict": self._scaler_D.state_dict(),
            "l1_weight": self.config.l1_weight,
            "lr_g": self.config.lr_g,
            "lr_d": self.config.lr_d,
            "beta1": self.config.beta1,
            "beta2": self.config.beta2,
            "image_size": self.config.image_size,
            "batch_size": self.config.batch_size,
            "num_workers": self.config.num_workers,
            "dataset_root": str(self.config.dataset_root),
        }
        torch.save(checkpoint, checkpoint_path)

    def load_checkpoint(
        self, checkpoint_path: Path, log_file: Path | None = None
    ) -> int:
        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )

        checkpoint_image_size = checkpoint.get("image_size")
        if checkpoint_image_size is not None:
            checkpoint_image_size = tuple(checkpoint_image_size)
            if checkpoint_image_size != tuple(self.config.image_size):
                raise ValueError(
                    "Image size mismatch between checkpoint and resumed training. "
                    f"Checkpoint image_size={checkpoint_image_size}, "
                    f"current image_size={tuple(self.config.image_size)}."
                )

        self.generator.load_state_dict(checkpoint["generator_state_dict"])
        self.discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
        self._opt_G.load_state_dict(checkpoint["optimizerG_state_dict"])
        self._opt_D.load_state_dict(checkpoint["optimizerD_state_dict"])
        self._scaler_G.load_state_dict(checkpoint["scalerG_state_dict"])
        self._scaler_D.load_state_dict(checkpoint["scalerD_state_dict"])

        # Resume from the next epoch to avoid repeating the completed one.
        start_epoch: int = checkpoint["epoch"] + 1
        if log_file is not None:
            _log_message(
                f"Checkpoint loaded from {checkpoint_path}, resuming at epoch {start_epoch}",
                log_file,
                use_stdout=False,
            )
        return start_epoch

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

        last_loss_G: float | None = None
        last_loss_D: float | None = None

        for i, (x, y) in enumerate(self.train_loader):
            x, y = x.to(self.device), y.to(self.device)

            # --- Discriminator step ---
            with autocast(device_type=self.device.type, enabled=self._amp_enabled):
                # .detach() prevents gradients flowing back into G during D's update.
                fake = self.generator(x).detach()
                D_real = self.discriminator(x, y)
                D_fake = self.discriminator(x, fake)
                real_label = torch.ones_like(D_real, device=self.device)
                fake_label = torch.zeros_like(D_fake, device=self.device)
                loss_D = (
                    self._bce_loss(D_real, real_label)
                    + self._bce_loss(D_fake, fake_label)
                )

            self._opt_D.zero_grad()
            self._scaler_D.scale(loss_D).backward()
            self._scaler_D.step(self._opt_D)
            self._scaler_D.update()

            # --- Generator step ---
            with autocast(device_type=self.device.type, enabled=self._amp_enabled):
                fake = self.generator(x)
                D_fake = self.discriminator(x, fake)
                loss_G = (
                    self._bce_loss(D_fake, real_label)
                    + self._l1_loss(fake, y) * self.config.l1_weight
                )

            self._opt_G.zero_grad()
            self._scaler_G.scale(loss_G).backward()
            self._scaler_G.step(self._opt_G)
            self._scaler_G.update()

            last_loss_G = loss_G.item()
            last_loss_D = loss_D.item()

            progress, elapsed, eta, end_time = progress_tracker.calculate_progress(
                epoch, i
            )
            elapsed_str = _format_duration(elapsed)
            eta_str = _format_duration(eta)
            epoch_progress = (i + 1) / progress_tracker.total_batches

            console_message = (
                f"{_render_progress_bar(progress)} "
                f"global {_color_progress(progress)} | "
                f"ep {epoch + 1}/{progress_tracker.total_epochs} "
                f"({epoch_progress:.0%}) | "
                f"b {i + 1}/{progress_tracker.total_batches} | "
                f"loss_G {loss_G.item():.4f} | "
                f"loss_D {loss_D.item():.4f} | "
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
                    else datetime.datetime.fromtimestamp(end_time).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                )
                _log_message(
                    f"[ep {epoch} | b {i}] loss_G: {loss_G.item():.4f} "
                    f"loss_D: {loss_D.item():.4f} - "
                    f"{progress:.2%} | elapsed {elapsed_str} | "
                    f"ETA {eta_str} | end {end_time_str}",
                    log_file,
                    use_stdout=False,
                )

        if last_loss_G is None or last_loss_D is None:
            raise RuntimeError("Training loader was empty; cannot compute epoch metrics.")
        return EpochMetrics(loss_G=last_loss_G, loss_D=last_loss_D)

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
                    real_label = torch.ones_like(D_real, device=self.device)
                    fake_label = torch.zeros_like(D_fake, device=self.device)
                    loss_D = (
                        self._bce_loss(D_real, real_label)
                        + self._bce_loss(D_fake, fake_label)
                    )
                    loss_G = (
                        self._bce_loss(D_fake, real_label)
                        + self._l1_loss(fake, y) * self.config.l1_weight
                    )

                total_loss_D += loss_D.item()
                total_loss_G += loss_G.item()
                count += 1

                if i < 5:
                    _save_images(self._output_val_dir, x[0], fake[0], y[0], epoch, i)

        avg_loss_G = total_loss_G / count if count > 0 else 0.0
        avg_loss_D = total_loss_D / count if count > 0 else 0.0

        _log_message(
            f"[Epoch {epoch}] Validation: loss_G={avg_loss_G:.4f} loss_D={avg_loss_D:.4f}",
            log_file,
            use_stdout=False,
        )

        return EpochMetrics(loss_G=avg_loss_G, loss_D=avg_loss_D)

    def _build_run_config(
        self, seed: int, timestamp_str: str, log_file: Path, environment: dict
    ) -> dict:
        return {
            "timestamp": timestamp_str,
            "dataset_root": str(self.config.dataset_root),
            "run_root": str(self.config.run_root),
            "logs_dir": str(self._logs_dir),
            "checkpoints_dir": str(self._checkpoints_dir),
            "output_train_dir": str(self._output_train_dir),
            "output_val_dir": str(self._output_val_dir),
            "train_dir": str(Path(self.config.dataset_root) / "dataset_train"),
            "val_dir": str(Path(self.config.dataset_root) / "dataset_val"),
            "seed": seed,
            "device": str(self.device),
            "epochs": self.config.epochs,
            "batch_size": self.config.batch_size,
            "num_workers": self.config.num_workers,
            "image_size_resize": list(self.config.image_size),
            "log_rate": self.config.log_rate,
            "checkpoint_rate": self.config.checkpoint_rate,
            "validate_rate": self.config.validate_rate,
            "resume_checkpoint": str(self.config.resume) if self.config.resume else None,
            "l1_weight": self.config.l1_weight,
            "lr_g": self.config.lr_g,
            "lr_d": self.config.lr_d,
            "beta1": self.config.beta1,
            "beta2": self.config.beta2,
            "train_samples": len(self.train_loader.dataset),  # type: ignore[arg-type]
            "val_samples": len(self.val_loader.dataset),  # type: ignore[arg-type]
            "train_batches": len(self.train_loader),
            "val_batches": len(self.val_loader),
            "first_train_pair_info": _get_first_pair_size(self.train_loader.dataset),
            "first_val_pair_info": _get_first_pair_size(self.val_loader.dataset),
            "detailed_log": str(log_file),
            "environment": environment,
        }
