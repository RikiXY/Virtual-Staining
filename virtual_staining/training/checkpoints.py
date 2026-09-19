from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler

from virtual_staining.checkpoint_contract import (
    CHECKPOINT_FORMAT_VERSION,
    NORMALIZATION_CONTRACT,
    check_discriminator_arch,
    check_generator_arch,
    make_arch_metadata,
    validate_checkpoint_metadata,
)
from virtual_staining.checkpoint_selection import latest_checkpoint_path
from virtual_staining.training.runtime import TrainingMethodRuntime

logger = logging.getLogger(__name__)

METHOD_CHECKPOINT_FORMAT_VERSION = 4


class MethodCheckpointManager:
    """Persists a method-owned state mapping without knowing its model topology."""

    def __init__(
        self,
        checkpoints_dir: Path,
        runtime: TrainingMethodRuntime,
        schedulers: tuple[Any | None, ...],
        *,
        image_size: tuple[int, int],
        device: torch.device,
        resolved_config: dict[str, object],
    ) -> None:
        self.checkpoints_dir = checkpoints_dir
        self.runtime = runtime
        self.schedulers = schedulers
        self.image_size = image_size
        self.device = device
        self.resolved_config = resolved_config

    def save(self, epoch: int) -> Path:
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        path = self.checkpoints_dir / f"ep{epoch:03d}.pth"
        runtime = self.runtime
        checkpoint = {
            "format_version": METHOD_CHECKPOINT_FORMAT_VERSION,
            "epoch": epoch,
            "method": {
                "name": runtime.name,
                "pairing": runtime.pairing,
                "components": dict(runtime.component_metadata()),
            },
            "normalization_contract": NORMALIZATION_CONTRACT,
            "image_size": self.image_size,
            "resolved_config": self.resolved_config,
            "method_state": runtime.state_dict(),
            "scheduler_state_dicts": [
                scheduler.state_dict() if scheduler is not None else None
                for scheduler in self.schedulers
            ],
        }
        torch.save(checkpoint, path)
        logger.info("Checkpoint saved: %s", path)
        return path

    def load(self, path: Path) -> int:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        stored_size = checkpoint.get("image_size")
        if stored_size is not None and tuple(stored_size) != tuple(self.image_size):
            raise ValueError(
                "Image size mismatch between checkpoint and resumed training. "
                f"Checkpoint image_size={tuple(stored_size)}, current image_size={self.image_size}."
            )
        version = checkpoint.get("format_version")
        if version == CHECKPOINT_FORMAT_VERSION and self.runtime.name == "pix2pix":
            validate_checkpoint_metadata(checkpoint, path)
            self.runtime.load_legacy_v3(checkpoint)
            return int(checkpoint["epoch"]) + 1
        if version != METHOD_CHECKPOINT_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported checkpoint format version {version!r}; expected v4 method checkpoint"
            )
        method = checkpoint.get("method")
        if not isinstance(method, dict) or method.get("name") != self.runtime.name:
            raise ValueError(
                f"Checkpoint method {getattr(method, 'get', lambda *_: None)('name')!r} "
                f"does not match configured method {self.runtime.name!r}"
            )
        state = checkpoint.get("method_state")
        if not isinstance(state, dict):
            raise ValueError("Checkpoint method_state must be a mapping")
        self.runtime.load_state_dict(state)
        scheduler_states = checkpoint.get("scheduler_state_dicts", [])
        for scheduler, scheduler_state in zip(self.schedulers, scheduler_states, strict=False):
            if scheduler is not None and scheduler_state is not None:
                scheduler.load_state_dict(scheduler_state)
        return int(checkpoint["epoch"]) + 1

    def latest(self) -> Path | None:
        return latest_checkpoint_path(self.checkpoints_dir)


class CheckpointManager:
    """Manages saving and loading of Pix2Pix training checkpoints."""

    def __init__(
        self,
        checkpoints_dir: Path,
        generator: nn.Module,
        discriminator: nn.Module,
        opt_G: optim.Optimizer,
        opt_D: optim.Optimizer,
        scaler_G: GradScaler,
        scaler_D: GradScaler,
        image_size: tuple[int, int],
        device: torch.device,
        *,
        target_modality: str | None = None,
        scheduler_G: optim.lr_scheduler.LRScheduler
        | optim.lr_scheduler.ReduceLROnPlateau
        | None = None,
        scheduler_D: optim.lr_scheduler.LRScheduler
        | optim.lr_scheduler.ReduceLROnPlateau
        | None = None,
        lr_g: float | None = None,
        lr_d: float | None = None,
        beta1: float | None = None,
        beta2: float | None = None,
        batch_size: int | None = None,
        num_workers: int | None = None,
        dataset_root: str | None = None,
    ) -> None:
        self.checkpoints_dir, self.generator, self.discriminator = (
            checkpoints_dir,
            generator,
            discriminator,
        )
        self.opt_G, self.opt_D, self.scaler_G, self.scaler_D = opt_G, opt_D, scaler_G, scaler_D
        self.scheduler_G, self.scheduler_D = scheduler_G, scheduler_D
        self.image_size, self.device, self.target_modality = image_size, device, target_modality
        self.lr_g, self.lr_d, self.beta1, self.beta2 = lr_g, lr_d, beta1, beta2
        self.batch_size, self.num_workers, self.dataset_root = batch_size, num_workers, dataset_root

    def save(self, epoch: int) -> Path:
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        path = self.checkpoints_dir / f"ep{epoch:03d}.pth"
        checkpoint = {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "epoch": epoch,
            "architecture": make_arch_metadata(
                self.generator, self.discriminator, target_modality=self.target_modality
            ),
            "normalization_contract": NORMALIZATION_CONTRACT,
            "generator_state_dict": self.generator.state_dict(),
            "discriminator_state_dict": self.discriminator.state_dict(),
            "optimizerG_state_dict": self.opt_G.state_dict(),
            "optimizerD_state_dict": self.opt_D.state_dict(),
            "scalerG_state_dict": self.scaler_G.state_dict(),
            "scalerD_state_dict": self.scaler_D.state_dict(),
            "schedulerG_state_dict": (
                self.scheduler_G.state_dict() if self.scheduler_G is not None else None
            ),
            "schedulerD_state_dict": (
                self.scheduler_D.state_dict() if self.scheduler_D is not None else None
            ),
            "lr_g": self.lr_g,
            "lr_d": self.lr_d,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "image_size": self.image_size,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "dataset_root": self.dataset_root,
        }
        torch.save(checkpoint, path)
        logger.info("Checkpoint saved: %s", path)
        return path

    def load(self, path: Path) -> int:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        stored_image_size = checkpoint.get("image_size")
        if stored_image_size is not None and tuple(stored_image_size) != tuple(self.image_size):
            raise ValueError(
                "Image size mismatch between checkpoint and resumed training. "
                f"Checkpoint image_size={tuple(stored_image_size)}, "
                f"current image_size={tuple(self.image_size)}."
            )
        arch = validate_checkpoint_metadata(checkpoint, path)
        check_generator_arch(arch, self.generator, target_modality=self.target_modality)
        check_discriminator_arch(arch, self.discriminator)
        self.generator.load_state_dict(checkpoint["generator_state_dict"])
        self.discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
        self.opt_G.load_state_dict(checkpoint["optimizerG_state_dict"])
        self.opt_D.load_state_dict(checkpoint["optimizerD_state_dict"])
        self.scaler_G.load_state_dict(checkpoint["scalerG_state_dict"])
        self.scaler_D.load_state_dict(checkpoint["scalerD_state_dict"])
        if self.scheduler_G is not None and checkpoint.get("schedulerG_state_dict") is not None:
            self.scheduler_G.load_state_dict(checkpoint["schedulerG_state_dict"])
        if self.scheduler_D is not None and checkpoint.get("schedulerD_state_dict") is not None:
            self.scheduler_D.load_state_dict(checkpoint["schedulerD_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        logger.info("Checkpoint loaded from %s, resuming at epoch %s", path, start_epoch)
        return start_epoch

    def latest(self) -> Path | None:
        return latest_checkpoint_path(self.checkpoints_dir)
