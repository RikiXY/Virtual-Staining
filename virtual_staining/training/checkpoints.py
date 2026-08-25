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
    make_arch_metadata,
    validate_checkpoint_metadata,
)

logger = logging.getLogger(__name__)


def _check_arch_match(
    checkpoint_arch: dict[str, Any],
    generator: nn.Module,
    discriminator: nn.Module,
    *,
    target_modality: str | None = None,
) -> None:
    gen_arch = checkpoint_arch.get("generator", {})
    if target_modality is not None and gen_arch.get("target_modality") != target_modality:
        raise ValueError("Checkpoint target modality does not match current model.")
    current = getattr(generator, "unet", generator)
    for key in (
        "class",
        "architecture",
        "input_names",
        "in_channels",
        "out_channels",
        "base_channels",
        "norm",
        "dropout",
        "bilinear",
    ):
        ckpt_val = gen_arch.get(key)
        if key == "class":
            curr_val = type(generator).__name__
        elif key == "architecture":
            curr_val = "concat_unet"
        elif key == "input_names":
            curr_val = list(getattr(generator, "input_names", ()))
        else:
            curr_val = getattr(current, key, 3 if key == "out_channels" else None)
        if ckpt_val != curr_val:
            raise ValueError(
                f"Architecture mismatch for generator.{key}: checkpoint has "
                f"{ckpt_val!r}, current model has {curr_val!r}."
            )
    disc_arch = checkpoint_arch.get("discriminator", {})
    for key in ("class", "in_channels", "ndf", "norm", "use_sigmoid"):
        ckpt_val = disc_arch.get(key)
        curr_val = (
            type(discriminator).__name__ if key == "class" else getattr(discriminator, key, None)
        )
        if ckpt_val != curr_val:
            raise ValueError(
                f"Architecture mismatch for discriminator.{key}: checkpoint has "
                f"{ckpt_val!r}, current model has {curr_val!r}."
            )


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
        _check_arch_match(
            arch, self.generator, self.discriminator, target_modality=self.target_modality
        )
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
        candidates = sorted(self.checkpoints_dir.glob("ep*.pth"))
        return candidates[-1] if candidates else None
