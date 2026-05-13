from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler

logger = logging.getLogger(__name__)


@dataclass
class CheckpointState:
    epoch: int
    generator_state_dict: dict[str, Any]
    discriminator_state_dict: dict[str, Any]
    opt_G_state_dict: dict[str, Any]
    opt_D_state_dict: dict[str, Any]
    scaler_G_state_dict: dict[str, Any]
    scaler_D_state_dict: dict[str, Any]


def _make_arch_metadata(
    generator: nn.Module,
    discriminator: nn.Module,
    *,
    model_name: str | None = None,
    gan_loss: str | None = None,
) -> dict[str, Any]:
    """Build the architecture dict saved inside every checkpoint."""
    return {
        "name": model_name,
        "gan_loss": gan_loss,
        "generator": {
            "class": type(generator).__name__,
            "in_channels": getattr(generator, "in_channels", None),
            "out_channels": getattr(generator, "out_channels", None),
            "base_channels": getattr(generator, "base_channels", None),
            "norm": getattr(generator, "norm", None),
            "dropout": getattr(generator, "dropout", None),
            "bilinear": getattr(generator, "bilinear", None),
        },
        "discriminator": {
            "class": type(discriminator).__name__,
            "in_channels": getattr(discriminator, "in_channels", None),
            "ndf": getattr(discriminator, "ndf", None),
            "norm": getattr(discriminator, "norm", None),
            "use_sigmoid": getattr(discriminator, "use_sigmoid", None),
        },
    }


def _check_arch_match(
    checkpoint_arch: dict[str, Any],
    generator: nn.Module,
    discriminator: nn.Module,
) -> None:
    """Raise ValueError if checkpoint architecture does not match the current models."""
    gen_arch = checkpoint_arch.get("generator", {})
    for key in ("in_channels", "out_channels", "base_channels", "norm", "dropout", "bilinear"):
        ckpt_val = gen_arch.get(key)
        curr_val = getattr(generator, key, None)
        if ckpt_val != curr_val:
            raise ValueError(
                f"Architecture mismatch for generator.{key}: "
                f"checkpoint has {ckpt_val!r}, current model has {curr_val!r}. "
                "Instantiate the model with the same parameters used during training."
            )

    disc_arch = checkpoint_arch.get("discriminator", {})
    for key in ("in_channels", "ndf", "norm", "use_sigmoid"):
        ckpt_val = disc_arch.get(key)
        curr_val = getattr(discriminator, key, None)
        if ckpt_val != curr_val:
            raise ValueError(
                f"Architecture mismatch for discriminator.{key}: "
                f"checkpoint has {ckpt_val!r}, current model has {curr_val!r}. "
                "Instantiate the model with the same parameters used during training."
            )


def _check_generator_arch(checkpoint_arch: dict[str, Any], generator: nn.Module) -> None:
    """Raise ValueError if the checkpoint's generator architecture does not match."""
    gen_arch = checkpoint_arch.get("generator", {})
    for key in ("in_channels", "out_channels", "base_channels", "norm", "dropout", "bilinear"):
        ckpt_val = gen_arch.get(key)
        curr_val = getattr(generator, key, None)
        if ckpt_val != curr_val:
            raise ValueError(
                f"Architecture mismatch for generator.{key}: "
                f"checkpoint has {ckpt_val!r}, inference model has {curr_val!r}. "
                "Instantiate the generator with the same parameters used during training."
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
        model_name: str | None = None,
        gan_loss: str | None = None,
        l1_weight: float | None = None,
        lr_g: float | None = None,
        lr_d: float | None = None,
        beta1: float | None = None,
        beta2: float | None = None,
        batch_size: int | None = None,
        num_workers: int | None = None,
        dataset_root: str | None = None,
    ) -> None:
        self.checkpoints_dir = checkpoints_dir
        self.generator = generator
        self.discriminator = discriminator
        self.opt_G = opt_G
        self.opt_D = opt_D
        self.scaler_G = scaler_G
        self.scaler_D = scaler_D
        self.image_size = image_size
        self.device = device
        self.model_name = model_name
        self.gan_loss = gan_loss
        self.l1_weight = l1_weight
        self.lr_g = lr_g
        self.lr_d = lr_d
        self.beta1 = beta1
        self.beta2 = beta2
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_root = dataset_root

    def save(self, epoch: int) -> Path:
        """Save a full training checkpoint. Returns the path."""
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        path = self.checkpoints_dir / f"ep{epoch:03d}.pth"
        checkpoint = {
            "schema_version": 1,
            "epoch": epoch,
            "architecture": _make_arch_metadata(
                self.generator,
                self.discriminator,
                model_name=self.model_name,
                gan_loss=self.gan_loss,
            ),
            "generator_state_dict": self.generator.state_dict(),
            "discriminator_state_dict": self.discriminator.state_dict(),
            "optimizerG_state_dict": self.opt_G.state_dict(),
            "optimizerD_state_dict": self.opt_D.state_dict(),
            "scalerG_state_dict": self.scaler_G.state_dict(),
            "scalerD_state_dict": self.scaler_D.state_dict(),
            "l1_weight": self.l1_weight,
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
        """Load a training checkpoint. Returns the start_epoch for resuming."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        stored_image_size = checkpoint.get("image_size")
        if stored_image_size is not None and tuple(stored_image_size) != tuple(self.image_size):
            raise ValueError(
                "Image size mismatch between checkpoint and resumed training. "
                f"Checkpoint image_size={tuple(stored_image_size)}, "
                f"current image_size={tuple(self.image_size)}."
            )

        arch = checkpoint.get("architecture")
        if arch is None:
            raise ValueError(
                f"Checkpoint '{path}' has no architecture metadata. "
                "Only checkpoints saved with the current version are supported."
            )
        _check_arch_match(arch, self.generator, self.discriminator)

        self.generator.load_state_dict(checkpoint["generator_state_dict"])
        self.discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
        self.opt_G.load_state_dict(checkpoint["optimizerG_state_dict"])
        self.opt_D.load_state_dict(checkpoint["optimizerD_state_dict"])
        self.scaler_G.load_state_dict(checkpoint["scalerG_state_dict"])
        self.scaler_D.load_state_dict(checkpoint["scalerD_state_dict"])

        start_epoch: int = checkpoint["epoch"] + 1
        logger.info("Checkpoint loaded from %s, resuming at epoch %s", path, start_epoch)
        return start_epoch

    def latest(self) -> Path | None:
        """Return the path to the most recent ep*.pth file, or None."""
        candidates = sorted(self.checkpoints_dir.glob("ep*.pth"))
        return candidates[-1] if candidates else None
