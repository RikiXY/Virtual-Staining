from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn


@dataclass(frozen=True)
class LossRegistryEntry:
    name: str
    roles: tuple[Literal["generator", "discriminator"], ...]
    targets: tuple[Literal["image"], ...]
    default_weight: float = 0.0


LOSS_REGISTRY: dict[str, LossRegistryEntry] = {
    "ssim": LossRegistryEntry(name="ssim", roles=("generator",), targets=("image",)),
}


@dataclass
class StepLosses:
    loss_G: float
    loss_D: float


class Pix2PixLoss:
    """Computes Pix2Pix discriminator and generator losses."""

    def __init__(self, l1_weight: float = 25.0) -> None:
        self.l1_weight = l1_weight
        self._bce = nn.BCEWithLogitsLoss()
        self._l1 = nn.L1Loss()

    def discriminator_loss(
        self,
        D_real: torch.Tensor,
        D_fake: torch.Tensor,
    ) -> torch.Tensor:
        real_label = torch.ones_like(D_real)
        fake_label = torch.zeros_like(D_fake)
        return self._bce(D_real, real_label) + self._bce(D_fake, fake_label)

    def generator_loss(
        self,
        D_fake: torch.Tensor,
        fake: torch.Tensor,
        real: torch.Tensor,
    ) -> torch.Tensor:
        real_label = torch.ones_like(D_fake)
        return self._bce(D_fake, real_label) + self._l1(fake, real) * self.l1_weight


def build_gan_loss(name: Literal["bce"], l1_weight: float = 25.0) -> Pix2PixLoss:
    if name == "bce":
        return Pix2PixLoss(l1_weight=l1_weight)
    raise ValueError(f"Unsupported gan_loss: {name!r}")
