from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler

from virtual_staining.training.config import LossTermConfig
from virtual_staining.training.losses import LOSS_REGISTRY, Pix2PixLoss, StepLosses
from virtual_staining.training.steps import Pix2PixTrainingStep


class _TinyGen(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


class _TinyDisc(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.zeros(x.shape[0], 1, 1, 1) * self.scale


def _make_step() -> Pix2PixTrainingStep:
    gen = _TinyGen()
    disc = _TinyDisc()
    return Pix2PixTrainingStep(
        generator=gen,
        discriminator=disc,
        opt_G=optim.Adam(gen.parameters(), lr=1e-4),
        opt_D=optim.Adam(disc.parameters(), lr=1e-4),
        scaler_G=GradScaler(enabled=False),
        scaler_D=GradScaler(enabled=False),
        loss_fn=Pix2PixLoss(l1_weight=25.0),
        device=torch.device("cpu"),
        amp_enabled=False,
    )


# ---------------------------------------------------------------------------
# Pix2PixLoss
# ---------------------------------------------------------------------------


def test_discriminator_loss_non_negative() -> None:
    loss = Pix2PixLoss(l1_weight=25.0)
    val = loss.discriminator_loss(torch.zeros(2, 1, 1, 1), torch.zeros(2, 1, 1, 1))
    assert val.item() >= 0.0


def test_generator_loss_non_negative() -> None:
    loss = Pix2PixLoss(l1_weight=25.0)
    D_fake = torch.zeros(2, 1, 1, 1)
    val = loss.generator_loss(D_fake, torch.zeros(2, 3, 8, 8), torch.zeros(2, 3, 8, 8))
    assert val.item() >= 0.0


def test_l1_weight_scales_generator_loss() -> None:
    D_fake = torch.zeros(2, 1, 1, 1)
    fake = torch.ones(2, 3, 8, 8)
    real = torch.zeros(2, 3, 8, 8)
    loss_0 = Pix2PixLoss(l1_weight=0.0).generator_loss(D_fake, fake, real)
    loss_25 = Pix2PixLoss(l1_weight=25.0).generator_loss(D_fake, fake, real)
    assert loss_25.item() > loss_0.item()


def test_legacy_l1_weight_zero_removes_l1_contribution() -> None:
    D_fake = torch.zeros(2, 1, 1, 1)
    fake = torch.ones(2, 3, 8, 8)
    real = torch.zeros(2, 3, 8, 8)
    with_mismatch = Pix2PixLoss(l1_weight=0.0).generator_loss(D_fake, fake, real)
    without_mismatch = Pix2PixLoss(l1_weight=0.0).generator_loss(D_fake, real, real)
    assert with_mismatch.item() == without_mismatch.item()


def test_listed_zero_weight_registry_loss_is_inactive() -> None:
    term = LossTermConfig(name="ssim", weight=0.0, enabled=True)
    assert term.is_active is False


def test_enabled_alone_does_not_activate_registry_loss() -> None:
    term = LossTermConfig(name="ssim", weight=0.0, enabled=True)
    assert term.is_active is False


def test_disabled_nonzero_weight_registry_loss_is_inactive() -> None:
    term = LossTermConfig(name="ssim", weight=1.0, enabled=False)
    assert term.is_active is False


def test_loss_registry_entries_default_to_zero_weight() -> None:
    assert LOSS_REGISTRY
    assert all(entry.default_weight == 0.0 for entry in LOSS_REGISTRY.values())


# ---------------------------------------------------------------------------
# Pix2PixTrainingStep
# ---------------------------------------------------------------------------


def test_training_step_returns_finite_losses() -> None:
    step = _make_step()
    result = step.step(torch.randn(2, 3, 8, 8), torch.randn(2, 3, 8, 8))
    assert isinstance(result, StepLosses)
    assert math.isfinite(result.loss_G)
    assert math.isfinite(result.loss_D)


def test_training_step_runs_on_cpu() -> None:
    step = _make_step()
    result = step.step(torch.zeros(1, 3, 4, 4), torch.zeros(1, 3, 4, 4))
    assert result.loss_G >= 0.0
    assert result.loss_D >= 0.0
