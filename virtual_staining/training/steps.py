from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast

from virtual_staining.training.config import LossTermConfig
from virtual_staining.training.losses import Pix2PixLoss, StepLosses, evaluate_loss_term


class Pix2PixTrainingStep:
    """Executes one discriminator + one generator update for a single batch."""

    def __init__(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        opt_G: optim.Optimizer,
        opt_D: optim.Optimizer,
        scaler_G: GradScaler,
        scaler_D: GradScaler,
        loss_fn: Pix2PixLoss,
        device: torch.device,
        amp_enabled: bool,
        generator_loss_terms: tuple[LossTermConfig, ...] = (),
    ) -> None:
        self.generator = generator
        self.discriminator = discriminator
        self.opt_G = opt_G
        self.opt_D = opt_D
        self.scaler_G = scaler_G
        self.scaler_D = scaler_D
        self.loss_fn = loss_fn
        self.device = device
        self.amp_enabled = amp_enabled
        self.generator_loss_terms = generator_loss_terms

    def step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        *,
        epoch: int = 0,
        global_step: int | None = None,
        masks: dict[str, torch.Tensor] | None = None,
    ) -> StepLosses:
        """Run one discriminator step and one generator step. Returns scalar losses."""
        with autocast(device_type=self.device.type, enabled=self.amp_enabled):
            # .detach() prevents gradients flowing back into G during D's update.
            fake = self.generator(x).detach()
            D_real = self.discriminator(x, y)
            D_fake = self.discriminator(x, fake)
            loss_D = self.loss_fn.discriminator_loss(D_real, D_fake)

        self.opt_D.zero_grad()
        self.scaler_D.scale(loss_D).backward()
        self.scaler_D.step(self.opt_D)
        self.scaler_D.update()

        with autocast(device_type=self.device.type, enabled=self.amp_enabled):
            fake = self.generator(x)
            D_fake = self.discriminator(x, fake)
            loss_G = self.loss_fn.generator_loss(D_fake, fake, y)
            component_raw: dict[str, float] = {}
            component_weighted: dict[str, float] = {}
            component_current_weight: dict[str, float] = {}
            for term in self.generator_loss_terms:
                result = evaluate_loss_term(
                    term,
                    fake,
                    y,
                    epoch=epoch,
                    global_step=global_step,
                    masks=masks,
                )
                loss_G = loss_G + result.weighted
                component_raw[result.name] = float(result.raw.detach().item())
                component_weighted[result.name] = float(result.weighted.detach().item())
                component_current_weight[result.name] = result.current_weight

        self.opt_G.zero_grad()
        self.scaler_G.scale(loss_G).backward()
        self.scaler_G.step(self.opt_G)
        self.scaler_G.update()

        return StepLosses(
            loss_G=loss_G.item(),
            loss_D=loss_D.item(),
            raw=component_raw,
            weighted=component_weighted,
            current_weight=component_current_weight,
        )
