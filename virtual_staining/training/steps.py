from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast

from virtual_staining.training.loss_config import LossTermConfig
from virtual_staining.training.losses import (
    ConfiguredLossEvaluator,
    LossEvaluationContext,
    StepLosses,
)


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
        device: torch.device,
        amp_enabled: bool,
        generator_loss_terms: tuple[LossTermConfig, ...] = (),
        discriminator_loss_terms: tuple[LossTermConfig, ...] = (),
    ) -> None:
        self.generator = generator
        self.discriminator = discriminator
        self.opt_G = opt_G
        self.opt_D = opt_D
        self.scaler_G = scaler_G
        self.scaler_D = scaler_D
        self.device = device
        self.amp_enabled = amp_enabled
        self.generator_loss_terms = generator_loss_terms
        self.discriminator_loss_terms = discriminator_loss_terms
        self.loss_evaluator = ConfiguredLossEvaluator(
            generator_terms=generator_loss_terms,
            discriminator_terms=discriminator_loss_terms,
        )

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
            context = LossEvaluationContext(epoch=epoch, global_step=global_step)
            discriminator_loss = self.loss_evaluator.discriminator_total(
                discriminator_real=D_real,
                discriminator_fake=D_fake,
                context=context,
            )
            loss_D = discriminator_loss.total
            component_raw = dict(discriminator_loss.raw)
            component_weighted = dict(discriminator_loss.weighted)
            component_current_weight = dict(discriminator_loss.current_weight)

        self.opt_D.zero_grad()
        self.scaler_D.scale(loss_D).backward()
        self.scaler_D.step(self.opt_D)
        self.scaler_D.update()

        with autocast(device_type=self.device.type, enabled=self.amp_enabled):
            fake = self.generator(x)
            D_fake = self.discriminator(x, fake)
            context = LossEvaluationContext(epoch=epoch, global_step=global_step, masks=masks)
            generator_loss = self.loss_evaluator.generator_total(
                prediction=fake,
                target=y,
                discriminator_fake=D_fake,
                context=context,
            )
            loss_G = generator_loss.total
            component_raw.update(generator_loss.raw)
            component_weighted.update(generator_loss.weighted)
            component_current_weight.update(generator_loss.current_weight)

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
