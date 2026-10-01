from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast

from virtual_staining.training.losses import (
    ConfiguredLossEvaluator,
    LossEvaluationContext,
    StepLosses,
)

if TYPE_CHECKING:
    from virtual_staining.training.benchmarking import TrainingBenchmarkRecorder


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
        loss_evaluator: ConfiguredLossEvaluator,
        benchmark_recorder: TrainingBenchmarkRecorder | None = None,
    ) -> None:
        self.generator = generator
        self.discriminator = discriminator
        self.opt_G = opt_G
        self.opt_D = opt_D
        self.scaler_G = scaler_G
        self.scaler_D = scaler_D
        self.device = device
        self.amp_enabled = amp_enabled
        self.benchmark_recorder = benchmark_recorder
        self.loss_evaluator = loss_evaluator

    def step(
        self,
        inputs: Mapping[str, torch.Tensor],
        targets: Mapping[str, torch.Tensor],
        *,
        epoch: int = 0,
        global_step: int | None = None,
        masks: Mapping[str, Mapping[str, torch.Tensor]] | None = None,
    ) -> StepLosses:
        """Update the joint discriminator, then the generator, on all named outputs."""
        discriminator_phase = (
            self.benchmark_recorder.phase("discriminator_update")
            if self.benchmark_recorder is not None
            else nullcontext()
        )
        with discriminator_phase:
            with autocast(device_type=self.device.type, enabled=self.amp_enabled):
                fake = {name: value.detach() for name, value in self.generator(inputs).items()}
                D_real = self.discriminator(inputs, targets)
                D_fake = self.discriminator(inputs, fake)
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

        generator_phase = (
            self.benchmark_recorder.phase("generator_update")
            if self.benchmark_recorder is not None
            else nullcontext()
        )
        with generator_phase:
            with autocast(device_type=self.device.type, enabled=self.amp_enabled):
                fake = self.generator(inputs)
                D_fake = self.discriminator(inputs, fake)
                context = LossEvaluationContext(epoch=epoch, global_step=global_step, masks=masks)
                generator_loss = self.loss_evaluator.generator_total(
                    predictions=fake,
                    targets=targets,
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
