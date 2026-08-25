from __future__ import annotations

import logging
from pathlib import Path
from typing import cast

import torch
import torch.nn as nn
from torch.amp import autocast

from virtual_staining.config.losses import LossConfig
from virtual_staining.models.generator import concat_inputs
from virtual_staining.training.helpers import (
    LossComponentAccumulator,
    configured_loss_names,
    save_images,
    unpack_batch,
)
from virtual_staining.training.losses import ConfiguredLossEvaluator, LossEvaluationContext
from virtual_staining.training.results import EpochMetrics
from virtual_staining.training.validation_metrics import ValidationImageMetricAccumulator

logger = logging.getLogger(__name__)


def validate_epoch(
    *,
    epoch: int,
    generator: nn.Module,
    discriminator: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    loss_evaluator: ConfiguredLossEvaluator,
    losses: LossConfig | None,
    device: torch.device,
    amp_enabled: bool,
    output_dir: Path,
) -> EpochMetrics:
    generator_was_training = generator.training
    discriminator_was_training = discriminator.training
    generator.eval()
    discriminator.eval()

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        total_loss_G = 0.0
        total_loss_D = 0.0
        component_totals = LossComponentAccumulator(configured_loss_names(losses))
        needs_discriminator = _needs_discriminator_logits(losses)
        image_metric_totals = ValidationImageMetricAccumulator()
        count = 0
        with torch.no_grad():
            input_names = cast(tuple[str, ...], generator.input_names)
            for batch_index, batch in enumerate(val_loader):
                inputs, target, masks = unpack_batch(batch, device, input_names)
                condition = concat_inputs(inputs, input_names)
                with autocast(device_type=device.type, enabled=amp_enabled):
                    generated = generator(inputs)
                    context = LossEvaluationContext(epoch=epoch, masks=masks)
                    discriminator_fake: torch.Tensor | None = None
                    if needs_discriminator:
                        discriminator_real = discriminator(condition, target)
                        discriminator_fake_logits = discriminator(condition, generated)
                        discriminator_fake = discriminator_fake_logits
                        discriminator_loss = loss_evaluator.discriminator_total(
                            discriminator_real=discriminator_real,
                            discriminator_fake=discriminator_fake_logits,
                            context=context,
                        )
                    else:
                        discriminator_loss = None
                    generator_loss = loss_evaluator.generator_total(
                        prediction=generated,
                        target=target,
                        discriminator_fake=discriminator_fake,
                        context=context,
                    )

                    if discriminator_loss is not None:
                        component_totals.add(
                            raw=discriminator_loss.raw,
                            weighted=discriminator_loss.weighted,
                            current_weight=discriminator_loss.current_weight,
                        )
                    component_totals.add(
                        raw=generator_loss.raw,
                        weighted=generator_loss.weighted,
                        current_weight=generator_loss.current_weight,
                    )

                total_loss_D += (
                    discriminator_loss.total.item() if discriminator_loss is not None else 0.0
                )
                total_loss_G += generator_loss.total.item()
                image_metric_totals.add_batch(generated, target)
                count += 1
                if batch_index < 5:
                    save_images(
                        output_dir,
                        inputs[next(iter(inputs))][0],
                        generated[0],
                        target[0],
                        epoch,
                        batch_index,
                    )

        averages = component_totals.average(count)
        loss_G = total_loss_G / count if count else 0.0
        loss_D = total_loss_D / count if count else 0.0
        logger.info("[Epoch %s] Validation: loss_G=%.4f loss_D=%.4f", epoch, loss_G, loss_D)
        return EpochMetrics(
            loss_G=loss_G,
            loss_D=loss_D,
            raw=averages.raw,
            weighted=averages.weighted,
            current_weight=averages.current_weight,
            image=image_metric_totals.mean(),
        )
    finally:
        if generator_was_training:
            generator.train()
        if discriminator_was_training:
            discriminator.train()


def _needs_discriminator_logits(losses: LossConfig | None) -> bool:
    if losses is None:
        return False
    return any(
        term.name == "adversarial_bce" for term in (*losses.generator, *losses.discriminator)
    )
