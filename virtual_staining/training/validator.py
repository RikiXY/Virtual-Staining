from __future__ import annotations

import logging
from collections.abc import Mapping

import torch
import torch.nn as nn
from torch.amp import autocast

from virtual_staining.config.losses import LossConfig, configured_loss_names
from virtual_staining.metrics import ResolvedMetric
from virtual_staining.training.helpers import LossComponentAccumulator, unpack_batch
from virtual_staining.training.losses import ConfiguredLossEvaluator, LossEvaluationContext
from virtual_staining.training.preview import ValidationPreview, ValidationPreviewSink
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
    input_names: tuple[str, ...],
    output_names: tuple[str, ...],
    image_metrics: Mapping[str, tuple[str, ResolvedMetric]],
    preview_sink: ValidationPreviewSink | None = None,
) -> EpochMetrics:
    """Validate a Pix2Pix epoch; image metrics compare each output only with its target."""
    generator_was_training = generator.training
    discriminator_was_training = discriminator.training
    generator.eval()
    discriminator.eval()

    try:
        total_loss_G = 0.0
        total_loss_D = 0.0
        component_totals = LossComponentAccumulator(configured_loss_names(losses, output_names))
        needs_discriminator = loss_evaluator.needs_discriminator_logits
        image_metric_totals = {
            output: ValidationImageMetricAccumulator(
                {
                    column: metric
                    for column, (name, metric) in image_metrics.items()
                    if name == output
                }
            )
            for output in output_names
        }
        count = 0
        with torch.no_grad():
            for batch_index, batch in enumerate(val_loader):
                inputs, targets, masks = unpack_batch(batch, device, input_names, output_names)
                with autocast(device_type=device.type, enabled=amp_enabled):
                    generated = generator(inputs)
                    context = LossEvaluationContext(epoch=epoch, masks=masks)
                    discriminator_fake: torch.Tensor | None = None
                    if needs_discriminator:
                        discriminator_real = discriminator(inputs, targets)
                        discriminator_fake_logits = discriminator(inputs, generated)
                        discriminator_fake = discriminator_fake_logits
                        discriminator_loss = loss_evaluator.discriminator_total(
                            discriminator_real=discriminator_real,
                            discriminator_fake=discriminator_fake_logits,
                            context=context,
                        )
                    else:
                        discriminator_loss = None
                    generator_loss = loss_evaluator.generator_total(
                        predictions=generated,
                        targets=targets,
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
                for output, accumulator in image_metric_totals.items():
                    accumulator.add_batch(generated[output], targets[output])
                count += 1
                if preview_sink is not None and preview_sink.wants(epoch, batch_index):
                    images = {"input": inputs[input_names[0]].detach()}
                    for output in output_names:
                        images[f"output__{output}"] = generated[output].detach()
                        images[f"target__{output}"] = targets[output].detach()
                    preview_sink.write(
                        ValidationPreview(epoch=epoch, batch_index=batch_index, images=images)
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
            image={
                column: value
                for accumulator in image_metric_totals.values()
                for column, value in accumulator.mean().items()
            },
        )
    finally:
        if generator_was_training:
            generator.train()
        if discriminator_was_training:
            discriminator.train()
