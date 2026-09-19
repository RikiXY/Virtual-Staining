from __future__ import annotations

import math

import pytest
import torch

from virtual_staining.config.losses import LossScheduleConfig, LossTermConfig
from virtual_staining.training.losses import (
    ConfiguredLossEvaluator,
    LossEvaluationContext,
    SsimLoss,
)


def test_generator_total_composes_weighted_l1_and_adversarial_losses() -> None:
    evaluator = ConfiguredLossEvaluator(
        generator_terms=(
            LossTermConfig(name="l1", weight=2.0),
            LossTermConfig(name="adversarial_bce", weight=0.5),
        )
    )
    prediction = torch.zeros(1, 1, 2, 2)
    target = torch.ones_like(prediction)
    logits = torch.zeros(1, 1, 2, 2)

    result = evaluator.generator_total(
        prediction=prediction,
        target=target,
        discriminator_fake=logits,
        context=LossEvaluationContext(epoch=0),
    )

    assert result.raw["generator_l1"] == pytest.approx(1.0)
    assert result.current_weight == {
        "generator_l1": 2.0,
        "generator_adversarial_bce": 0.5,
    }
    expected_bce = math.log(2.0)
    assert result.raw["generator_adversarial_bce"] == pytest.approx(expected_bce)
    assert result.total.item() == pytest.approx(2.0 + 0.5 * expected_bce)


@pytest.mark.parametrize(
    ("epoch", "expected_weight"),
    [(0, 0.0), (2, 2.0), (4, 4.0), (8, 4.0)],
)
def test_generator_total_applies_loss_schedule(epoch: int, expected_weight: float) -> None:
    term = LossTermConfig(
        name="l1",
        weight=4.0,
        schedule=LossScheduleConfig(type="linear_warmup", start_epoch=0, end_epoch=4),
    )
    evaluator = ConfiguredLossEvaluator(generator_terms=(term,))
    prediction = torch.zeros(1, 1, 2, 2)
    target = torch.ones_like(prediction)

    result = evaluator.generator_total(
        prediction=prediction,
        target=target,
        context=LossEvaluationContext(epoch=epoch),
    )

    assert result.current_weight["generator_l1"] == pytest.approx(expected_weight)
    assert result.total.item() == pytest.approx(expected_weight)


def test_discriminator_total_reports_weighted_adversarial_component() -> None:
    evaluator = ConfiguredLossEvaluator(
        discriminator_terms=(LossTermConfig(name="adversarial_bce", weight=0.25),)
    )
    logits = torch.zeros(1, 1, 2, 2)

    result = evaluator.discriminator_total(
        discriminator_real=logits,
        discriminator_fake=logits,
        context=LossEvaluationContext(epoch=0),
    )

    expected_raw = 2.0 * math.log(2.0)
    assert result.raw["discriminator_adversarial_bce"] == pytest.approx(expected_raw)
    assert result.current_weight["discriminator_adversarial_bce"] == pytest.approx(0.25)
    assert result.total.item() == pytest.approx(expected_raw * 0.25)


def test_masked_l1_uses_foreground_weighting_through_public_evaluator() -> None:
    term = LossTermConfig(
        name="l1",
        weight=1.0,
        params={
            "mask": {
                "enabled": True,
                "foreground_weight": 1.0,
                "background_weight": 0.0,
            }
        },
    )
    evaluator = ConfiguredLossEvaluator(generator_terms=(term,))
    prediction = torch.tensor([[[[1.0, 10.0], [1.0, 10.0]]]])
    target = torch.zeros_like(prediction)
    mask = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])

    result = evaluator.generator_total(
        prediction=prediction,
        target=target,
        context=LossEvaluationContext(epoch=0, masks={"foreground_mask": mask}),
    )

    assert result.raw["generator_l1"] == pytest.approx(1.0)
    assert result.total.item() == pytest.approx(1.0)


def test_ssim_loss_is_zero_for_identical_normalized_images() -> None:
    image = torch.zeros(2, 3, 16, 16)

    loss = SsimLoss()(image, image)

    assert loss.item() == pytest.approx(0.0, abs=1e-6)
