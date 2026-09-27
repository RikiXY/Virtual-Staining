from __future__ import annotations

import dataclasses
import math
from types import MappingProxyType

import pytest
import torch

import virtual_staining.config.losses as config_losses
from virtual_staining.config.losses import LossScheduleConfig, LossTermConfig
from virtual_staining.loss_definitions import LOSS_DEFINITIONS
from virtual_staining.training.losses import (
    ConfiguredLossEvaluator,
    LossEvaluationContext,
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

    loss = LOSS_DEFINITIONS["ssim"].reconstruction_loss(image, image)

    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_bce_generator_and_discriminator_match_reference_formulas() -> None:
    generator = torch.Generator().manual_seed(4)
    real = torch.randn(2, 1, 3, 3, generator=generator)
    fake = torch.randn(2, 1, 3, 3, generator=generator)
    evaluator = ConfiguredLossEvaluator(
        generator_terms=(LossTermConfig(name="adversarial_bce", weight=2.0),),
        discriminator_terms=(LossTermConfig(name="adversarial_bce", weight=0.5),),
    )
    context = LossEvaluationContext(epoch=0)

    g = evaluator.generator_total(
        prediction=torch.zeros(1), target=torch.zeros(1), discriminator_fake=fake, context=context
    )
    d = evaluator.discriminator_total(
        discriminator_real=real, discriminator_fake=fake, context=context
    )

    # BCE-with-logits: -log(sigmoid(x)) = softplus(-x); -log(1 - sigmoid(x)) = softplus(x)
    expected_g = torch.nn.functional.softplus(-fake).mean().item()
    expected_d = (
        torch.nn.functional.softplus(-real).mean() + torch.nn.functional.softplus(fake).mean()
    ).item()
    assert g.raw == {"generator_adversarial_bce": pytest.approx(expected_g)}
    assert g.total.item() == pytest.approx(2.0 * expected_g)
    assert d.raw == {"discriminator_adversarial_bce": pytest.approx(expected_d)}
    assert d.total.item() == pytest.approx(0.5 * expected_d)


def test_generator_adversarial_term_requires_discriminator_logits() -> None:
    evaluator = ConfiguredLossEvaluator(
        generator_terms=(LossTermConfig(name="adversarial_bce", weight=1.0),)
    )
    assert evaluator.needs_discriminator_logits
    assert not ConfiguredLossEvaluator(
        generator_terms=(LossTermConfig(name="l1", weight=1.0),)
    ).needs_discriminator_logits
    with pytest.raises(ValueError, match="requires discriminator_fake logits"):
        evaluator.generator_total(
            prediction=torch.zeros(1),
            target=torch.zeros(1),
            context=LossEvaluationContext(epoch=0),
        )


def test_evaluator_resolves_primitives_through_canonical_definitions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def primitive(prediction, target, params, mask):  # type: ignore[no-untyped-def]
        calls.append(params.reduction)
        return (prediction - target).sum()

    patched = {
        **LOSS_DEFINITIONS,
        "l1": dataclasses.replace(LOSS_DEFINITIONS["l1"], reconstruction=primitive),
    }
    monkeypatch.setattr(config_losses, "LOSS_DEFINITIONS", MappingProxyType(patched))
    evaluator = ConfiguredLossEvaluator(
        generator_terms=(LossTermConfig(name="l1", weight=1.0, params={"reduction": "sum"}),)
    )

    result = evaluator.generator_total(
        prediction=torch.full((1, 1, 2, 2), 3.0),
        target=torch.zeros(1, 1, 2, 2),
        context=LossEvaluationContext(epoch=0),
    )

    assert calls == ["sum"]
    assert result.raw == {"generator_l1": 12.0}


def test_disabled_and_zero_weight_terms_are_reported_with_zero_weight() -> None:
    evaluator = ConfiguredLossEvaluator(
        generator_terms=(
            LossTermConfig(name="l1", weight=2.0, enabled=False),
            LossTermConfig(name="ssim", weight=0.0),
        )
    )
    image = torch.zeros(1, 3, 16, 16)

    result = evaluator.generator_total(
        prediction=image + 0.5, target=image, context=LossEvaluationContext(epoch=0)
    )

    assert set(result.raw) == {"generator_l1", "generator_ssim"}
    assert result.raw["generator_l1"] == pytest.approx(0.5)
    assert result.current_weight == {"generator_l1": 0.0, "generator_ssim": 0.0}
    assert result.weighted == {"generator_l1": 0.0, "generator_ssim": 0.0}
    assert result.total.item() == 0.0


@pytest.mark.parametrize("name", ["l1", "ssim"])
def test_reconstruction_terms_backpropagate_weighted_gradients(name: str) -> None:
    generator = torch.Generator().manual_seed(5)
    prediction = (torch.rand(2, 3, 16, 16, generator=generator) * 2 - 1).requires_grad_()
    target = torch.rand(2, 3, 16, 16, generator=generator) * 2 - 1
    evaluator = ConfiguredLossEvaluator(
        generator_terms=(LossTermConfig(name=name, weight=3.0),)  # type: ignore[arg-type]
    )

    result = evaluator.generator_total(
        prediction=prediction, target=target, context=LossEvaluationContext(epoch=0)
    )
    result.total.backward()
    weighted_grad = prediction.grad
    assert weighted_grad is not None and torch.isfinite(weighted_grad).all()

    raw_prediction = prediction.detach().clone().requires_grad_()
    LOSS_DEFINITIONS[name].reconstruction_loss(raw_prediction, target).backward()
    assert raw_prediction.grad is not None
    assert torch.allclose(weighted_grad, 3.0 * raw_prediction.grad)
