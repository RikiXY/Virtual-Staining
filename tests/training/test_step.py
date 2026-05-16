from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler

from virtual_staining.training.config import LossScheduleConfig, LossTermConfig
from virtual_staining.training.losses import (
    LOSS_REGISTRY,
    ConfiguredLossEvaluator,
    LossEvaluationContext,
    LossTermResult,
    SsimLoss,
    StepLosses,
    evaluate_loss_term,
    reduce_masked_loss,
)
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
        device=torch.device("cpu"),
        amp_enabled=False,
        generator_loss_terms=(
            LossTermConfig(name="adversarial_bce", weight=1.0),
            LossTermConfig(name="l1", weight=25.0),
        ),
        discriminator_loss_terms=(LossTermConfig(name="adversarial_bce", weight=1.0),),
    )


def _make_configured_step(term: LossTermConfig) -> Pix2PixTrainingStep:
    gen = _TinyGen()
    disc = _TinyDisc()
    return Pix2PixTrainingStep(
        generator=gen,
        discriminator=disc,
        opt_G=optim.Adam(gen.parameters(), lr=1e-4),
        opt_D=optim.Adam(disc.parameters(), lr=1e-4),
        scaler_G=GradScaler(enabled=False),
        scaler_D=GradScaler(enabled=False),
        device=torch.device("cpu"),
        amp_enabled=False,
        generator_loss_terms=(term,),
    )


# ---------------------------------------------------------------------------
# Registry losses
# ---------------------------------------------------------------------------


def test_l1_registry_weight_scales_generator_loss() -> None:
    fake = torch.ones(2, 3, 8, 8)
    real = torch.zeros(2, 3, 8, 8)
    term_0 = LossTermConfig(name="l1", weight=0.0)
    term_25 = LossTermConfig(name="l1", weight=25.0)

    loss_0 = evaluate_loss_term(term_0, fake, real)
    loss_25 = evaluate_loss_term(term_25, fake, real)

    assert loss_25.weighted.item() > loss_0.weighted.item()


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


def test_configured_loss_evaluator_returns_generator_component_maps() -> None:
    evaluator = ConfiguredLossEvaluator(
        generator_terms=(LossTermConfig(name="l1", weight=2.0),),
    )
    prediction = torch.ones(1, 3, 8, 8)
    target = torch.zeros(1, 3, 8, 8)
    discriminator_fake = torch.zeros(1, 1, 1, 1)

    result = evaluator.generator_total(
        prediction=prediction,
        target=target,
        discriminator_fake=discriminator_fake,
        context=LossEvaluationContext(epoch=0),
    )

    assert result.total.item() == pytest.approx(2.0)
    assert result.raw == {"generator_l1": pytest.approx(1.0)}
    assert result.weighted == {"generator_l1": pytest.approx(2.0)}
    assert result.current_weight == {"generator_l1": pytest.approx(2.0)}


def test_configured_loss_evaluator_returns_discriminator_component_maps() -> None:
    evaluator = ConfiguredLossEvaluator(
        discriminator_terms=(LossTermConfig(name="adversarial_bce", weight=1.0),),
    )
    discriminator_real = torch.zeros(1, 1, 1, 1)
    discriminator_fake = torch.zeros(1, 1, 1, 1)

    result = evaluator.discriminator_total(
        discriminator_real=discriminator_real,
        discriminator_fake=discriminator_fake,
        context=LossEvaluationContext(epoch=0),
    )

    expected = math.log(2.0) * 2.0
    assert result.total.item() == pytest.approx(expected)
    assert result.raw == {"discriminator_adversarial_bce": pytest.approx(expected)}
    assert result.weighted == {"discriminator_adversarial_bce": pytest.approx(expected)}
    assert result.current_weight == {"discriminator_adversarial_bce": pytest.approx(1.0)}


# ---------------------------------------------------------------------------
# SsimLoss
# ---------------------------------------------------------------------------


def test_ssim_loss_identical_tensors_near_zero() -> None:
    loss = SsimLoss(window_size=5)
    x = torch.rand(2, 3, 16, 16) * 2.0 - 1.0

    value = loss(x, x)

    assert value.item() == pytest.approx(0.0, abs=1e-6)


def test_ssim_loss_perturbed_tensor_is_higher_than_identical() -> None:
    loss = SsimLoss(window_size=5)
    target = torch.zeros(1, 3, 16, 16)
    perturbed = target.clone()
    perturbed[:, :, 4:12, 4:12] = 0.5

    identical = loss(target, target)
    changed = loss(perturbed, target)

    assert changed.item() > identical.item()


def test_ssim_loss_gradients_flow_to_prediction() -> None:
    loss = SsimLoss(window_size=5)
    prediction = (torch.rand(1, 3, 16, 16) * 2.0 - 1.0).requires_grad_()
    target = torch.rand(1, 3, 16, 16) * 2.0 - 1.0

    value = loss(prediction, target)
    value.backward()

    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
    assert prediction.grad.abs().sum().item() > 0.0


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_ssim_loss_low_precision_inputs_use_stable_float32_math(dtype: torch.dtype) -> None:
    loss = SsimLoss(window_size=5)
    prediction = torch.linspace(-1.0, 1.0, 16).view(1, 1, 1, 16).expand(1, 3, 16, 16)
    target = -prediction

    expected = loss(prediction, target)
    value = loss(prediction.to(dtype=dtype), target.to(dtype=dtype))

    assert value.dtype == torch.float32
    assert value.item() == pytest.approx(expected.item(), abs=2e-3)


def test_ssim_loss_gray_mode_accepts_rgb_inputs() -> None:
    loss = SsimLoss(window_size=5, channel_mode="gray")
    x = torch.rand(1, 3, 16, 16) * 2.0 - 1.0

    value = loss(x, x)

    assert value.item() == pytest.approx(0.0, abs=1e-6)


def test_ssim_loss_none_reduction_returns_per_sample_values() -> None:
    loss = SsimLoss(window_size=5, reduction="none")
    x = torch.rand(2, 3, 16, 16) * 2.0 - 1.0

    value = loss(x, x)

    assert value.shape == (2,)
    assert torch.allclose(value, torch.zeros_like(value), atol=1e-6)


def test_ssim_loss_rejects_tiny_images() -> None:
    loss = SsimLoss(window_size=11)
    x = torch.zeros(1, 3, 8, 8)

    with pytest.raises(ValueError, match="window_size"):
        loss(x, x)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"window_size": 4}, "window_size"),
        ({"data_range": 0.0}, "data_range"),
        ({"sigma": 0.0}, "sigma"),
        ({"channel_mode": "lab"}, "channel_mode"),
        ({"reduction": "median"}, "reduction"),
    ],
)
def test_ssim_loss_rejects_invalid_parameters(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        SsimLoss(**kwargs)  # type: ignore[arg-type]


def test_evaluate_loss_term_returns_raw_and_weighted_ssim_values() -> None:
    term = LossTermConfig(
        name="ssim",
        weight=2.0,
        params={"window_size": 5, "data_range": 1.0, "channel_mode": "rgb"},
    )
    prediction = torch.zeros(1, 3, 16, 16)
    target = torch.ones(1, 3, 16, 16) * 0.25

    result = evaluate_loss_term(term, prediction, target)

    assert result.name == "ssim"
    assert result.weighted.item() == pytest.approx(result.raw.item() * 2.0)
    assert result.current_weight == pytest.approx(2.0)


def test_scheduled_loss_term_uses_current_weight() -> None:
    term = LossTermConfig(
        name="ssim",
        weight=2.0,
        params={"window_size": 5},
        schedule=LossScheduleConfig(type="constant"),
    )
    prediction = torch.zeros(1, 3, 16, 16)
    target = torch.ones(1, 3, 16, 16) * 0.25

    result = evaluate_loss_term(term, prediction, target, epoch=3)

    assert isinstance(result, LossTermResult)
    assert result.current_weight == pytest.approx(2.0)


def test_masked_loss_reduction_weights_foreground_and_background() -> None:
    loss_map = torch.tensor([[[[1.0, 1.0], [3.0, 3.0]]]])
    mask = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
    mask_config = LossTermConfig(
        name="ssim",
        weight=1.0,
        params={
            "mask": {
                "enabled": True,
                "foreground_weight": 1.0,
                "background_weight": 0.0,
            }
        },
    ).mask

    value = reduce_masked_loss(loss_map, mask, mask_config=mask_config)

    assert value.item() == pytest.approx(2.0)


def test_empty_mask_can_be_ignored_without_creating_loss() -> None:
    loss_map = torch.ones(1, 1, 2, 2)
    mask = torch.zeros(1, 1, 2, 2)
    mask_config = LossTermConfig(
        name="ssim",
        weight=1.0,
        params={
            "mask": {
                "enabled": True,
                "foreground_weight": 1.0,
                "background_weight": 0.25,
                "ignore_empty_mask": True,
            }
        },
    ).mask

    value = reduce_masked_loss(loss_map, mask, mask_config=mask_config)

    assert value.item() == pytest.approx(0.0)


def test_mask_enabled_loss_requires_batch_mask() -> None:
    term = LossTermConfig(
        name="ssim",
        weight=1.0,
        params={"window_size": 5, "mask": {"enabled": True}},
    )
    x = torch.zeros(1, 3, 16, 16)

    with pytest.raises(ValueError, match="foreground_mask"):
        evaluate_loss_term(term, x, x, masks=None)


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


def test_training_step_applies_configured_ssim_loss() -> None:
    term = LossTermConfig(name="ssim", weight=1.0, params={"window_size": 3})
    step = _make_configured_step(term)

    result = step.step(torch.zeros(1, 3, 8, 8), torch.ones(1, 3, 8, 8) * 0.5, epoch=0)

    assert result.loss_G > 0.0


def test_training_step_returns_component_maps() -> None:
    term = LossTermConfig(name="ssim", weight=2.0, params={"window_size": 3})
    step = _make_configured_step(term)

    result = step.step(torch.zeros(1, 3, 8, 8), torch.ones(1, 3, 8, 8) * 0.5, epoch=0)

    assert result.raw is not None
    assert result.weighted is not None
    assert result.current_weight is not None
    assert result.raw["generator_ssim"] > 0
    assert result.weighted["generator_ssim"] == pytest.approx(result.raw["generator_ssim"] * 2.0)
    assert result.current_weight["generator_ssim"] == pytest.approx(2.0)
