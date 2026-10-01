from __future__ import annotations

import pytest
import torch
from torch import nn

from virtual_staining.config.losses import LossTermConfig
from virtual_staining.models.generator import concat_named, split_named
from virtual_staining.training.losses import ConfiguredLossEvaluator
from virtual_staining.training.steps import Pix2PixTrainingStep

_INPUTS = ("LF", "AF")
_OUTPUTS = ("PAS", "HE")


class TinyGenerator(nn.Module):
    input_names = _INPUTS

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(6, 6, 1)

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        stacked = concat_named(inputs, self.input_names, "Generator inputs")
        return split_named(torch.tanh(self.conv(stacked)), _OUTPUTS)


class TinyDiscriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(12, 1, 1)

    def forward(
        self, inputs: dict[str, torch.Tensor], images: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        joint = torch.cat(
            [concat_named(inputs, _INPUTS, "in"), concat_named(images, _OUTPUTS, "out")], dim=1
        )
        return self.conv(joint).mean((2, 3))


def _step(
    generator: nn.Module, discriminator: nn.Module, *, discriminator_terms: bool = True
) -> Pix2PixTrainingStep:
    return Pix2PixTrainingStep(
        generator,
        discriminator,
        torch.optim.Adam(generator.parameters(), lr=1e-3),
        torch.optim.SGD(discriminator.parameters(), lr=1e-1),
        torch.amp.GradScaler("cpu", enabled=False),
        torch.amp.GradScaler("cpu", enabled=False),
        torch.device("cpu"),
        False,
        loss_evaluator=ConfiguredLossEvaluator(
            generator_terms=(LossTermConfig("adversarial_bce", 1.0), LossTermConfig("l1", 1.0)),
            discriminator_terms=(
                (LossTermConfig("adversarial_bce", 1.0),) if discriminator_terms else ()
            ),
        ),
    )


def _batch() -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    inputs = {name: torch.randn(2, 3, 8, 8) for name in _INPUTS}
    targets = {name: torch.randn(2, 3, 8, 8).clamp(-1, 1) for name in _OUTPUTS}
    return inputs, targets


def test_training_step_updates_on_every_named_output() -> None:
    generator = TinyGenerator()
    result = _step(generator, TinyDiscriminator()).step(*_batch(), masks={})

    assert set(result.raw or {}) == {
        "discriminator_adversarial_bce",
        "generator_adversarial_bce",
        "generator_l1__PAS",
        "generator_l1__HE",
    }
    assert result.loss_G == result.loss_G and result.loss_D == result.loss_D
    grad = generator.conv.weight.grad
    assert grad is not None
    # Rows 0-2 produce PAS and rows 3-5 produce HE; both received generator gradient.
    assert grad[0:3].abs().sum() > 0 and grad[3:6].abs().sum() > 0


def test_generator_phase_never_updates_the_discriminator() -> None:
    discriminator = TinyDiscriminator()
    before = [parameter.detach().clone() for parameter in discriminator.parameters()]

    # No discriminator loss: the discriminator phase has zero gradient, so any change
    # would have to come from the generator phase.
    _step(TinyGenerator(), discriminator, discriminator_terms=False).step(*_batch(), masks={})

    after = list(discriminator.parameters())
    assert all(torch.equal(old, new) for old, new in zip(before, after, strict=True))


def test_training_step_rejects_missing_named_input() -> None:
    step = _step(TinyGenerator(), TinyDiscriminator())
    inputs, targets = _batch()
    with pytest.raises(ValueError, match="Generator inputs must have exact ordered names"):
        step.step({"LF": inputs["LF"]}, targets, masks={})


def test_training_step_rejects_reordered_targets() -> None:
    step = _step(TinyGenerator(), TinyDiscriminator())
    inputs, targets = _batch()
    with pytest.raises(ValueError, match="exact ordered names"):
        step.step(inputs, dict(reversed(targets.items())), masks={})
