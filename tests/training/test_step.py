from __future__ import annotations

import torch
from torch import nn

from virtual_staining.config.losses import LossTermConfig
from virtual_staining.training.steps import Pix2PixTrainingStep


class TinyGenerator(nn.Module):
    input_names = ("LF", "AF")

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(6, 3, 1)

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.tanh(self.conv(torch.cat([inputs[name] for name in self.input_names], dim=1)))


class TinyDiscriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(9, 1, 1)

    def forward(self, condition: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        return self.conv(torch.cat([condition, image], dim=1)).mean((2, 3))


def test_training_step_accepts_named_inputs_and_derived_channels() -> None:
    generator = TinyGenerator()
    discriminator = TinyDiscriminator()
    step = Pix2PixTrainingStep(
        generator,
        discriminator,
        torch.optim.Adam(generator.parameters(), lr=1e-3),
        torch.optim.Adam(discriminator.parameters(), lr=1e-3),
        torch.amp.GradScaler("cpu", enabled=False),
        torch.amp.GradScaler("cpu", enabled=False),
        torch.device("cpu"),
        False,
        generator_loss_terms=(LossTermConfig("adversarial_bce", 1.0), LossTermConfig("l1", 1.0)),
        discriminator_loss_terms=(LossTermConfig("adversarial_bce", 1.0),),
    )
    inputs = {"LF": torch.randn(2, 3, 8, 8), "AF": torch.randn(2, 3, 8, 8)}
    result = step.step(inputs, torch.randn(2, 3, 8, 8), masks={})
    assert result.loss_G == result.loss_G
    assert result.loss_D == result.loss_D


def test_training_step_rejects_missing_named_input() -> None:
    generator = TinyGenerator()
    step = Pix2PixTrainingStep(
        generator,
        TinyDiscriminator(),
        torch.optim.SGD(generator.parameters(), lr=1e-3),
        torch.optim.SGD(TinyDiscriminator().parameters(), lr=1e-3),
        torch.amp.GradScaler("cpu", enabled=False),
        torch.amp.GradScaler("cpu", enabled=False),
        torch.device("cpu"),
        False,
        generator_loss_terms=(LossTermConfig("l1", 1.0),),
        discriminator_loss_terms=(),
    )
    try:
        step.step({"LF": torch.zeros(1, 3, 4, 4)}, torch.zeros(1, 3, 4, 4), masks={})
    except ValueError as exc:
        assert "Generator inputs must have exact ordered names" in str(exc)
    else:
        raise AssertionError("missing named input was accepted")
