"""Named N-input -> M-output ConcatUNet generator and the joint conditional PatchGAN."""

from __future__ import annotations

import pytest
import torch

from virtual_staining.models.discriminator import PatchGANDiscriminator
from virtual_staining.models.generator import ConcatUNetGenerator, UNetGenerator


def _images(names: tuple[str, ...], batch: int = 2, size: int = 16) -> dict[str, torch.Tensor]:
    return {name: torch.randn(batch, 3, size, size) for name in names}


@pytest.mark.parametrize(
    ("inputs", "outputs"),
    [(("LF",), ("HE",)), (("LF", "AF"), ("HE",)), (("LF", "AF"), ("PAS", "HE"))],
)
def test_generator_maps_named_inputs_to_exactly_ordered_rgb_outputs(
    inputs: tuple[str, ...], outputs: tuple[str, ...]
) -> None:
    generator = ConcatUNetGenerator(inputs, outputs, base_channels=4)

    predicted = generator(_images(inputs))

    assert generator.unet.in_channels == 3 * len(inputs)
    assert generator.unet.out_channels == 3 * len(outputs)
    assert tuple(predicted) == outputs
    for tensor in predicted.values():
        assert tensor.shape == (2, 3, 16, 16)
        assert tensor.min() >= -1.0 and tensor.max() <= 1.0


def test_generator_splits_output_channels_by_configured_order() -> None:
    torch.manual_seed(0)
    generator = ConcatUNetGenerator(("LF",), ("PAS", "HE"), base_channels=4).eval()
    batch = _images(("LF",))

    with torch.no_grad():
        predicted = generator(batch)
        raw = generator.unet(batch["LF"])

    assert torch.equal(predicted["PAS"], raw[:, 0:3])
    assert torch.equal(predicted["HE"], raw[:, 3:6])


def test_one_output_is_numerically_the_underlying_unet() -> None:
    torch.manual_seed(0)
    generator = ConcatUNetGenerator(("LF", "AF"), ("HE",), base_channels=4).eval()
    reference = UNetGenerator(in_channels=6, out_channels=3, base_channels=4).eval()
    reference.load_state_dict(generator.unet.state_dict())
    batch = _images(("LF", "AF"))

    with torch.no_grad():
        predicted = generator(batch)
        expected = reference(torch.cat([batch["LF"], batch["AF"]], dim=1))

    assert list(predicted) == ["HE"]
    assert torch.equal(predicted["HE"], expected)


def test_every_output_receives_gradient() -> None:
    generator = ConcatUNetGenerator(("LF",), ("HE", "PAS"), base_channels=4)
    predicted = generator(_images(("LF",)))

    predicted["PAS"].sum().backward()
    out_weight = generator.unet.outc.conv.weight.grad
    assert out_weight is not None
    # Only PAS was used: its three output channels get gradient, HE's get none.
    assert out_weight[3:6].abs().sum() > 0
    assert out_weight[0:3].abs().sum() == 0

    generator.zero_grad()
    generator(_images(("LF",)))["HE"].sum().backward()
    he_grad = generator.unet.outc.conv.weight.grad
    assert he_grad is not None and he_grad[0:3].abs().sum() > 0


@pytest.mark.parametrize(
    ("batch", "match"),
    [
        ({"AF": torch.zeros(1, 3, 16, 16), "LF": torch.zeros(1, 3, 16, 16)}, "exact ordered names"),
        ({"LF": torch.zeros(1, 3, 16, 16)}, "exact ordered names"),
        ({"LF": torch.zeros(1, 1, 16, 16), "AF": torch.zeros(1, 3, 16, 16)}, "RGB NCHW"),
        ({"LF": torch.zeros(1, 3, 16, 16), "AF": torch.zeros(2, 3, 16, 16)}, "batch and spatial"),
    ],
)
def test_generator_rejects_wrong_named_inputs(batch: dict[str, torch.Tensor], match: str) -> None:
    generator = ConcatUNetGenerator(("LF", "AF"), ("HE",), base_channels=4)
    with pytest.raises(ValueError, match=match):
        generator(batch)


@pytest.mark.parametrize(
    ("inputs", "outputs"),
    [((), ("HE",)), (("LF",), ()), (("LF", "LF"), ("HE",)), (("LF",), ("LF",))],
)
def test_generator_rejects_invalid_names(inputs: tuple[str, ...], outputs: tuple[str, ...]) -> None:
    with pytest.raises(ValueError):
        ConcatUNetGenerator(inputs, outputs, base_channels=4)


def test_joint_patchgan_concatenates_inputs_then_outputs_in_order() -> None:
    discriminator = PatchGANDiscriminator(
        ndf=4, input_names=("LF", "AF"), output_names=("PAS", "HE")
    )
    seen: list[torch.Tensor] = []
    discriminator.model.register_forward_pre_hook(lambda _, args: seen.append(args[0]))
    inputs = {"LF": torch.full((1, 3, 32, 32), 1.0), "AF": torch.full((1, 3, 32, 32), 2.0)}
    outputs = {"PAS": torch.full((1, 3, 32, 32), 3.0), "HE": torch.full((1, 3, 32, 32), 4.0)}

    discriminator(inputs, outputs)

    assert discriminator.in_channels == 12
    channel_means = seen[0].mean(dim=(0, 2, 3)).tolist()
    assert channel_means == [1.0] * 3 + [2.0] * 3 + [3.0] * 3 + [4.0] * 3


@pytest.mark.parametrize(
    ("images", "match"),
    [
        ({"HE": torch.zeros(1, 3, 32, 32), "PAS": torch.zeros(1, 3, 32, 32)}, "exact ordered"),
        ({"PAS": torch.zeros(1, 3, 32, 32)}, "exact ordered"),
        (
            {
                "PAS": torch.zeros(1, 3, 32, 32),
                "HE": torch.zeros(1, 3, 32, 32),
                "X": torch.zeros(1),
            },
            "exact ordered",
        ),
        ({"PAS": torch.zeros(1, 3, 32, 32), "HE": torch.zeros(1, 3, 16, 16)}, "batch and spatial"),
        ({"PAS": torch.zeros(2, 3, 32, 32), "HE": torch.zeros(2, 3, 32, 32)}, "batch and spatial"),
        ({"PAS": torch.zeros(1, 3, 32, 32), "HE": torch.zeros(1, 1, 32, 32)}, "RGB NCHW"),
    ],
)
def test_joint_patchgan_rejects_wrong_outputs(images: dict[str, torch.Tensor], match: str) -> None:
    discriminator = PatchGANDiscriminator(ndf=4, input_names=("LF",), output_names=("PAS", "HE"))
    with pytest.raises(ValueError, match=match):
        discriminator({"LF": torch.zeros(1, 3, 32, 32)}, images)


def test_joint_patchgan_passes_gradient_to_every_fake_output() -> None:
    discriminator = PatchGANDiscriminator(ndf=4, input_names=("LF",), output_names=("PAS", "HE"))
    fakes = {name: torch.randn(1, 3, 32, 32, requires_grad=True) for name in ("PAS", "HE")}

    discriminator({"LF": torch.randn(1, 3, 32, 32)}, fakes).mean().backward()

    for fake in fakes.values():
        assert fake.grad is not None and fake.grad.abs().sum() > 0


def test_joint_patchgan_one_output_matches_the_previous_channel_layout() -> None:
    torch.manual_seed(0)
    discriminator = PatchGANDiscriminator(ndf=4, input_names=("LF", "AF"), output_names=("HE",))
    inputs = _images(("LF", "AF"), batch=1, size=32)
    target = torch.randn(1, 3, 32, 32)

    expected = discriminator.model(torch.cat([inputs["LF"], inputs["AF"], target], dim=1))

    assert torch.equal(discriminator(inputs, {"HE": target}), expected)
