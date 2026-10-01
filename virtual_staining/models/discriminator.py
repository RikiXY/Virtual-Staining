from collections.abc import Mapping

import torch
import torch.nn as nn

from virtual_staining.models.generator import concat_named


def _make_norm(norm: str, channels: int) -> nn.Module:
    if norm == "batch":
        return nn.BatchNorm2d(channels)
    if norm == "instance":
        return nn.InstanceNorm2d(channels)
    raise ValueError(f"Unknown discriminator norm: {norm!r}")


class PatchGANDiscriminator(nn.Module):
    """
    PatchGAN discriminator for image-to-image tasks.

    Produces an NxN map of real/fake predictions, one per patch. Built with
    ``input_names``/``output_names`` it is one joint conditional discriminator over all
    ``3 * N`` input and ``3 * M`` output channels; otherwise it scores one pre-assembled
    ``in_channels`` tensor (CycleGAN's unconditional use).
    """

    def __init__(
        self,
        in_channels: int | None = None,
        ndf: int = 64,
        norm: str = "instance",
        use_sigmoid: bool = False,
        *,
        input_names: tuple[str, ...] = (),
        output_names: tuple[str, ...] = (),
    ) -> None:
        super().__init__()
        if bool(input_names) != bool(output_names):
            raise ValueError("a joint conditional PatchGAN needs both input and output names")
        if input_names:
            joint = 3 * (len(input_names) + len(output_names))
            if in_channels is not None and in_channels != joint:
                raise ValueError(f"in_channels={in_channels} contradicts {joint} named channels")
            in_channels = joint
        if in_channels is None:
            raise ValueError("PatchGANDiscriminator needs in_channels or input/output names")
        self.input_names = input_names
        self.output_names = output_names
        self.in_channels = in_channels
        self.ndf = ndf
        self.norm = norm
        self.use_sigmoid = use_sigmoid

        curr_dim = ndf
        next_dim = curr_dim * 2
        layers = [
            nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(curr_dim, next_dim, kernel_size=4, stride=2, padding=1),
            _make_norm(norm, next_dim),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        curr_dim = next_dim
        next_dim = curr_dim * 2
        layers += [
            nn.Conv2d(curr_dim, next_dim, kernel_size=4, stride=2, padding=1),
            _make_norm(norm, next_dim),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        curr_dim = next_dim
        next_dim = curr_dim * 2
        layers += [
            nn.Conv2d(curr_dim, next_dim, kernel_size=4, stride=1, padding=1),
            _make_norm(norm, next_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(next_dim, 1, kernel_size=4, stride=1, padding=1),
        ]

        if use_sigmoid:
            layers += [nn.Sigmoid()]

        self.model = nn.Sequential(*layers)

    def forward(
        self,
        x: torch.Tensor | Mapping[str, torch.Tensor],
        images: Mapping[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Score all ``images`` jointly conditioned on named inputs ``x``, or tensor ``x``."""
        if images is None:
            if not isinstance(x, torch.Tensor):
                raise TypeError("an unconditional PatchGAN scores one tensor")
            return self.model(x)
        if not self.input_names or isinstance(x, torch.Tensor):
            raise TypeError("joint conditional scoring needs a PatchGAN built with names")
        condition = concat_named(x, self.input_names, "Discriminator inputs")
        first = x[self.input_names[0]]
        scored = concat_named(images, self.output_names, "Discriminator outputs", like=first)
        return self.model(torch.cat([condition, scored], dim=1))
