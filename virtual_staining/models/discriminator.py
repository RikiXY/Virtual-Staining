import torch
import torch.nn as nn


def _make_norm(norm: str, channels: int) -> nn.Module:
    if norm == "batch":
        return nn.BatchNorm2d(channels)
    if norm == "instance":
        return nn.InstanceNorm2d(channels)
    raise ValueError(f"Unknown discriminator norm: {norm!r}")


class PatchGANDiscriminator(nn.Module):
    """
    PatchGAN discriminator for image-to-image tasks.

    Produces an NxN map of real/fake predictions, one per patch.
    """

    def __init__(
        self,
        in_channels: int = 6,
        ndf: int = 64,
        norm: str = "instance",
        use_sigmoid: bool = False,
    ) -> None:
        super().__init__()
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

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.model(torch.cat([x, y], dim=1))
