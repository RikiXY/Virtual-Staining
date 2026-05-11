import torch
import torch.nn as nn


class PatchGANDiscriminator(nn.Module):
    """
    PatchGAN discriminator for image-to-image tasks.

    Produces an NxN map of real/fake predictions, one per patch.
    """

    def __init__(self, in_channels: int = 6, ndf: int = 64, use_sigmoid: bool = False) -> None:
        """
        Args:
            in_channels (int): Number of input channels. In pix2pix,
                               concatenating the RGB input and output gives 6.
            ndf (int): Base number of discriminator filters.
            use_sigmoid (bool): Whether to apply a final sigmoid activation.
        """
        super().__init__()
        self.in_channels = in_channels
        self.ndf = ndf
        self.use_sigmoid = use_sigmoid

        curr_dim = ndf
        next_dim = curr_dim * 2
        layers = [
            nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(curr_dim, next_dim, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(next_dim),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        curr_dim = next_dim
        next_dim = curr_dim * 2
        layers += [
            nn.Conv2d(curr_dim, next_dim, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(next_dim),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        # stride=1 keeps the receptive field at ~70x70 (standard PatchGAN)
        curr_dim = next_dim
        next_dim = curr_dim * 2
        layers += [
            nn.Conv2d(curr_dim, next_dim, kernel_size=4, stride=1, padding=1),
            nn.InstanceNorm2d(next_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(next_dim, 1, kernel_size=4, stride=1, padding=1),
        ]

        if use_sigmoid:
            layers += [nn.Sigmoid()]

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): Input image.
            y (Tensor): Real target or generated output.

        Returns:
            Tensor: Map of real/fake predictions per patch.
        """
        return self.model(torch.cat([x, y], dim=1))
