from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn

_CONV_KERNEL = 3
_CONV_PADDING = 1
_POOL_KERNEL = 2


def _make_norm(norm: str, channels: int) -> nn.Module:
    if norm == "batch":
        return nn.BatchNorm2d(channels)
    if norm == "instance":
        return nn.InstanceNorm2d(channels)
    raise ValueError(f"Unknown generator norm: {norm!r}")


class DoubleConv(nn.Module):
    """
    Block consisting of two consecutive convolutions, each followed by
    batch normalisation and ReLU activation.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
    """

    def __init__(self, in_channels: int, out_channels: int, norm: str) -> None:
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=_CONV_KERNEL,
                padding=_CONV_PADDING,
                bias=False,
            ),
            _make_norm(norm, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=_CONV_KERNEL,
                padding=_CONV_PADDING,
                bias=False,
            ),
            _make_norm(norm, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """
    Downsampling block for U-Net-style architectures.

    Reduces the spatial resolution via max pooling and then applies
    a `DoubleConv` block to extract richer features.
    """

    def __init__(self, in_channels: int, out_channels: int, norm: str) -> None:
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=_POOL_KERNEL, stride=_POOL_KERNEL),
            DoubleConv(in_channels, out_channels, norm),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    Upsampling block followed by `DoubleConv`.

    The upsampled feature map is concatenated with the encoder skip
    connection before the final convolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bilinear: bool = True,
        *,
        norm: str,
        dropout: bool = False,
    ) -> None:
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        layers: list[nn.Module] = [DoubleConv(in_channels, out_channels, norm)]
        if dropout:
            layers.append(nn.Dropout(p=0.5))
        self.conv = nn.Sequential(*layers)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """
    Final 1x1 convolution that maps the features
    to the required number of output channels.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNetGenerator(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 64,
        norm: str = "batch",
        dropout: bool = False,
        bilinear: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.norm = norm
        self.dropout = dropout
        self.bilinear = bilinear
        b = base_channels
        self.inc = DoubleConv(in_channels, b, norm)
        self.down1 = Down(b, b * 2, norm)
        self.down2 = Down(b * 2, b * 4, norm)
        self.down3 = Down(b * 4, b * 8, norm)
        self.down4 = Down(b * 8, b * 16, norm)
        self.up1 = Up(b * 16, b * 8, bilinear, norm=norm, dropout=dropout)
        self.up2 = Up(b * 8, b * 4, bilinear, norm=norm, dropout=dropout)
        self.up3 = Up(b * 4, b * 2, bilinear, norm=norm, dropout=dropout)
        self.up4 = Up(b * 2, b, bilinear, norm=norm, dropout=False)
        self.outc = OutConv(b, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return torch.tanh(self.outc(x))


def concat_inputs(inputs: Mapping[str, torch.Tensor], input_names: tuple[str, ...]) -> torch.Tensor:
    if tuple(inputs) != input_names:
        raise ValueError(
            f"Generator inputs must have exact ordered names {input_names}, got {tuple(inputs)}"
        )
    return torch.cat([inputs[name] for name in input_names], dim=1)


class ConcatUNetGenerator(nn.Module):
    def __init__(
        self,
        input_names: tuple[str, ...],
        channels_per_input: int = 3,
        **unet_kwargs: Any,
    ) -> None:
        super().__init__()
        if not input_names or len(set(input_names)) != len(input_names):
            raise ValueError("input_names must be non-empty and unique")
        self.input_names = input_names
        self.channels_per_input = channels_per_input
        self.unet = UNetGenerator(
            in_channels=len(input_names) * channels_per_input,
            out_channels=3,
            **unet_kwargs,
        )

    def forward(self, inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return self.unet(concat_inputs(inputs, self.input_names))


RESNET_SIZE_MULTIPLE = 4


class ResnetBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class ResnetGenerator(nn.Module):
    """CycleGAN ResNet generator mapping one RGB tensor to one RGB tensor."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 64,
        blocks: int = 9,
    ) -> None:
        super().__init__()
        if blocks < 1:
            raise ValueError("ResnetGenerator requires at least one residual block")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.blocks = blocks
        b = base_channels
        layers: list[nn.Module] = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, b, kernel_size=7),
            nn.InstanceNorm2d(b),
            nn.ReLU(inplace=True),
        ]
        for mult in (1, 2):
            layers += [
                nn.Conv2d(b * mult, b * mult * 2, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(b * mult * 2),
                nn.ReLU(inplace=True),
            ]
        layers += [ResnetBlock(b * 4) for _ in range(blocks)]
        for mult in (4, 2):
            layers += [
                nn.ConvTranspose2d(
                    b * mult, b * mult // 2, kernel_size=3, stride=2, padding=1, output_padding=1
                ),
                nn.InstanceNorm2d(b * mult // 2),
                nn.ReLU(inplace=True),
            ]
        layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(b, out_channels, kernel_size=7),
            nn.Tanh(),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError(
                f"ResnetGenerator expects NCHW input with {self.in_channels} channels, "
                f"got shape {tuple(x.shape)}"
            )
        if x.shape[-2] % RESNET_SIZE_MULTIPLE or x.shape[-1] % RESNET_SIZE_MULTIPLE:
            raise ValueError(
                f"ResnetGenerator spatial dimensions must be multiples of {RESNET_SIZE_MULTIPLE}, "
                f"got {tuple(x.shape[-2:])}"
            )
        return self.model(x)
