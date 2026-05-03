import torch
import torch.nn as nn

# Fixed conv-block hyperparameters for the standard UNet architecture.
_CONV_KERNEL = 3
_CONV_PADDING = 1
_POOL_KERNEL = 2


class DoubleConv(nn.Module):
    """
    Block consisting of two consecutive convolutions, each followed by
    batch normalisation and ReLU activation.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
    """
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=_CONV_KERNEL, padding=_CONV_PADDING, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=_CONV_KERNEL, padding=_CONV_PADDING, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """
    Downsampling block for U-Net-style architectures.

    Reduces the spatial resolution via max pooling and then applies
    a `DoubleConv` block to extract richer features.
    """
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=_POOL_KERNEL, stride=_POOL_KERNEL),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    Upsampling block followed by `DoubleConv`.

    The upsampled feature map is concatenated with the encoder skip
    connection before the final convolution.
    """
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2,
                                         kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """
    Final 1x1 convolution that maps the features
    to the required number of output channels.
    """
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNetGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_channels=64, bilinear=False):
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            base_channels (int): Number of filters in the first encoder block; doubles at each depth level.
            bilinear (bool): Whether to use bilinear upsampling or transposed convolution.
        """
        super(UNetGenerator, self).__init__()
        b = base_channels
        self.inc = DoubleConv(in_channels, b)
        self.down1 = Down(b, b * 2)
        self.down2 = Down(b * 2, b * 4)
        self.down3 = Down(b * 4, b * 8)
        self.down4 = Down(b * 8, b * 16)
        self.up1 = Up(b * 16, b * 8, bilinear)
        self.up2 = Up(b * 8, b * 4, bilinear)
        self.up3 = Up(b * 4, b * 2, bilinear)
        self.up4 = Up(b * 2, b, bilinear)
        self.outc = OutConv(b, out_channels)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)
