from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

ModelName = Literal["pix2pix"]
GanLossName = Literal["bce"]
NormName = Literal["batch", "instance"]


@dataclass(frozen=True)
class GeneratorConfig:
    name: Literal["unet"] = "unet"
    in_channels: int = 3
    out_channels: int = 3
    base_channels: int = 64
    norm: NormName = "batch"
    dropout: bool = False
    bilinear: bool = False


@dataclass(frozen=True)
class DiscriminatorConfig:
    name: Literal["patchgan"] = "patchgan"
    in_channels: int = 6
    ndf: int = 64
    norm: NormName = "instance"
    use_sigmoid: bool = False


@dataclass(frozen=True)
class ModelConfig:
    name: ModelName = "pix2pix"
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    discriminator: DiscriminatorConfig = field(default_factory=DiscriminatorConfig)
    gan_loss: GanLossName = "bce"
