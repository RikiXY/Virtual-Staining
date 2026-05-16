from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from virtual_staining.training.config import LossTermConfig


@dataclass(frozen=True)
class LossRegistryEntry:
    name: str
    roles: tuple[Literal["generator", "discriminator"], ...]
    targets: tuple[Literal["image"], ...]
    default_weight: float = 0.0


LOSS_REGISTRY: dict[str, LossRegistryEntry] = {
    "ssim": LossRegistryEntry(name="ssim", roles=("generator",), targets=("image",)),
}


@dataclass
class StepLosses:
    loss_G: float
    loss_D: float


@dataclass(frozen=True)
class LossTermResult:
    name: str
    raw: torch.Tensor
    weighted: torch.Tensor


class SsimLoss(nn.Module):
    """Differentiable SSIM loss for normalized training tensors.

    The training pipeline uses tensors in [-1, 1]. SSIM is computed after the
    affine mapping to [0, 1], matching inference/evaluation scale without
    clamping gradients at the range boundaries.
    """

    def __init__(
        self,
        *,
        data_range: float = 1.0,
        window_size: int = 11,
        sigma: float = 1.5,
        channel_mode: Literal["rgb", "gray"] = "rgb",
        reduction: Literal["mean", "sum", "none"] = "mean",
    ) -> None:
        super().__init__()
        if data_range <= 0:
            raise ValueError("data_range must be greater than 0")
        if window_size <= 0 or window_size % 2 == 0:
            raise ValueError("window_size must be a positive odd integer")
        if sigma <= 0:
            raise ValueError("sigma must be greater than 0")
        if channel_mode not in {"rgb", "gray"}:
            raise ValueError("channel_mode must be one of ['gray', 'rgb']")
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be one of ['mean', 'none', 'sum']")
        self.data_range = float(data_range)
        self.window_size = int(window_size)
        self.sigma = float(sigma)
        self.channel_mode = channel_mode
        self.reduction = reduction

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if prediction.shape != target.shape:
            raise ValueError(
                f"prediction and target must have the same shape. "
                f"Got {tuple(prediction.shape)} and {tuple(target.shape)}."
            )
        if prediction.ndim != 4:
            raise ValueError("prediction and target must be NCHW tensors")
        if prediction.shape[-2] < self.window_size or prediction.shape[-1] < self.window_size:
            raise ValueError(
                "prediction and target spatial dimensions must be at least "
                f"window_size={self.window_size}"
            )

        prediction_01 = (prediction + 1.0) * 0.5
        target_01 = (target + 1.0) * 0.5
        if self.channel_mode == "gray":
            prediction_01 = _rgb_to_gray_tensor(prediction_01)
            target_01 = _rgb_to_gray_tensor(target_01)

        channels = prediction_01.shape[1]
        window = _gaussian_window(
            self.window_size,
            self.sigma,
            channels,
            device=prediction_01.device,
            dtype=prediction_01.dtype,
        )
        padding = self.window_size // 2

        mu_x = F.conv2d(prediction_01, window, padding=padding, groups=channels)
        mu_y = F.conv2d(target_01, window, padding=padding, groups=channels)
        mu_x_sq = mu_x.pow(2)
        mu_y_sq = mu_y.pow(2)
        mu_xy = mu_x * mu_y

        sigma_x_sq = (
            F.conv2d(prediction_01 * prediction_01, window, padding=padding, groups=channels)
            - mu_x_sq
        )
        sigma_y_sq = (
            F.conv2d(target_01 * target_01, window, padding=padding, groups=channels) - mu_y_sq
        )
        sigma_xy = (
            F.conv2d(prediction_01 * target_01, window, padding=padding, groups=channels) - mu_xy
        )

        c1 = (0.01 * self.data_range) ** 2
        c2 = (0.03 * self.data_range) ** 2
        numerator = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
        denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
        ssim_map = numerator / denominator.clamp_min(torch.finfo(denominator.dtype).eps)
        loss = 1.0 - ssim_map.flatten(start_dim=1).mean(dim=1)

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def build_ssim_loss(params: dict[str, Any] | None = None) -> SsimLoss:
    params = {} if params is None else params
    return SsimLoss(
        data_range=float(params.get("data_range", 1.0)),
        window_size=int(params.get("window_size", 11)),
        sigma=float(params.get("sigma", 1.5)),
        channel_mode=cast(
            Literal["rgb", "gray"],
            _ssim_choice(params.get("channel_mode", "rgb"), "channel_mode", {"rgb", "gray"}),
        ),
        reduction=cast(
            Literal["mean", "sum", "none"],
            _ssim_choice(
                params.get("reduction", "mean"),
                "reduction",
                {"mean", "sum", "none"},
            ),
        ),
    )


def evaluate_loss_term(
    term: LossTermConfig,
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> LossTermResult:
    if term.name == "ssim":
        raw = build_ssim_loss(term.params)(prediction, target)
        return LossTermResult(name=term.name, raw=raw, weighted=raw * term.weight)
    raise ValueError(f"Unsupported loss term: {term.name!r}")


def _ssim_choice(raw: object, field_name: str, choices: set[str]) -> str:
    if not isinstance(raw, str):
        raise TypeError(f"{field_name} must be a string. Supported values: {sorted(choices)}.")
    if raw not in choices:
        raise ValueError(f"{field_name} must be one of {sorted(choices)}. Got {raw!r}.")
    return raw


def _gaussian_window(
    window_size: int,
    sigma: float,
    channels: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - (window_size - 1) / 2
    kernel_1d = torch.exp(-(coords**2) / (2 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    return kernel_2d.expand(channels, 1, window_size, window_size).contiguous()


def _rgb_to_gray_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.shape[1] == 1:
        return tensor
    if tensor.shape[1] < 3:
        raise ValueError("gray channel_mode requires either 1 or at least 3 channels")
    weights = tensor.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
    return (tensor[:, :3] * weights).sum(dim=1, keepdim=True)


class Pix2PixLoss:
    """Computes Pix2Pix discriminator and generator losses."""

    def __init__(self, l1_weight: float = 25.0) -> None:
        self.l1_weight = l1_weight
        self._bce = nn.BCEWithLogitsLoss()
        self._l1 = nn.L1Loss()

    def discriminator_loss(
        self,
        D_real: torch.Tensor,
        D_fake: torch.Tensor,
    ) -> torch.Tensor:
        real_label = torch.ones_like(D_real)
        fake_label = torch.zeros_like(D_fake)
        return self._bce(D_real, real_label) + self._bce(D_fake, fake_label)

    def generator_loss(
        self,
        D_fake: torch.Tensor,
        fake: torch.Tensor,
        real: torch.Tensor,
    ) -> torch.Tensor:
        real_label = torch.ones_like(D_fake)
        return self._bce(D_fake, real_label) + self._l1(fake, real) * self.l1_weight


def build_gan_loss(name: Literal["bce"], l1_weight: float = 25.0) -> Pix2PixLoss:
    if name == "bce":
        return Pix2PixLoss(l1_weight=l1_weight)
    raise ValueError(f"Unsupported gan_loss: {name!r}")
