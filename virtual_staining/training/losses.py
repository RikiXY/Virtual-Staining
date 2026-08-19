from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast

from virtual_staining.training.loss_config import LossMaskConfig, LossTermConfig


@dataclass(frozen=True)
class LossRegistryEntry:
    name: str
    roles: tuple[Literal["generator", "discriminator"], ...]
    targets: tuple[str, ...]
    default_weight: float = 0.0


LOSS_REGISTRY: dict[str, LossRegistryEntry] = {
    "adversarial_bce": LossRegistryEntry(
        name="adversarial_bce",
        roles=("generator", "discriminator"),
        targets=("discriminator_logits",),
    ),
    "l1": LossRegistryEntry(name="l1", roles=("generator",), targets=("image",)),
    "ssim": LossRegistryEntry(name="ssim", roles=("generator",), targets=("image",)),
}


@dataclass
class StepLosses:
    loss_G: float
    loss_D: float
    raw: dict[str, float] | None = None
    weighted: dict[str, float] | None = None
    current_weight: dict[str, float] | None = None


@dataclass(frozen=True)
class LossTermResult:
    name: str
    raw: torch.Tensor
    weighted: torch.Tensor
    current_weight: float
    stage: Literal["generator", "discriminator"] = "generator"

    @property
    def component_key(self) -> str:
        return f"{self.stage}_{self.name}"


@dataclass(frozen=True)
class LossEvaluationContext:
    epoch: int = 0
    global_step: int | None = None
    masks: dict[str, torch.Tensor] | None = None


@dataclass
class LossAggregate:
    total: torch.Tensor
    raw: dict[str, float]
    weighted: dict[str, float]
    current_weight: dict[str, float]


class ConfiguredLossEvaluator:
    """Evaluates configured loss terms for training and validation."""

    def __init__(
        self,
        *,
        generator_terms: tuple[LossTermConfig, ...] = (),
        discriminator_terms: tuple[LossTermConfig, ...] = (),
    ) -> None:
        self.generator_terms = generator_terms
        self.discriminator_terms = discriminator_terms

    def generator_total(
        self,
        *,
        prediction: torch.Tensor,
        target: torch.Tensor,
        discriminator_fake: torch.Tensor | None = None,
        context: LossEvaluationContext,
    ) -> LossAggregate:
        total = prediction.sum() * 0.0
        results: list[LossTermResult] = []
        for term in self.generator_terms:
            result = evaluate_generator_loss_term(
                term,
                prediction,
                target,
                discriminator_fake=discriminator_fake,
                epoch=context.epoch,
                global_step=context.global_step,
                masks=context.masks,
            )
            total = total + result.weighted
            results.append(result)
        return _aggregate_loss_results(total, results)

    def discriminator_total(
        self,
        *,
        discriminator_real: torch.Tensor,
        discriminator_fake: torch.Tensor,
        context: LossEvaluationContext,
    ) -> LossAggregate:
        total = discriminator_real.sum() * 0.0
        results: list[LossTermResult] = []
        for term in self.discriminator_terms:
            result = evaluate_discriminator_loss_term(
                term,
                discriminator_real,
                discriminator_fake,
                epoch=context.epoch,
                global_step=context.global_step,
            )
            total = total + result.weighted
            results.append(result)
        return _aggregate_loss_results(total, results)


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

    def loss_map(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
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

        with autocast(device_type=prediction.device.type, enabled=False):
            compute_dtype = (
                torch.float32
                if prediction.dtype in {torch.float16, torch.bfloat16}
                else prediction.dtype
            )
            prediction_01 = (prediction.to(dtype=compute_dtype) + 1.0) * 0.5
            target_01 = (target.to(dtype=compute_dtype) + 1.0) * 0.5
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
                F.conv2d(prediction_01 * target_01, window, padding=padding, groups=channels)
                - mu_xy
            )

            c1 = (0.01 * self.data_range) ** 2
            c2 = (0.03 * self.data_range) ** 2
            numerator = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
            denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
            ssim_map = numerator / denominator.clamp_min(torch.finfo(denominator.dtype).eps)
            return 1.0 - ssim_map

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss_map = self.loss_map(prediction, target)
        loss = loss_map.flatten(start_dim=1).mean(dim=1)

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


def build_l1_loss(params: dict[str, Any] | None = None) -> nn.L1Loss:
    params = {} if params is None else params
    reduction = cast(
        Literal["mean", "sum", "none"],
        _ssim_choice(params.get("reduction", "mean"), "reduction", {"mean", "sum", "none"}),
    )
    return nn.L1Loss(reduction=reduction)


def evaluate_loss_term(
    term: LossTermConfig,
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    epoch: int = 0,
    global_step: int | None = None,
    masks: dict[str, torch.Tensor] | None = None,
    discriminator_fake: torch.Tensor | None = None,
) -> LossTermResult:
    return evaluate_generator_loss_term(
        term,
        prediction,
        target,
        discriminator_fake=discriminator_fake,
        epoch=epoch,
        global_step=global_step,
        masks=masks,
    )


def evaluate_generator_loss_term(
    term: LossTermConfig,
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    discriminator_fake: torch.Tensor | None = None,
    epoch: int = 0,
    global_step: int | None = None,
    masks: dict[str, torch.Tensor] | None = None,
) -> LossTermResult:
    current_weight = term.current_weight(epoch=epoch, global_step=global_step)
    if term.name == "ssim":
        raw = _ensure_scalar(_evaluate_ssim_term(term, prediction, target, masks=masks))
    elif term.name == "l1":
        raw = _ensure_scalar(_evaluate_l1_term(term, prediction, target, masks=masks))
    elif term.name == "adversarial_bce":
        if discriminator_fake is None:
            raise ValueError("generator adversarial_bce loss requires discriminator_fake logits")
        raw = _bce_with_logits(discriminator_fake, torch.ones_like(discriminator_fake))
    else:
        raise ValueError(f"Unsupported generator loss term: {term.name!r}")
    return LossTermResult(
        name=term.name,
        raw=raw,
        weighted=raw * current_weight,
        current_weight=current_weight,
        stage="generator",
    )


def evaluate_discriminator_loss_term(
    term: LossTermConfig,
    discriminator_real: torch.Tensor,
    discriminator_fake: torch.Tensor,
    *,
    epoch: int = 0,
    global_step: int | None = None,
) -> LossTermResult:
    current_weight = term.current_weight(epoch=epoch, global_step=global_step)
    if term.name == "adversarial_bce":
        raw = _bce_with_logits(
            discriminator_real, torch.ones_like(discriminator_real)
        ) + _bce_with_logits(discriminator_fake, torch.zeros_like(discriminator_fake))
        return LossTermResult(
            name=term.name,
            raw=raw,
            weighted=raw * current_weight,
            current_weight=current_weight,
            stage="discriminator",
        )
    raise ValueError(f"Unsupported discriminator loss term: {term.name!r}")


def _evaluate_ssim_term(
    term: LossTermConfig,
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    masks: dict[str, torch.Tensor] | None,
) -> torch.Tensor:
    loss = build_ssim_loss(term.params)
    mask_config = term.mask
    if not mask_config.enabled:
        return loss(prediction, target)
    if masks is None or mask_config.source not in masks:
        raise ValueError(
            f"loss '{term.name}' requires batch mask '{mask_config.source}', "
            "but the training batch did not provide it"
        )
    loss_map = loss.loss_map(prediction, target)
    return reduce_masked_loss(
        loss_map,
        masks[mask_config.source],
        mask_config=mask_config,
        reduction=cast(Literal["mean", "sum", "none"], loss.reduction),
    )


def _evaluate_l1_term(
    term: LossTermConfig,
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    masks: dict[str, torch.Tensor] | None,
) -> torch.Tensor:
    mask_config = term.mask
    if not mask_config.enabled:
        return build_l1_loss(term.params)(prediction, target)
    if masks is None or mask_config.source not in masks:
        raise ValueError(
            f"loss '{term.name}' requires batch mask '{mask_config.source}', "
            "but the training batch did not provide it"
        )
    loss_map = torch.abs(prediction - target)
    reduction = cast(
        Literal["mean", "sum", "none"],
        _ssim_choice(term.params.get("reduction", "mean"), "reduction", {"mean", "sum", "none"}),
    )
    return reduce_masked_loss(
        loss_map,
        masks[mask_config.source],
        mask_config=mask_config,
        reduction=reduction,
    )


def _bce_with_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, labels)


def _ensure_scalar(value: torch.Tensor) -> torch.Tensor:
    if value.ndim == 0:
        return value
    return value.mean()


def _aggregate_loss_results(
    total: torch.Tensor,
    results: list[LossTermResult],
) -> LossAggregate:
    raw: dict[str, float] = {}
    weighted: dict[str, float] = {}
    current_weight: dict[str, float] = {}
    for result in results:
        raw[result.component_key] = float(result.raw.detach().item())
        weighted[result.component_key] = float(result.weighted.detach().item())
        current_weight[result.component_key] = result.current_weight
    return LossAggregate(
        total=total,
        raw=raw,
        weighted=weighted,
        current_weight=current_weight,
    )


def reduce_masked_loss(
    loss_map: torch.Tensor,
    mask: torch.Tensor,
    *,
    mask_config: LossMaskConfig,
    reduction: Literal["mean", "sum", "none"] = "mean",
) -> torch.Tensor:
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    if mask.ndim != 4:
        raise ValueError("foreground_mask must be an NCHW or NHW tensor")
    if mask.shape[0] != loss_map.shape[0]:
        raise ValueError("foreground_mask batch dimension must match loss tensor")
    if mask.shape[-2:] != loss_map.shape[-2:]:
        raise ValueError("foreground_mask spatial dimensions must match loss tensor")
    mask = mask.to(device=loss_map.device, dtype=loss_map.dtype)
    foreground = mask > 0.5
    weights = torch.where(
        foreground,
        loss_map.new_tensor(mask_config.foreground_weight),
        loss_map.new_tensor(mask_config.background_weight),
    )
    if loss_map.shape[1] != weights.shape[1]:
        weights = weights.expand(-1, loss_map.shape[1], -1, -1)

    sample_values: list[torch.Tensor] = []
    for index in range(loss_map.shape[0]):
        sample_weights = weights[index]
        if mask_config.ignore_empty_mask and not foreground[index].any():
            continue
        denom = sample_weights.sum().clamp_min(torch.finfo(loss_map.dtype).eps)
        sample_values.append((loss_map[index] * sample_weights).sum() / denom)

    if not sample_values:
        empty = loss_map.sum() * 0.0
        if reduction == "none":
            return empty.reshape(1)[:0]
        return empty

    values = torch.stack(sample_values)
    if reduction == "mean":
        return values.mean()
    if reduction == "sum":
        return values.sum()
    return values


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
