"""Canonical definitions of the built-in configured losses.

Each ``LossDefinition`` owns one loss's name, allowed roles, supported methods,
parameter contract, and primitive tensor math. Configuration validation and method
runtimes both resolve losses here. Methods still own objective composition (which
tensors a primitive is applied to and how terms are summed), and
``LossScheduleConfig`` still owns the current scalar weight.

Torch is imported inside the primitives so configuration loading stays torch-free.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, cast

from virtual_staining.config.validation import (
    parse_bool_strict,
    parse_choice,
    parse_int,
    reject_unknown_keys,
    require_finite,
)

if TYPE_CHECKING:
    import torch

LossRole = Literal["generator", "discriminator"]
LossContext = Literal["adversarial", "reconstruction"]
LossReduction = Literal["mean", "sum", "none"]
LossMaskSource = Literal["foreground_mask"]
SsimChannelMode = Literal["rgb", "gray"]

_REDUCTIONS = {"mean", "sum", "none"}
_LOSS_MASK_KEYS: frozenset[str] = frozenset(
    {"enabled", "source", "foreground_weight", "background_weight", "ignore_empty_mask"}
)


@dataclass(frozen=True)
class LossMaskConfig:
    enabled: bool = False
    source: LossMaskSource = "foreground_mask"
    foreground_weight: float = 1.0
    background_weight: float = 1.0
    ignore_empty_mask: bool = True

    def validate(self) -> None:
        if self.source != "foreground_mask":
            raise ValueError("loss mask source must be one of ['foreground_mask']")
        for name in ("foreground_weight", "background_weight"):
            value = getattr(self, name)
            require_finite(value, f"loss mask {name}")
            if value < 0:
                raise ValueError(f"loss mask {name} must be greater than or equal to 0")

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "source": self.source,
            "foreground_weight": self.foreground_weight,
            "background_weight": self.background_weight,
            "ignore_empty_mask": self.ignore_empty_mask,
        }


def parse_loss_mask_config(raw: Any, context: str = "loss mask") -> LossMaskConfig:
    if raw is None:
        return LossMaskConfig()
    if not isinstance(raw, dict):
        raise TypeError(f"{context} must be a YAML mapping")
    reject_unknown_keys(raw, _LOSS_MASK_KEYS, context)
    source = parse_choice(
        raw.get("source", "foreground_mask"), f"{context}.source", {"foreground_mask"}
    )
    config = LossMaskConfig(
        enabled=parse_bool_strict(raw.get("enabled", False), f"{context}.enabled"),
        source=cast(LossMaskSource, source),
        foreground_weight=float(raw.get("foreground_weight", 1.0)),
        background_weight=float(raw.get("background_weight", 1.0)),
        ignore_empty_mask=parse_bool_strict(
            raw.get("ignore_empty_mask", True), f"{context}.ignore_empty_mask"
        ),
    )
    config.validate()
    return config


@dataclass(frozen=True)
class LossParams:
    """Resolved parameters; a definition's ``param_keys`` limit which may be configured."""

    reduction: LossReduction = "mean"
    mask: LossMaskConfig = field(default_factory=LossMaskConfig)
    data_range: float = 1.0
    window_size: int = 11
    sigma: float = 1.5
    channel_mode: SsimChannelMode = "rgb"


def _positive_float(value: Any, field_name: str) -> float:
    number = float(value)
    require_finite(number, field_name)
    if number <= 0:
        raise ValueError(f"{field_name} must be greater than 0")
    return number


def _positive_odd_int(value: Any, field_name: str) -> int:
    number = parse_int(value, field_name)
    if number <= 0 or number % 2 == 0:
        raise ValueError(f"{field_name} must be a positive odd integer")
    return number


_PARAM_PARSERS: Mapping[str, Callable[[Any, str], Any]] = MappingProxyType(
    {
        "reduction": lambda value, name: parse_choice(value, name, _REDUCTIONS),
        "mask": parse_loss_mask_config,
        "data_range": _positive_float,
        "window_size": _positive_odd_int,
        "sigma": _positive_float,
        "channel_mode": lambda value, name: parse_choice(value, name, {"rgb", "gray"}),
    }
)

AdversarialPrimitive = Callable[["torch.Tensor", bool], "torch.Tensor"]
ReconstructionPrimitive = Callable[
    ["torch.Tensor", "torch.Tensor", LossParams, "torch.Tensor | None"], "torch.Tensor"
]


@dataclass(frozen=True)
class LossDefinition:
    """One built-in loss: metadata, parameter contract, and exactly one primitive.

    ``adversarial`` primitives score discriminator outputs against a real/fake target.
    ``reconstruction`` primitives compare a prediction with a target image.
    """

    name: str
    roles: frozenset[LossRole]
    methods: frozenset[str]
    param_keys: frozenset[str] = frozenset()
    adversarial: AdversarialPrimitive | None = None
    reconstruction: ReconstructionPrimitive | None = None

    def __post_init__(self) -> None:
        if (self.adversarial is None) == (self.reconstruction is None):
            raise ValueError(f"loss '{self.name}' must define exactly one primitive")
        unknown = self.param_keys - set(_PARAM_PARSERS)
        if unknown:
            raise ValueError(f"loss '{self.name}' declares unknown params {sorted(unknown)}")

    @property
    def context(self) -> LossContext:
        return "adversarial" if self.adversarial is not None else "reconstruction"

    @property
    def supports_mask(self) -> bool:
        return "mask" in self.param_keys

    def validate_role(self, role: LossRole) -> None:
        if role not in self.roles:
            only = " and ".join(f"losses.{name}" for name in sorted(self.roles))
            raise ValueError(f"loss '{self.name}' is supported only in {only}")

    def parse_params(self, raw: Mapping[str, Any]) -> LossParams:
        context = f"loss '{self.name}' params"
        reject_unknown_keys(raw, self.param_keys, context)
        return LossParams(
            **{key: _PARAM_PARSERS[key](value, f"{context}.{key}") for key, value in raw.items()}
        )

    def adversarial_loss(self, logits: torch.Tensor, *, target_is_real: bool) -> torch.Tensor:
        if self.adversarial is None:
            raise ValueError(f"loss '{self.name}' is not an adversarial loss")
        return self.adversarial(logits, target_is_real)

    def reconstruction_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        params: LossParams | None = None,
        *,
        masks: Mapping[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if self.reconstruction is None:
            raise ValueError(f"loss '{self.name}' is not a reconstruction loss")
        params = LossParams() if params is None else params
        mask = None
        if params.mask.enabled:
            if masks is None or params.mask.source not in masks:
                raise ValueError(
                    f"loss '{self.name}' requires batch mask '{params.mask.source}', "
                    "but the training batch did not provide it"
                )
            mask = masks[params.mask.source]
        return self.reconstruction(prediction, target, params, mask)


def _adversarial_bce(logits: torch.Tensor, target_is_real: bool) -> torch.Tensor:
    import torch
    import torch.nn.functional as F

    labels = torch.ones_like(logits) if target_is_real else torch.zeros_like(logits)
    return F.binary_cross_entropy_with_logits(logits, labels)


def _adversarial_lsgan(prediction: torch.Tensor, target_is_real: bool) -> torch.Tensor:
    import torch
    import torch.nn.functional as F

    target = torch.ones_like(prediction) if target_is_real else torch.zeros_like(prediction)
    return F.mse_loss(prediction, target)


def _l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    params: LossParams,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    import torch
    import torch.nn.functional as F

    if mask is None:
        return F.l1_loss(prediction, target, reduction=params.reduction)
    return _reduce_masked_loss(torch.abs(prediction - target), mask, params)


def _ssim(
    prediction: torch.Tensor,
    target: torch.Tensor,
    params: LossParams,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    loss_map = ssim_loss_map(prediction, target, params)
    if mask is not None:
        return _reduce_masked_loss(loss_map, mask, params)
    loss = loss_map.flatten(start_dim=1).mean(dim=1)
    if params.reduction == "mean":
        return loss.mean()
    if params.reduction == "sum":
        return loss.sum()
    return loss


LOSS_DEFINITIONS: Mapping[str, LossDefinition] = MappingProxyType(
    {
        definition.name: definition
        for definition in (
            LossDefinition(
                name="adversarial_bce",
                roles=frozenset({"generator", "discriminator"}),
                methods=frozenset({"pix2pix"}),
                adversarial=_adversarial_bce,
            ),
            LossDefinition(
                name="l1",
                roles=frozenset({"generator"}),
                methods=frozenset({"pix2pix"}),
                param_keys=frozenset({"reduction", "mask"}),
                reconstruction=_l1,
            ),
            LossDefinition(
                name="ssim",
                roles=frozenset({"generator"}),
                methods=frozenset({"pix2pix"}),
                param_keys=frozenset(
                    {"data_range", "window_size", "sigma", "channel_mode", "reduction", "mask"}
                ),
                reconstruction=_ssim,
            ),
            LossDefinition(
                name="adversarial_lsgan",
                roles=frozenset({"generator", "discriminator"}),
                methods=frozenset({"cyclegan"}),
                adversarial=_adversarial_lsgan,
            ),
            LossDefinition(
                name="cycle_l1",
                roles=frozenset({"generator"}),
                methods=frozenset({"cyclegan"}),
                reconstruction=_l1,
            ),
            LossDefinition(
                name="identity_l1",
                roles=frozenset({"generator"}),
                methods=frozenset({"cyclegan"}),
                reconstruction=_l1,
            ),
        )
    }
)


def method_loss_names(method: str) -> frozenset[str]:
    return frozenset(name for name, d in LOSS_DEFINITIONS.items() if method in d.methods)


def ssim_loss_map(
    prediction: torch.Tensor,
    target: torch.Tensor,
    params: LossParams | None = None,
) -> torch.Tensor:
    """Return the differentiable per-pixel ``1 - SSIM`` map for tensors in [-1, 1].

    SSIM is computed after the affine mapping to [0, 1], matching inference/evaluation
    scale without clamping gradients at the range boundaries.
    """
    import torch
    import torch.nn.functional as F
    from torch.amp import autocast

    params = LossParams() if params is None else params
    window_size = params.window_size
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction and target must have the same shape. "
            f"Got {tuple(prediction.shape)} and {tuple(target.shape)}."
        )
    if prediction.ndim != 4:
        raise ValueError("prediction and target must be NCHW tensors")
    if prediction.shape[-2] < window_size or prediction.shape[-1] < window_size:
        raise ValueError(
            f"prediction and target spatial dimensions must be at least window_size={window_size}"
        )

    with autocast(device_type=prediction.device.type, enabled=False):
        compute_dtype = (
            torch.float32
            if prediction.dtype in {torch.float16, torch.bfloat16}
            else prediction.dtype
        )
        prediction_01 = (prediction.to(dtype=compute_dtype) + 1.0) * 0.5
        target_01 = (target.to(dtype=compute_dtype) + 1.0) * 0.5
        if params.channel_mode == "gray":
            prediction_01 = _rgb_to_gray_tensor(prediction_01)
            target_01 = _rgb_to_gray_tensor(target_01)

        channels = prediction_01.shape[1]
        window = _gaussian_window(
            window_size,
            params.sigma,
            channels,
            device=prediction_01.device,
            dtype=prediction_01.dtype,
        )
        padding = window_size // 2

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

        c1 = (0.01 * params.data_range) ** 2
        c2 = (0.03 * params.data_range) ** 2
        numerator = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
        denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
        ssim_map = numerator / denominator.clamp_min(torch.finfo(denominator.dtype).eps)
        return 1.0 - ssim_map


def _reduce_masked_loss(
    loss_map: torch.Tensor,
    mask: torch.Tensor,
    params: LossParams,
) -> torch.Tensor:
    import torch

    mask_config = params.mask
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
        if params.reduction == "none":
            return empty.reshape(1)[:0]
        return empty

    values = torch.stack(sample_values)
    if params.reduction == "mean":
        return values.mean()
    if params.reduction == "sum":
        return values.sum()
    return values


def _gaussian_window(
    window_size: int,
    sigma: float,
    channels: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    import torch

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
