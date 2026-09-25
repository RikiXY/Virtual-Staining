from __future__ import annotations

import logging
import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import NamedTuple

import torch
import torch.optim as optim
from torchvision.utils import save_image

from virtual_staining.config.losses import LossConfig
from virtual_staining.config.training import LearningRateSchedulerConfig, TrainingConfig
from virtual_staining.models.io_contract import denormalize_model_output
from virtual_staining.training.runtime import MethodMetrics

logger = logging.getLogger(__name__)


def is_amp_enabled(device: torch.device) -> bool:
    return isinstance(device, torch.device) and device.type == "cuda"


def save_images(
    path: Path,
    source_tensor: torch.Tensor,
    output: torch.Tensor,
    target: torch.Tensor,
    epoch: int,
    batch_index: int,
) -> None:
    save_image(
        denormalize_model_output(source_tensor), path / f"epoch{epoch}_batch{batch_index}_input.tif"
    )
    save_image(
        denormalize_model_output(output), path / f"epoch{epoch}_batch{batch_index}_output.tif"
    )
    save_image(
        denormalize_model_output(target), path / f"epoch{epoch}_batch{batch_index}_target.tif"
    )


def dataset_len(loader: torch.utils.data.DataLoader) -> int:
    assert loader.dataset is not None
    return len(loader.dataset)  # type: ignore[arg-type]  -- Dataset.__len__ exists at runtime but is absent from torch stubs


def unpack_batch(
    batch: object,
    device: torch.device,
    input_names: tuple[str, ...],
) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict[str, torch.Tensor]]:
    if not isinstance(batch, dict) or set(batch) != {"inputs", "target", "masks"}:
        raise TypeError("training batches must contain exactly inputs, target, and masks")
    raw_inputs, raw_target, raw_masks = batch["inputs"], batch["target"], batch["masks"]
    if not isinstance(raw_inputs, dict) or tuple(raw_inputs) != input_names:
        raise TypeError(f"training batch inputs must match configured names {input_names}")
    if not isinstance(raw_target, torch.Tensor) or raw_target.ndim != 4 or raw_target.shape[1] != 3:
        raise TypeError("training batch target must be an RGB NCHW tensor")
    inputs: dict[str, torch.Tensor] = {}
    for name in input_names:
        value = raw_inputs[name]
        if not isinstance(value, torch.Tensor) or value.ndim != 4 or value.shape[1] != 3:
            raise TypeError(f"training batch input {name!r} must be an RGB NCHW tensor")
        if value.shape[0] != raw_target.shape[0] or value.shape[2:] != raw_target.shape[2:]:
            raise ValueError(
                "training batch inputs and target must have matching batch/spatial shapes"
            )
        inputs[name] = value.to(device)
    if not isinstance(raw_masks, dict):
        raise TypeError("training batch masks must be a mapping")
    masks: dict[str, torch.Tensor] = {}
    for name, value in raw_masks.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"training batch mask {name!r} must be a tensor")
        masks[str(name)] = value.to(device)
    return inputs, raw_target.to(device), masks


def configured_loss_names(losses: LossConfig | None) -> list[str]:
    if losses is None:
        return []
    names = [f"generator_{term.name}" for term in losses.generator]
    names.extend(f"discriminator_{term.name}" for term in losses.discriminator)
    return names


def metrics_fieldnames(
    loss_names: list[str],
    *,
    metric_names: tuple[str, ...],
    component_total_names: tuple[str, ...],
    stage: str | None = None,
) -> list[str]:
    stages = (stage,) if stage in {"train", "val"} else ("train", "val")
    fields = ["epoch"]
    for selected_stage in stages:
        fields.extend(f"{name}_{selected_stage}" for name in metric_names)
    if loss_names:
        for selected_stage in stages:
            fields.extend(f"loss_{selected_stage}_total_{name}" for name in component_total_names)
    for selected_stage in stages:
        for term_name in loss_names:
            fields.extend(
                [
                    f"loss_{selected_stage}_raw_{term_name}",
                    f"loss_{selected_stage}_weighted_{term_name}",
                    f"loss_{selected_stage}_current_weight_{term_name}",
                ]
            )
    return fields


def _average_components(
    totals: dict[str, float],
    count: int,
    loss_names: list[str],
) -> dict[str, float]:
    if count == 0:
        return {}
    return {name: totals.get(name, 0.0) / count for name in loss_names}


def _accumulate_components(totals: dict[str, float], values: dict[str, float] | None) -> None:
    if values is None:
        return
    for name, value in values.items():
        totals[name] = totals.get(name, 0.0) + value


class ComponentAverages(NamedTuple):
    raw: dict[str, float]
    weighted: dict[str, float]
    current_weight: dict[str, float]


class LossComponentAccumulator:
    def __init__(self, loss_names: list[str]) -> None:
        self.loss_names = loss_names
        self.raw: dict[str, float] = {}
        self.weighted: dict[str, float] = {}
        self.current_weight: dict[str, float] = {}

    def add(
        self,
        *,
        raw: dict[str, float] | None,
        weighted: dict[str, float] | None,
        current_weight: dict[str, float] | None,
    ) -> None:
        _accumulate_components(self.raw, raw)
        _accumulate_components(self.weighted, weighted)
        _accumulate_components(self.current_weight, current_weight)

    def average(self, count: int) -> ComponentAverages:
        return ComponentAverages(
            raw=_average_components(self.raw, count, self.loss_names),
            weighted=_average_components(self.weighted, count, self.loss_names),
            current_weight=_average_components(self.current_weight, count, self.loss_names),
        )


Scheduler = optim.lr_scheduler.LRScheduler | optim.lr_scheduler.ReduceLROnPlateau


def build_lr_scheduler(training: TrainingConfig, optimizer: optim.Optimizer) -> Scheduler | None:
    scheduler_config = training.scheduler
    if scheduler_config.name == "none":
        return None
    if scheduler_config.name == "linear_decay":
        assert scheduler_config.decay_start_epoch is not None
        decay_start_epoch = scheduler_config.decay_start_epoch
        decay_span = max(1, training.epochs - decay_start_epoch)

        def lr_lambda(epoch: int) -> float:
            if epoch <= decay_start_epoch:
                return 1.0
            return max(0.0, 1.0 - (epoch - decay_start_epoch) / decay_span)

        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    if scheduler_config.name == "reduce_on_plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=scheduler_config.mode,
            factor=scheduler_config.factor,
            patience=scheduler_config.patience,
            min_lr=scheduler_config.min_lr,
        )
    raise AssertionError(f"Unsupported scheduler {scheduler_config.name!r}")


def step_lr_schedulers(
    scheduler_config: LearningRateSchedulerConfig,
    schedulers: Sequence[Scheduler | None],
    *,
    epoch: int,
    monitor_value: Callable[[], float | None] | None,
) -> bool:
    """Step method-owned schedulers; ``monitor_value`` is None when validation did not run."""
    if scheduler_config.name == "none":
        return False

    if scheduler_config.name == "linear_decay":
        for scheduler in schedulers:
            if scheduler is not None and not isinstance(
                scheduler,
                optim.lr_scheduler.ReduceLROnPlateau,
            ):
                scheduler.step()
        return True

    if scheduler_config.name == "reduce_on_plateau":
        if monitor_value is None:
            return False
        metric_value = monitor_value()
        if metric_value is None or not math.isfinite(metric_value):
            logger.warning(
                "Skipping learning-rate scheduler step at epoch %s because %s is unavailable",
                epoch,
                scheduler_config.monitor,
            )
            return False
        for scheduler in schedulers:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(metric_value)
        return True

    raise AssertionError(f"Unsupported scheduler {scheduler_config.name!r}")


def loss_validation_metric(metrics: MethodMetrics, name: str) -> float | None:
    """Resolve the loss-derived validation CSV column ``name`` from method metrics."""
    if name == "loss_G_val":
        return metrics.losses.get("loss_G")
    if name == "loss_D_val":
        return metrics.losses.get("loss_D")
    if name == "loss_val_total_generator":
        return metrics.component_totals.get("generator")
    if name == "loss_val_total_discriminator":
        return metrics.component_totals.get("discriminator")
    prefix_maps = (
        ("loss_val_raw_", metrics.raw),
        ("loss_val_weighted_", metrics.weighted),
        ("loss_val_current_weight_", metrics.current_weight),
    )
    for prefix, values in prefix_maps:
        if name.startswith(prefix):
            return values.get(name.removeprefix(prefix))
    return None
