from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import torch
from torchvision.utils import save_image

from virtual_staining.training.loss_config import LossConfig
from virtual_staining.training.results import EpochMetrics


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
    # Images are normalised to [-1, 1]; bring back to [0, 1] before saving.
    save_image((source_tensor * 0.5 + 0.5), path / f"epoch{epoch}_batch{batch_index}_input.tif")
    save_image((output * 0.5 + 0.5), path / f"epoch{epoch}_batch{batch_index}_output.tif")
    save_image((target * 0.5 + 0.5), path / f"epoch{epoch}_batch{batch_index}_target.tif")


def dataset_len(loader: torch.utils.data.DataLoader) -> int:
    assert loader.dataset is not None
    return len(loader.dataset)  # type: ignore[arg-type]  -- Dataset.__len__ exists at runtime but is absent from torch stubs


def unpack_batch(
    batch: object,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor] | None]:
    if not isinstance(batch, (tuple, list)) or len(batch) not in {2, 3}:
        raise TypeError("training batches must contain (source, target) or (source, target, masks)")
    x = batch[0]
    y = batch[1]
    if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
        raise TypeError("training batch source and target must be tensors")
    masks = None
    if len(batch) == 3:
        raw_masks = batch[2]
        if not isinstance(raw_masks, dict):
            raise TypeError("training batch masks must be a mapping")
        masks = {}
        for name, value in raw_masks.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"training batch mask {name!r} must be a tensor")
            masks[str(name)] = value.to(device)
    return x.to(device), y.to(device), masks


def configured_loss_names(losses: LossConfig | None) -> list[str]:
    if losses is None:
        return []
    names = [f"generator_{term.name}" for term in losses.generator]
    names.extend(f"discriminator_{term.name}" for term in losses.discriminator)
    return names


def metrics_fieldnames(loss_names: list[str], *, stage: str | None = None) -> list[str]:
    if stage == "train":
        fields = ["epoch", "loss_G_train", "loss_D_train"]
        stages = ("train",)
    elif stage == "val":
        fields = ["epoch", "loss_G_val", "loss_D_val"]
        stages = ("val",)
    else:
        fields = [
            "epoch",
            "loss_G_train",
            "loss_D_train",
            "loss_G_val",
            "loss_D_val",
        ]
        stages = ("train", "val")
    if loss_names:
        for selected_stage in stages:
            fields.extend(
                [
                    f"loss_{selected_stage}_total_generator",
                    f"loss_{selected_stage}_total_discriminator",
                ]
            )
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


def component_metric_row(stage: str, metrics: EpochMetrics | None) -> dict[str, str]:
    if metrics is None:
        return {}
    if not metrics.raw and not metrics.weighted and not metrics.current_weight:
        return {}
    row = {
        f"loss_{stage}_total_generator": f"{metrics.loss_G:.6f}",
        f"loss_{stage}_total_discriminator": f"{metrics.loss_D:.6f}",
    }
    for term_name in sorted(metrics.raw):
        row[f"loss_{stage}_raw_{term_name}"] = f"{metrics.raw[term_name]:.6f}"
    for term_name in sorted(metrics.weighted):
        row[f"loss_{stage}_weighted_{term_name}"] = f"{metrics.weighted[term_name]:.6f}"
    for term_name in sorted(metrics.current_weight):
        row[f"loss_{stage}_current_weight_{term_name}"] = f"{metrics.current_weight[term_name]:.6f}"
    return row


def average_components(
    totals: dict[str, float],
    count: int,
    loss_names: list[str],
) -> dict[str, float]:
    if count == 0:
        return {}
    return {name: totals.get(name, 0.0) / count for name in loss_names}


def accumulate_components(totals: dict[str, float], values: dict[str, float] | None) -> None:
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
        accumulate_components(self.raw, raw)
        accumulate_components(self.weighted, weighted)
        accumulate_components(self.current_weight, current_weight)

    def average(self, count: int) -> ComponentAverages:
        return ComponentAverages(
            raw=average_components(self.raw, count, self.loss_names),
            weighted=average_components(self.weighted, count, self.loss_names),
            current_weight=average_components(self.current_weight, count, self.loss_names),
        )
