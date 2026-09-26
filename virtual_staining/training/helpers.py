from __future__ import annotations

import logging
import math
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler
from torchvision.utils import save_image

from virtual_staining.checkpoint_contract import CheckpointCompatibilityError, first_difference
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


# Constructor policy persisted per optimizer role; learning rates are the configured initial
# values from ``optimizer.defaults``, never the scheduler-decayed ``param_groups`` values.
_OPTIMIZER_POLICY_KEYS = ("lr", "betas", "eps", "weight_decay", "amsgrad", "maximize")
_SCALER_STATE_KEYS = frozenset(
    {"scale", "growth_factor", "backoff_factor", "growth_interval", "_growth_tracker"}
)
TRAINING_STATE_KEYS = frozenset({"models", "optimization", "optimizers", "scalers", "schedulers"})


@dataclass(frozen=True)
class OptimizationRole:
    """One method-owned optimizer with its AMP scaler and optional LR scheduler."""

    optimizer: optim.Optimizer
    scaler: GradScaler
    scheduler: Scheduler | None


def state_error(context: str, detail: str) -> CheckpointCompatibilityError:
    return CheckpointCompatibilityError(f"method state {context} {detail}")


def require_state_keys(value: object, keys: Collection[str], context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise state_error(context, "must be a mapping")
    if set(value) != set(keys):
        missing = sorted(set(keys) - set(value))
        unexpected = sorted(map(str, set(value) - set(keys)))
        raise state_error(
            context, f"has mismatched keys: missing {missing}, unexpected {unexpected}"
        )
    return value


def check_module_state(stored: object, module: nn.Module, context: str) -> None:
    """Check ``stored`` has exactly ``module``'s state keys, tensor shapes, and dtypes."""
    expected = module.state_dict()
    stored = require_state_keys(stored, expected.keys(), context)
    for key, tensor in expected.items():
        value = stored[key]
        if not isinstance(value, torch.Tensor):
            raise state_error(f"{context}.{key}", "must be a tensor")
        if value.shape != tensor.shape:
            raise state_error(
                f"{context}.{key}",
                f"has shape {tuple(value.shape)}; expected {tuple(tensor.shape)}",
            )
        if value.dtype != tensor.dtype:
            raise state_error(
                f"{context}.{key}", f"has dtype {value.dtype}; expected {tensor.dtype}"
            )


def validated_model_state(
    state: Mapping[str, Any], name: str, module: nn.Module, checkpoint_path: Path
) -> Mapping[str, Any]:
    """Return ``state.models[name]`` after checking it fits ``module`` (inference loading)."""
    try:
        models = state.get("models")
        if not isinstance(models, Mapping) or name not in models:
            raise state_error("state.models", f"has no {name!r} entry")
        check_module_state(models[name], module, f"state.models.{name}")
    except CheckpointCompatibilityError as exc:
        raise CheckpointCompatibilityError(
            f"Checkpoint '{checkpoint_path}' is incompatible: {exc}"
        ) from exc
    return models[name]


def scheduler_policy(training: TrainingConfig) -> dict[str, Any]:
    """Return the reconstruction-relevant policy of the scheduler ``build_lr_scheduler`` makes."""
    policy = training.scheduler.to_dict()
    if training.scheduler.name == "linear_decay":
        # LambdaLR state does not carry its closure; the decay horizon is fixed by this basis.
        policy["epochs"] = training.epochs
    return policy


def optimization_identity(training: TrainingConfig, role: OptimizationRole) -> dict[str, Any]:
    defaults = role.optimizer.defaults
    optimizer: dict[str, Any] = {"class": type(role.optimizer).__name__}
    for key in _OPTIMIZER_POLICY_KEYS:
        value = defaults[key]
        optimizer[key] = list(value) if isinstance(value, tuple) else value
    return {
        "optimizer": optimizer,
        "scheduler": None if role.scheduler is None else scheduler_policy(training),
    }


def _check_identity(stored: object, current: dict[str, Any], context: str) -> None:
    if stored == current:
        return
    field, stored_value, current_value = first_difference(stored, current, context) or (
        context,
        stored,
        current,
    )
    raise CheckpointCompatibilityError(
        f"method state {field} is {stored_value!r} in the checkpoint but {current_value!r} in "
        "the current run; resume requires the same optimizer and scheduler policy."
    )


def _check_optimizer_state(stored: object, optimizer: optim.Optimizer, context: str) -> None:
    stored = require_state_keys(stored, ("state", "param_groups"), context)
    current_groups = optimizer.state_dict()["param_groups"]
    groups = stored["param_groups"]
    if not isinstance(groups, list) or len(groups) != len(current_groups):
        raise state_error(
            f"{context}.param_groups", f"must be a list of {len(current_groups)} groups"
        )
    for index, (group, current) in enumerate(zip(groups, current_groups, strict=True)):
        group_context = f"{context}.param_groups[{index}]"
        group = require_state_keys(group, current.keys(), group_context)
        if group["params"] != current["params"]:
            raise state_error(f"{group_context}.params", "does not match the optimizer parameters")
    params = [param for group in optimizer.param_groups for param in group["params"]]
    param_state = stored["state"]
    if not isinstance(param_state, Mapping):
        raise state_error(f"{context}.state", "must be a mapping")
    for key, values in param_state.items():
        if type(key) is not int or not 0 <= key < len(params):
            raise state_error(f"{context}.state", f"has unknown parameter index {key!r}")
        if not isinstance(values, Mapping):
            raise state_error(f"{context}.state[{key}]", "must be a mapping")
        for name, value in values.items():
            if not isinstance(value, torch.Tensor):
                raise state_error(f"{context}.state[{key}].{name}", "must be a tensor")
            if value.ndim and value.shape != params[key].shape:
                raise state_error(
                    f"{context}.state[{key}].{name}",
                    f"has shape {tuple(value.shape)}; expected {tuple(params[key].shape)}",
                )


def _check_scaler_state(stored: object, scaler: GradScaler, context: str) -> None:
    if not isinstance(stored, Mapping):
        raise state_error(context, "must be a mapping")
    if not stored:
        if scaler.is_enabled():
            raise state_error(
                context, "was saved without AMP scaling; resume on the same device type"
            )
        return
    require_state_keys(stored, _SCALER_STATE_KEYS, context)
    for key, value in stored.items():
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise state_error(f"{context}.{key}", "must be a number")


def _check_scheduler_state(stored: object, scheduler: Scheduler | None, context: str) -> None:
    if scheduler is None:
        if stored is not None:
            raise state_error(context, "is present but the current run has no scheduler")
        return
    if stored is None:
        raise state_error(context, "is absent but the current run configures a scheduler")
    require_state_keys(stored, scheduler.state_dict().keys(), context)


def training_state_dict(
    training: TrainingConfig,
    models: Mapping[str, nn.Module],
    roles: Mapping[str, OptimizationRole],
) -> dict[str, Any]:
    """Serialize method-owned models plus per-role optimizer, scaler, and scheduler state."""
    return {
        "models": {name: model.state_dict() for name, model in models.items()},
        "optimization": {
            name: optimization_identity(training, role) for name, role in roles.items()
        },
        "optimizers": {name: role.optimizer.state_dict() for name, role in roles.items()},
        "scalers": {name: role.scaler.state_dict() for name, role in roles.items()},
        "schedulers": {
            name: None if role.scheduler is None else role.scheduler.state_dict()
            for name, role in roles.items()
        },
    }


def check_training_state(
    state: Mapping[str, Any],
    training: TrainingConfig,
    models: Mapping[str, nn.Module],
    roles: Mapping[str, OptimizationRole],
) -> None:
    """Preflight the ``training_state_dict`` groups of ``state`` without mutating anything."""
    stored_models = require_state_keys(state["models"], models.keys(), "state.models")
    for name, model in models.items():
        check_module_state(stored_models[name], model, f"state.models.{name}")
    identities = require_state_keys(state["optimization"], roles.keys(), "state.optimization")
    for name, role in roles.items():
        _check_identity(
            identities[name], optimization_identity(training, role), f"state.optimization.{name}"
        )
    optimizers = require_state_keys(state["optimizers"], roles.keys(), "state.optimizers")
    scalers = require_state_keys(state["scalers"], roles.keys(), "state.scalers")
    schedulers = require_state_keys(state["schedulers"], roles.keys(), "state.schedulers")
    for name, role in roles.items():
        _check_optimizer_state(optimizers[name], role.optimizer, f"state.optimizers.{name}")
        _check_scaler_state(scalers[name], role.scaler, f"state.scalers.{name}")
        _check_scheduler_state(schedulers[name], role.scheduler, f"state.schedulers.{name}")


def load_training_state(
    state: Mapping[str, Any],
    models: Mapping[str, nn.Module],
    roles: Mapping[str, OptimizationRole],
) -> None:
    """Restore groups already accepted by ``check_training_state``."""
    for name, model in models.items():
        model.load_state_dict(state["models"][name])
    for name, role in roles.items():
        role.optimizer.load_state_dict(state["optimizers"][name])
        role.scaler.load_state_dict(state["scalers"][name])
        if role.scheduler is not None:
            role.scheduler.load_state_dict(state["schedulers"][name])


@contextmanager
def restoring_validated_state(method: str) -> Iterator[None]:
    """Flag a restore that failed after preflight: the runtime is partially mutated."""
    try:
        yield
    except Exception as exc:
        raise RuntimeError(
            f"{method} state restoration failed after validation; this runtime is partially "
            "restored and must be discarded and rebuilt."
        ) from exc
