from __future__ import annotations

import itertools
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import GradScaler, autocast

from virtual_staining.checkpoint_contract import (
    CheckpointIdentity,
    read_checkpoint,
    validate_checkpoint,
)
from virtual_staining.config.inference import InferenceDirection
from virtual_staining.config.model import ModelConfig
from virtual_staining.config.run import RunConfig
from virtual_staining.models.discriminator import PatchGANDiscriminator
from virtual_staining.models.factory import build_discriminator, build_resnet_generator
from virtual_staining.models.generator import ResnetGenerator
from virtual_staining.models.io_contract import GENERATOR_OUTPUT_ACTIVATION
from virtual_staining.training.helpers import (
    TRAINING_STATE_KEYS,
    LossComponentAccumulator,
    OptimizationRole,
    build_lr_scheduler,
    check_training_state,
    configured_loss_names,
    is_amp_enabled,
    load_training_state,
    loss_validation_metric,
    require_state_keys,
    restoring_validated_state,
    state_error,
    step_lr_schedulers,
    training_state_dict,
    validated_model_state,
)
from virtual_staining.training.preview import ValidationPreview, ValidationPreviewSink
from virtual_staining.training.runtime import MethodMetrics

_POOL_KEYS = frozenset({"fake_A", "fake_B"})
_STATE_KEYS = TRAINING_STATE_KEYS | {"replay_pools"}
_POOL_STATE_KEYS = frozenset({"capacity", "images", "rng_state"})


def init_cyclegan_weights(module: nn.Module) -> None:
    """Apply CycleGAN's N(0, 0.02) conv and N(1, 0.02) affine-norm initialization."""
    for layer in module.modules():
        if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(layer.weight, 0.0, 0.02)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        elif isinstance(layer, (nn.BatchNorm2d, nn.InstanceNorm2d)) and layer.affine:
            nn.init.normal_(layer.weight, 1.0, 0.02)
            nn.init.zeros_(layer.bias)


class ReplayPool:
    """History of generated images shown to a discriminator (Shrivastava et al. 2017)."""

    def __init__(self, capacity: int, seed: int) -> None:
        if capacity < 0:
            raise ValueError("replay pool capacity must be >= 0")
        self.capacity = capacity
        self.images: list[torch.Tensor] = []
        self._rng = torch.Generator().manual_seed(seed)

    def query(self, images: torch.Tensor) -> torch.Tensor:
        images = images.detach()
        if self.capacity == 0:
            return images
        selected: list[torch.Tensor] = []
        for image in images.split(1):
            if len(self.images) < self.capacity:
                self.images.append(image.clone())
                selected.append(image)
            elif torch.rand(1, generator=self._rng).item() < 0.5:
                index = int(torch.randint(self.capacity, (1,), generator=self._rng).item())
                selected.append(self.images[index].clone())
                self.images[index] = image.clone()
            else:
                selected.append(image)
        return torch.cat(selected)

    def state_dict(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "images": [image.clone() for image in self.images],
            "rng_state": self._rng.get_state(),
        }

    def validate_state(self, state: object, context: str) -> None:
        state = require_state_keys(state, _POOL_STATE_KEYS, context)
        if state["capacity"] != self.capacity:
            raise state_error(
                f"{context}.capacity",
                f"is {state['capacity']!r} but method.replay_buffer_size is {self.capacity}",
            )
        images = state["images"]
        if not isinstance(images, list) or len(images) > self.capacity:
            raise state_error(f"{context}.images", f"must be a list of <= {self.capacity} tensors")
        for index, image in enumerate(images):
            if (
                not isinstance(image, torch.Tensor)
                or not image.is_floating_point()
                or image.ndim != 4
                or image.shape[0] != 1
                or image.shape != images[0].shape
            ):
                raise state_error(
                    f"{context}.images[{index}]",
                    "must be a floating 1xCxHxW tensor shaped like the other pool images",
                )
        rng_state = state["rng_state"]
        expected_rng = self._rng.get_state()
        if (
            not isinstance(rng_state, torch.Tensor)
            or rng_state.dtype != torch.uint8
            or rng_state.shape != expected_rng.shape
        ):
            raise state_error(
                f"{context}.rng_state",
                f"must be a uint8 tensor of shape {tuple(expected_rng.shape)}",
            )

    def load_state_dict(self, state: Mapping[str, Any], device: torch.device) -> None:
        self.validate_state(state, "replay pool")
        self.images = [image.detach().to(device, copy=True) for image in state["images"]]
        self._rng.set_state(state["rng_state"].cpu())


def cyclegan_component_metadata(model: ModelConfig) -> dict[str, object]:
    """Describe the reconstruction-relevant CycleGAN components built from ``model``."""
    resolved = model.to_dict()
    generator = {
        "class": ResnetGenerator.__name__,
        "output_activation": GENERATOR_OUTPUT_ACTIVATION,
        "in_channels": 3,
        "out_channels": 3,
        **resolved["generator"],
    }
    discriminator = {
        "class": PatchGANDiscriminator.__name__,
        "architecture": "patchgan",
        "conditional": False,
        "in_channels": 3,
        **resolved["discriminator"],
    }
    return {
        "G_A_to_B": generator,
        "G_B_to_A": dict(generator),
        "D_A": discriminator,
        "D_B": dict(discriminator),
    }


def cyclegan_checkpoint_identity(
    model: ModelConfig,
    image_size: tuple[int, int],
) -> CheckpointIdentity:
    return CheckpointIdentity(
        method=CycleGANMethod.name,
        pairing=CycleGANMethod.pairing,
        inputs=(model.inputs[0],),
        outputs=(model.target,),
        prediction_directions=CycleGANMethod.prediction_directions,
        components=cyclegan_component_metadata(model),
        image_size=image_size,
    )


class CycleGANInferenceAdapter(nn.Module):
    """Expose one tensor-to-tensor CycleGAN generator through named-input inference."""

    def __init__(self, generator: ResnetGenerator, input_name: str) -> None:
        super().__init__()
        self.generator = generator
        self.input_names = (input_name,)

    def forward(self, inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        if tuple(inputs) != self.input_names:
            raise ValueError(f"Expected inputs {self.input_names}, got {tuple(inputs)}")
        return self.generator(inputs[self.input_names[0]])


def load_cyclegan_inference_generator(
    checkpoint_path: Path,
    config: RunConfig,
    direction: InferenceDirection,
    device: torch.device,
) -> CycleGANInferenceAdapter:
    """Validate a v4 CycleGAN checkpoint and restore the generator for ``direction``."""
    checkpoint = validate_checkpoint(
        read_checkpoint(checkpoint_path),
        cyclegan_checkpoint_identity(config.model, config.project.image_size),
        checkpoint_path,
    )
    generator = build_resnet_generator(config.model).to(device)
    generator.load_state_dict(
        validated_model_state(checkpoint.state, f"G_{direction}", generator, checkpoint_path)
    )
    input_name = config.model.inputs[0] if direction == "A_to_B" else config.model.target
    adapter = CycleGANInferenceAdapter(generator, input_name)
    adapter.eval()
    return adapter


@dataclass(frozen=True)
class _Objective:
    total: torch.Tensor
    raw: dict[str, float]
    weighted: dict[str, float]
    current_weight: dict[str, float]


def _objective(terms: list[tuple[str, torch.Tensor, float]]) -> _Objective:
    total = terms[0][1] * terms[0][2]
    for _, raw, weight in terms[1:]:
        total = total + raw * weight
    return _Objective(
        total=total,
        raw={key: float(raw.detach().item()) for key, raw, _ in terms},
        weighted={key: float((raw * weight).detach().item()) for key, raw, weight in terms},
        current_weight={key: weight for key, _, weight in terms},
    )


def _lsgan(prediction: torch.Tensor, real: bool) -> torch.Tensor:
    target = torch.ones_like(prediction) if real else torch.zeros_like(prediction)
    return F.mse_loss(prediction, target)


def _set_requires_grad(modules: Iterable[nn.Module], requires_grad: bool) -> None:
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad_(requires_grad)


class CycleGANMethod:
    """Own the unpaired CycleGAN topology: two ResNet generators and two PatchGANs.

    Domain A is ``model.inputs[0]`` and domain B is ``model.target``. ``D_A`` scores
    domain-A images and ``D_B`` scores domain-B images.
    """

    name: str = "cyclegan"
    pairing: str = "unpaired"
    prediction_directions: tuple[str, ...] = ("A_to_B", "B_to_A")
    default_checkpoint_metric: str = "loss_G_val"
    metric_names: tuple[str, ...] = ("loss_G", "loss_D")
    component_total_names: tuple[str, ...] = ("generator", "discriminator")

    def __init__(self, config: RunConfig, device: torch.device, *, seed: int) -> None:
        if config.training is None:
            raise ValueError("training config is required to construct CycleGAN")
        if config.method.replay_buffer_size is None:
            raise ValueError("method.replay_buffer_size is required to construct CycleGAN")
        self.config = config
        self.training = config.training
        self.device = device
        self._amp_enabled = is_amp_enabled(device)
        self.input_names = (config.model.inputs[0],)
        self.output_names = (config.model.target,)
        self.loss_config = self.training.losses
        self.loss_names = tuple(configured_loss_names(self.loss_config))

        self.G_A_to_B = build_resnet_generator(config.model)
        self.G_B_to_A = build_resnet_generator(config.model)
        self.D_A = build_discriminator(config.model, conditional=False)
        self.D_B = build_discriminator(config.model, conditional=False)
        for model in self._models().values():
            init_cyclegan_weights(model)
            model.to(device)

        betas = (self.training.beta1, self.training.beta2)
        self._opt_G = optim.Adam(
            itertools.chain(self.G_A_to_B.parameters(), self.G_B_to_A.parameters()),
            lr=self.training.lr_g,
            betas=betas,
        )
        self._opt_D = optim.Adam(
            itertools.chain(self.D_A.parameters(), self.D_B.parameters()),
            lr=self.training.lr_d,
            betas=betas,
        )
        self._scaler_G = GradScaler(enabled=self._amp_enabled)
        self._scaler_D = GradScaler(enabled=self._amp_enabled)
        self._scheduler_G = build_lr_scheduler(self.training, self._opt_G)
        self._scheduler_D = build_lr_scheduler(self.training, self._opt_D)

        capacity = config.method.replay_buffer_size
        seed_a, seed_b = (
            int(child.generate_state(1)[0]) for child in np.random.SeedSequence(seed).spawn(2)
        )
        self._pool_A = ReplayPool(capacity, seed_a)
        self._pool_B = ReplayPool(capacity, seed_b)

    def _models(self) -> dict[str, nn.Module]:
        return {
            "G_A_to_B": self.G_A_to_B,
            "G_B_to_A": self.G_B_to_A,
            "D_A": self.D_A,
            "D_B": self.D_B,
        }

    def _pools(self) -> dict[str, ReplayPool]:
        return {"fake_A": self._pool_A, "fake_B": self._pool_B}

    def _roles(self) -> dict[str, OptimizationRole]:
        return {
            "generators": OptimizationRole(self._opt_G, self._scaler_G, self._scheduler_G),
            "discriminators": OptimizationRole(self._opt_D, self._scaler_D, self._scheduler_D),
        }

    def train_mode(self) -> None:
        for model in self._models().values():
            model.train()

    def batch_size(self, batch: object) -> int:
        if not isinstance(batch, dict):
            return 0
        domain_a = batch.get("domain_a")
        if not isinstance(domain_a, torch.Tensor) or domain_a.ndim == 0:
            return 0
        return int(domain_a.shape[0])

    def _unpack_batch(self, batch: object) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(batch, dict):
            raise TypeError("CycleGAN batches must be mappings with domain_a and domain_b")
        tensors: list[torch.Tensor] = []
        for key in ("domain_a", "domain_b"):
            value = batch.get(key)
            if not isinstance(value, torch.Tensor) or value.ndim != 4 or value.shape[1] != 3:
                raise TypeError(f"CycleGAN batch {key!r} must be an RGB NCHW tensor")
            tensors.append(value.to(self.device))
        return tensors[0], tensors[1]

    def _translate(
        self, real_a: torch.Tensor, real_b: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``fake_a, fake_b, rec_a, rec_b`` for one batch."""
        fake_b = self.G_A_to_B(real_a)
        rec_a = self.G_B_to_A(fake_b)
        fake_a = self.G_B_to_A(real_b)
        rec_b = self.G_A_to_B(fake_a)
        return fake_a, fake_b, rec_a, rec_b

    def _generator_objective(
        self,
        real_a: torch.Tensor,
        real_b: torch.Tensor,
        fake_a: torch.Tensor,
        fake_b: torch.Tensor,
        rec_a: torch.Tensor,
        rec_b: torch.Tensor,
        *,
        epoch: int,
        global_step: int | None,
    ) -> _Objective:
        terms: list[tuple[str, torch.Tensor, float]] = []
        for term in self.loss_config.active_generator:
            weight = term.current_weight(epoch=epoch, global_step=global_step)
            if term.name == "adversarial_lsgan":
                raw = _lsgan(self.D_B(fake_b), True) + _lsgan(self.D_A(fake_a), True)
            elif term.name == "cycle_l1":
                raw = F.l1_loss(rec_a, real_a) + F.l1_loss(rec_b, real_b)
            elif term.name == "identity_l1":
                raw = F.l1_loss(self.G_B_to_A(real_a), real_a) + F.l1_loss(
                    self.G_A_to_B(real_b), real_b
                )
            else:
                raise AssertionError(f"Unsupported CycleGAN generator loss {term.name!r}")
            terms.append((f"generator_{term.name}", raw, weight))
        return _objective(terms)

    def _discriminator_objective(
        self,
        real_a: torch.Tensor,
        real_b: torch.Tensor,
        fake_a: torch.Tensor,
        fake_b: torch.Tensor,
        *,
        epoch: int,
        global_step: int | None,
    ) -> _Objective:
        terms: list[tuple[str, torch.Tensor, float]] = []
        for term in self.loss_config.active_discriminator:
            if term.name != "adversarial_lsgan":
                raise AssertionError(f"Unsupported CycleGAN discriminator loss {term.name!r}")
            loss_d_a = 0.5 * (_lsgan(self.D_A(real_a), True) + _lsgan(self.D_A(fake_a), False))
            loss_d_b = 0.5 * (_lsgan(self.D_B(real_b), True) + _lsgan(self.D_B(fake_b), False))
            weight = term.current_weight(epoch=epoch, global_step=global_step)
            terms.append((f"discriminator_{term.name}", loss_d_a + loss_d_b, weight))
        return _objective(terms)

    def _generator_phase(
        self,
        real_a: torch.Tensor,
        real_b: torch.Tensor,
        *,
        epoch: int,
        global_step: int | None,
    ) -> tuple[_Objective, torch.Tensor, torch.Tensor]:
        """Update both generators with frozen discriminators; return the fakes produced."""
        _set_requires_grad((self.D_A, self.D_B), False)
        try:
            self._opt_G.zero_grad()
            with autocast(device_type=self.device.type, enabled=self._amp_enabled):
                fake_a, fake_b, rec_a, rec_b = self._translate(real_a, real_b)
                objective = self._generator_objective(
                    real_a,
                    real_b,
                    fake_a,
                    fake_b,
                    rec_a,
                    rec_b,
                    epoch=epoch,
                    global_step=global_step,
                )
            self._scaler_G.scale(objective.total).backward()
            self._scaler_G.step(self._opt_G)
            self._scaler_G.update()
        finally:
            _set_requires_grad((self.D_A, self.D_B), True)
        return objective, fake_a, fake_b

    def _discriminator_phase(
        self,
        real_a: torch.Tensor,
        real_b: torch.Tensor,
        fake_a: torch.Tensor,
        fake_b: torch.Tensor,
        *,
        epoch: int,
        global_step: int | None,
    ) -> _Objective:
        """Update both discriminators against replayed, detached fakes."""
        self._opt_D.zero_grad()
        pooled_a = self._pool_A.query(fake_a.detach())
        pooled_b = self._pool_B.query(fake_b.detach())
        with autocast(device_type=self.device.type, enabled=self._amp_enabled):
            objective = self._discriminator_objective(
                real_a, real_b, pooled_a, pooled_b, epoch=epoch, global_step=global_step
            )
        self._scaler_D.scale(objective.total).backward()
        self._scaler_D.step(self._opt_D)
        self._scaler_D.update()
        return objective

    def step(self, batch: object, *, epoch: int, global_step: int) -> MethodMetrics:
        real_a, real_b = self._unpack_batch(batch)
        generator, fake_a, fake_b = self._generator_phase(
            real_a, real_b, epoch=epoch, global_step=global_step
        )
        discriminator = self._discriminator_phase(
            real_a, real_b, fake_a, fake_b, epoch=epoch, global_step=global_step
        )
        loss_G = float(generator.total.detach().item())
        loss_D = float(discriminator.total.detach().item())
        return MethodMetrics(
            losses={"loss_G": loss_G, "loss_D": loss_D},
            component_totals={"generator": loss_G, "discriminator": loss_D},
            raw={**generator.raw, **discriminator.raw},
            weighted={**generator.weighted, **discriminator.weighted},
            current_weight={**generator.current_weight, **discriminator.current_weight},
        )

    def validate(
        self,
        loader: torch.utils.data.DataLoader,
        *,
        epoch: int,
        preview_sink: ValidationPreviewSink | None = None,
    ) -> MethodMetrics:
        """Report deterministic training-objective diagnostics; no paired fidelity metrics."""
        was_training = {name: model.training for name, model in self._models().items()}
        for model in self._models().values():
            model.eval()
        try:
            components = LossComponentAccumulator(list(self.loss_names))
            total_G = total_D = 0.0
            count = 0
            with torch.no_grad():
                for batch_index, batch in enumerate(loader):
                    real_a, real_b = self._unpack_batch(batch)
                    with autocast(device_type=self.device.type, enabled=self._amp_enabled):
                        fake_a, fake_b, rec_a, rec_b = self._translate(real_a, real_b)
                        generator = self._generator_objective(
                            real_a,
                            real_b,
                            fake_a,
                            fake_b,
                            rec_a,
                            rec_b,
                            epoch=epoch,
                            global_step=None,
                        )
                        discriminator = self._discriminator_objective(
                            real_a, real_b, fake_a, fake_b, epoch=epoch, global_step=None
                        )
                    for objective in (generator, discriminator):
                        components.add(
                            raw=objective.raw,
                            weighted=objective.weighted,
                            current_weight=objective.current_weight,
                        )
                    total_G += float(generator.total.item())
                    total_D += float(discriminator.total.item())
                    count += 1
                    if preview_sink is not None and preview_sink.wants(epoch, batch_index):
                        preview_sink.write(
                            ValidationPreview(
                                epoch=epoch,
                                batch_index=batch_index,
                                images={
                                    "real_A": real_a.detach(),
                                    "fake_B": fake_b.detach(),
                                    "real_B": real_b.detach(),
                                    "fake_A": fake_a.detach(),
                                },
                            )
                        )
        finally:
            for name, model in self._models().items():
                model.train(was_training[name])
        averages = components.average(count)
        loss_G = total_G / count if count else math.nan
        loss_D = total_D / count if count else math.nan
        return MethodMetrics(
            losses={"loss_G": loss_G, "loss_D": loss_D},
            component_totals={"generator": loss_G, "discriminator": loss_D},
            raw=averages.raw,
            weighted=averages.weighted,
            current_weight=averages.current_weight,
        )

    def validation_metric(self, metrics: MethodMetrics, name: str) -> float | None:
        return loss_validation_metric(metrics, name)

    def checkpoint_selection_metrics(self, metrics: MethodMetrics) -> dict[str, float]:
        value = metrics.losses["loss_G"]
        return {"loss_G_val": value} if math.isfinite(value) else {}

    def checkpoint_selection_modes(self) -> dict[str, str]:
        return {"loss_G_val": "min"}

    def step_schedulers(
        self,
        *,
        epoch: int,
        validation_metrics: MethodMetrics | None,
    ) -> bool:
        return step_lr_schedulers(
            self.training.scheduler,
            (self._scheduler_G, self._scheduler_D),
            epoch=epoch,
            monitor_value=(
                None
                if validation_metrics is None
                else lambda: self.validation_metric(
                    validation_metrics, self.training.scheduler.monitor
                )
            ),
        )

    def learning_rates(self) -> Mapping[str, float]:
        return {
            "lr_g": float(self._opt_G.param_groups[0]["lr"]),
            "lr_d": float(self._opt_D.param_groups[0]["lr"]),
        }

    def component_metadata(self) -> Mapping[str, object]:
        return cyclegan_component_metadata(self.config.model)

    def state_dict(self) -> dict[str, Any]:
        return {
            **training_state_dict(self.training, self._models(), self._roles()),
            "replay_pools": {name: pool.state_dict() for name, pool in self._pools().items()},
        }

    def _validate_state(self, state: Mapping[str, Any]) -> None:
        require_state_keys(state, _STATE_KEYS, "state")
        check_training_state(state, self.training, self._models(), self._roles())
        pools = require_state_keys(state["replay_pools"], _POOL_KEYS, "state.replay_pools")
        for name, pool in self._pools().items():
            pool.validate_state(pools[name], f"state.replay_pools.{name}")

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Preflight every state group, then restore; nothing is mutated on rejection."""
        self._validate_state(state)
        with restoring_validated_state(self.name):
            load_training_state(state, self._models(), self._roles())
            for name, pool in self._pools().items():
                pool.load_state_dict(state["replay_pools"][name], self.device)
