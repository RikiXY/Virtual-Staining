from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torchvision.utils import save_image

from virtual_staining.config.run import RunConfig
from virtual_staining.models.factory import build_discriminator, build_generator
from virtual_staining.models.io_contract import denormalize_model_output
from virtual_staining.training.helpers import is_amp_enabled
from virtual_staining.training.losses import LeastSquaresAdversarialLoss, StepLosses
from virtual_staining.training.results import EpochMetrics


class CycleGANMethod:
    name = "cyclegan"
    pairing = "unpaired"

    def __init__(self, config: RunConfig, device: torch.device) -> None:
        if config.training is None:
            raise ValueError("training config is required to construct CycleGAN")
        self.config = config
        self.device = device
        self.G_A_to_B = build_generator(config.model).to(device)
        self.G_B_to_A = build_generator(config.model).to(device)
        self.D_A = build_discriminator(config.model, conditional=False).to(device)
        self.D_B = build_discriminator(config.model, conditional=False).to(device)
        training = config.training
        self.opt_G = optim.Adam(
            list(self.G_A_to_B.parameters()) + list(self.G_B_to_A.parameters()),
            lr=training.lr_g,
            betas=(training.beta1, training.beta2),
        )
        self.opt_D = optim.Adam(
            list(self.D_A.parameters()) + list(self.D_B.parameters()),
            lr=training.lr_d,
            betas=(training.beta1, training.beta2),
        )
        self.scaler_G = GradScaler(enabled=is_amp_enabled(device))
        self.scaler_D = GradScaler(enabled=is_amp_enabled(device))
        self.optimizers = (self.opt_G, self.opt_D)
        self.loss_names = [
            "generator_adversarial_lsgan",
            "generator_cycle_l1",
            "generator_identity_l1",
            "discriminator_adversarial_lsgan",
        ]
        self._adversarial = LeastSquaresAdversarialLoss()
        self._l1 = nn.L1Loss()

    def _weight(self, name: str, *, role: str, epoch: int, global_step: int | None = None) -> float:
        assert self.config.training is not None
        terms = (
            self.config.training.losses.generator
            if role == "generator"
            else self.config.training.losses.discriminator
        )
        return next(
            (
                term.current_weight(epoch=epoch, global_step=global_step)
                for term in terms
                if term.name == name
            ),
            0.0,
        )

    def train_mode(self) -> None:
        for model in (self.G_A_to_B, self.G_B_to_A, self.D_A, self.D_B):
            model.train()

    def _batch(self, batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(batch, dict) or "domain_a" not in batch or "domain_b" not in batch:
            raise TypeError("CycleGAN batches must contain domain_a and domain_b")
        real_a, real_b = batch["domain_a"], batch["domain_b"]
        if not isinstance(real_a, torch.Tensor) or not isinstance(real_b, torch.Tensor):
            raise TypeError("CycleGAN domain values must be tensors")
        return real_a.to(self.device), real_b.to(self.device)

    def step(self, batch: Any, *, epoch: int, global_step: int) -> StepLosses:
        real_a, real_b = self._batch(batch)
        amp = is_amp_enabled(self.device)
        adv_weight = self._weight(
            "adversarial_lsgan", role="generator", epoch=epoch, global_step=global_step
        )
        disc_weight = self._weight(
            "adversarial_lsgan", role="discriminator", epoch=epoch, global_step=global_step
        )
        cycle_weight = self._weight(
            "cycle_l1", role="generator", epoch=epoch, global_step=global_step
        )
        identity_weight = self._weight(
            "identity_l1", role="generator", epoch=epoch, global_step=global_step
        )
        self.opt_D.zero_grad()
        with autocast(device_type=self.device.type, enabled=amp):
            fake_b = self.G_A_to_B(real_a).detach()
            fake_a = self.G_B_to_A(real_b).detach()
            pred_real_a = self.D_A(real_a)
            pred_fake_a = self.D_A(fake_a)
            pred_real_b = self.D_B(real_b)
            pred_fake_b = self.D_B(fake_b)
            loss_d_a = self._adversarial.discriminator(pred_real_a, pred_fake_a)
            loss_d_b = self._adversarial.discriminator(pred_real_b, pred_fake_b)
            loss_d_raw = loss_d_a + loss_d_b
            loss_d = disc_weight * loss_d_raw
        self.scaler_D.scale(loss_d).backward()
        self.scaler_D.step(self.opt_D)
        self.scaler_D.update()

        self.opt_G.zero_grad()
        for discriminator in (self.D_A, self.D_B):
            discriminator.requires_grad_(False)
        with autocast(device_type=self.device.type, enabled=amp):
            fake_b = self.G_A_to_B(real_a)
            fake_a = self.G_B_to_A(real_b)
            adv_raw = self._adversarial.generator(self.D_B(fake_b)) + self._adversarial.generator(
                self.D_A(fake_a)
            )
            recovered_a = self.G_B_to_A(fake_b)
            recovered_b = self.G_A_to_B(fake_a)
            cycle_raw = self._l1(recovered_a, real_a) + self._l1(recovered_b, real_b)
            identity_raw = torch.zeros((), device=self.device)
            if identity_weight:
                identity_raw = self._l1(self.G_B_to_A(real_a), real_a) + self._l1(
                    self.G_A_to_B(real_b), real_b
                )
            loss_g = (
                adv_weight * adv_raw + cycle_weight * cycle_raw + identity_weight * identity_raw
            )
        self.scaler_G.scale(loss_g).backward()
        self.scaler_G.step(self.opt_G)
        self.scaler_G.update()
        for discriminator in (self.D_A, self.D_B):
            discriminator.requires_grad_(True)
        raw = {
            "generator_adversarial_lsgan": float(adv_raw.detach()),
            "generator_cycle_l1": float(cycle_raw.detach()),
            "generator_identity_l1": float(identity_raw.detach()),
            "discriminator_adversarial_lsgan": float(loss_d_raw.detach()),
        }
        weights = {
            "generator_adversarial_lsgan": adv_weight,
            "generator_cycle_l1": cycle_weight,
            "generator_identity_l1": identity_weight,
            "discriminator_adversarial_lsgan": disc_weight,
        }
        return StepLosses(
            loss_G=float(loss_g.detach()),
            loss_D=float(loss_d.detach()),
            raw=raw,
            weighted={name: raw[name] * weights[name] for name in raw},
            current_weight=weights,
        )

    def validate(
        self,
        loader: torch.utils.data.DataLoader,
        *,
        epoch: int,
        output_dir: Path,
    ) -> EpochMetrics:
        output_dir.mkdir(parents=True, exist_ok=True)
        models = (self.G_A_to_B, self.G_B_to_A, self.D_A, self.D_B)
        states = tuple(model.training for model in models)
        for model in models:
            model.eval()
        totals_g = totals_d = 0.0
        adv_weight = self._weight("adversarial_lsgan", role="generator", epoch=epoch)
        disc_weight = self._weight("adversarial_lsgan", role="discriminator", epoch=epoch)
        cycle_weight = self._weight("cycle_l1", role="generator", epoch=epoch)
        identity_weight = self._weight("identity_l1", role="generator", epoch=epoch)
        count = 0
        with torch.no_grad():
            for index, batch in enumerate(loader):
                real_a, real_b = self._batch(batch)
                fake_b = self.G_A_to_B(real_a)
                fake_a = self.G_B_to_A(real_b)
                cycle = self._l1(self.G_B_to_A(fake_b), real_a) + self._l1(
                    self.G_A_to_B(fake_a), real_b
                )
                identity = torch.zeros((), device=self.device)
                if identity_weight:
                    identity = self._l1(self.G_B_to_A(real_a), real_a) + self._l1(
                        self.G_A_to_B(real_b), real_b
                    )
                adv = self._adversarial.generator(self.D_B(fake_b)) + self._adversarial.generator(
                    self.D_A(fake_a)
                )
                discriminator = self._adversarial.discriminator(
                    self.D_A(real_a), self.D_A(fake_a)
                ) + self._adversarial.discriminator(self.D_B(real_b), self.D_B(fake_b))
                totals_g += float(
                    adv_weight * adv + cycle_weight * cycle + identity_weight * identity
                )
                totals_d += float(disc_weight * discriminator)
                count += 1
                if index < 5:
                    panel = torch.cat((real_a[:1], fake_b[:1], real_b[:1], fake_a[:1]), dim=0)
                    save_image(
                        denormalize_model_output(panel),
                        output_dir / f"epoch{epoch}_batch{index}_unpaired_grid.png",
                        nrow=4,
                    )
        for model, was_training in zip(models, states, strict=True):
            if was_training:
                model.train()
        divisor = max(1, count)
        return EpochMetrics(loss_G=totals_g / divisor, loss_D=totals_d / divisor)

    def component_metadata(self) -> Mapping[str, object]:
        generator = self.config.model.generator
        discriminator = self.config.model.discriminator
        model_config = self.config.model.to_dict()
        return {
            "generators": {
                "A_to_B": {
                    "architecture": generator.architecture,
                    "class": type(self.G_A_to_B).__name__,
                    "class_path": generator.class_path,
                    "resolved": model_config["generator"],
                },
                "B_to_A": {
                    "architecture": generator.architecture,
                    "class": type(self.G_B_to_A).__name__,
                    "class_path": generator.class_path,
                    "resolved": model_config["generator"],
                },
            },
            "discriminators": {
                "A": {
                    "architecture": discriminator.architecture,
                    "class": type(self.D_A).__name__,
                    "class_path": discriminator.class_path,
                    "resolved": model_config["discriminator"],
                },
                "B": {
                    "architecture": discriminator.architecture,
                    "class": type(self.D_B).__name__,
                    "class_path": discriminator.class_path,
                    "resolved": model_config["discriminator"],
                },
            },
            "directions": ["A_to_B", "B_to_A"],
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "models": {name: model.state_dict() for name, model in self._models().items()},
            "optimizers": {
                "generators": self.opt_G.state_dict(),
                "discriminators": self.opt_D.state_dict(),
            },
            "scalers": {
                "generators": self.scaler_G.state_dict(),
                "discriminators": self.scaler_D.state_dict(),
            },
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        for name, model in self._models().items():
            model.load_state_dict(state["models"][name])
        self.opt_G.load_state_dict(state["optimizers"]["generators"])
        self.opt_D.load_state_dict(state["optimizers"]["discriminators"])
        scalers = state.get("scalers", {})
        if "generators" in scalers:
            self.scaler_G.load_state_dict(scalers["generators"])
        if "discriminators" in scalers:
            self.scaler_D.load_state_dict(scalers["discriminators"])

    def load_legacy_v3(self, checkpoint: Mapping[str, Any]) -> None:
        del checkpoint
        raise ValueError("CycleGAN cannot load a Pix2Pix v3 checkpoint")

    def _models(self) -> dict[str, nn.Module]:
        return {
            "G_A_to_B": self.G_A_to_B,
            "G_B_to_A": self.G_B_to_A,
            "D_A": self.D_A,
            "D_B": self.D_B,
        }
