from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import torch
import torch.optim as optim
from torch.amp import GradScaler

from virtual_staining.config.run import RunConfig
from virtual_staining.models.factory import build_discriminator, build_generator
from virtual_staining.training.helpers import configured_loss_names, is_amp_enabled, unpack_batch
from virtual_staining.training.losses import ConfiguredLossEvaluator, StepLosses
from virtual_staining.training.results import EpochMetrics
from virtual_staining.training.steps import Pix2PixTrainingStep
from virtual_staining.training.validator import validate_epoch


class Pix2PixMethod:
    name = "pix2pix"
    pairing = "paired"

    def __init__(self, config: RunConfig, device: torch.device) -> None:
        if config.training is None:
            raise ValueError("training config is required to construct Pix2Pix")
        self.config = config
        self.device = device
        self.generator = build_generator(config.model).to(device)
        self.discriminator = build_discriminator(config.model, conditional=True).to(device)
        training = config.training
        self.opt_G = optim.Adam(
            self.generator.parameters(), lr=training.lr_g, betas=(training.beta1, training.beta2)
        )
        self.opt_D = optim.Adam(
            self.discriminator.parameters(),
            lr=training.lr_d,
            betas=(training.beta1, training.beta2),
        )
        self.scaler_G = GradScaler(enabled=is_amp_enabled(device))
        self.scaler_D = GradScaler(enabled=is_amp_enabled(device))
        self.optimizers = (self.opt_G, self.opt_D)
        self.loss_names = configured_loss_names(training.losses)
        self._step = Pix2PixTrainingStep(
            self.generator,
            self.discriminator,
            self.opt_G,
            self.opt_D,
            self.scaler_G,
            self.scaler_D,
            device,
            is_amp_enabled(device),
            training.losses.generator,
            training.losses.discriminator,
        )
        self._loss_evaluator = ConfiguredLossEvaluator(
            generator_terms=training.losses.generator,
            discriminator_terms=training.losses.discriminator,
        )

    def train_mode(self) -> None:
        self.generator.train()
        self.discriminator.train()

    def step(self, batch: Any, *, epoch: int, global_step: int) -> StepLosses:
        inputs, target, masks = unpack_batch(
            batch, self.device, cast(tuple[str, ...], self.generator.input_names)
        )
        return self._step.step(inputs, target, epoch=epoch, global_step=global_step, masks=masks)

    def validate(
        self,
        loader: torch.utils.data.DataLoader,
        *,
        epoch: int,
        output_dir: Path,
    ) -> EpochMetrics:
        assert self.config.training is not None
        return validate_epoch(
            epoch=epoch,
            generator=self.generator,
            discriminator=self.discriminator,
            val_loader=loader,
            loss_evaluator=self._loss_evaluator,
            losses=self.config.training.losses,
            device=self.device,
            amp_enabled=is_amp_enabled(self.device),
            output_dir=output_dir,
        )

    def component_metadata(self) -> Mapping[str, object]:
        model_config = self.config.model.to_dict()
        return {
            "generator": {
                "architecture": self.config.model.generator.architecture,
                "class": type(self.generator).__name__,
                "class_path": self.config.model.generator.class_path,
                "resolved": model_config["generator"],
            },
            "discriminator": {
                "architecture": self.config.model.discriminator.architecture,
                "class": type(self.discriminator).__name__,
                "class_path": self.config.model.discriminator.class_path,
                "resolved": model_config["discriminator"],
            },
            "directions": [f"{'+'.join(self.config.model.inputs)}_to_{self.config.model.target}"],
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "models": {
                "generator": self.generator.state_dict(),
                "discriminator": self.discriminator.state_dict(),
            },
            "optimizers": {
                "generator": self.opt_G.state_dict(),
                "discriminator": self.opt_D.state_dict(),
            },
            "scalers": {
                "generator": self.scaler_G.state_dict(),
                "discriminator": self.scaler_D.state_dict(),
            },
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        models = state["models"]
        optimizers = state["optimizers"]
        scalers = state.get("scalers", {})
        self.generator.load_state_dict(models["generator"])
        self.discriminator.load_state_dict(models["discriminator"])
        self.opt_G.load_state_dict(optimizers["generator"])
        self.opt_D.load_state_dict(optimizers["discriminator"])
        if "generator" in scalers:
            self.scaler_G.load_state_dict(scalers["generator"])
        if "discriminator" in scalers:
            self.scaler_D.load_state_dict(scalers["discriminator"])

    def load_legacy_v3(self, checkpoint: Mapping[str, Any]) -> None:
        architecture = checkpoint.get("architecture", {})
        generator_metadata = architecture.get("generator", {})
        if generator_metadata.get("input_names") != list(self.config.model.inputs):
            raise ValueError("Legacy checkpoint input domains do not match the configured model")
        if generator_metadata.get("target_modality") != self.config.model.target:
            raise ValueError("Legacy checkpoint target domain does not match the configured model")
        self.generator.load_state_dict(checkpoint["generator_state_dict"])
        self.discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
        self.opt_G.load_state_dict(checkpoint["optimizerG_state_dict"])
        self.opt_D.load_state_dict(checkpoint["optimizerD_state_dict"])
        self.scaler_G.load_state_dict(checkpoint["scalerG_state_dict"])
        self.scaler_D.load_state_dict(checkpoint["scalerD_state_dict"])
