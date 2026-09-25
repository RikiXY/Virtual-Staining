from __future__ import annotations

import logging
from pathlib import Path

import torch

from virtual_staining.checkpoint_contract import (
    CheckpointIdentity,
    build_checkpoint_payload,
    read_checkpoint,
    validate_checkpoint,
)
from virtual_staining.checkpoint_selection import latest_checkpoint_path
from virtual_staining.training.runtime import TrainingMethodRuntime

logger = logging.getLogger(__name__)


class MethodCheckpointManager:
    """Persist and restore method-owned training state in the v4 checkpoint format."""

    def __init__(
        self,
        method: TrainingMethodRuntime,
        checkpoints_dir: Path,
        *,
        image_size: tuple[int, int],
        device: torch.device,
        config_hash: str | None = None,
    ) -> None:
        self.method = method
        self.checkpoints_dir = checkpoints_dir
        self.image_size = image_size
        self.device = device
        self.config_hash = config_hash

    def identity(self) -> CheckpointIdentity:
        method = self.method
        return CheckpointIdentity(
            method=method.name,
            pairing=method.pairing,
            inputs=tuple(method.input_names),
            outputs=tuple(method.output_names),
            prediction_directions=tuple(method.prediction_directions),
            components=method.component_metadata(),
            image_size=self.image_size,
        )

    def save(self, epoch: int) -> Path:
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        path = self.checkpoints_dir / f"ep{epoch:03d}.pth"
        payload = build_checkpoint_payload(
            self.identity(),
            epoch=epoch,
            state=self.method.state_dict(),
            config_hash=self.config_hash,
        )
        torch.save(payload, path)
        logger.info("Checkpoint saved: %s", path)
        return path

    def load(self, path: Path) -> int:
        checkpoint = validate_checkpoint(
            read_checkpoint(path, self.device),
            self.identity(),
            path,
        )
        self.method.load_state_dict(checkpoint.state)
        start_epoch = checkpoint.epoch + 1
        logger.info("Checkpoint loaded from %s, resuming at epoch %s", path, start_epoch)
        return start_epoch

    def latest(self) -> Path | None:
        return latest_checkpoint_path(self.checkpoints_dir)
