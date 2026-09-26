from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

import torch

from virtual_staining.checkpoint_contract import (
    CheckpointCompatibilityError,
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
        config_hash: str | None = None,
    ) -> None:
        self.method = method
        self.checkpoints_dir = checkpoints_dir
        self.image_size = image_size
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
        """Write, read back, validate, then atomically publish ``ep<NNN>.pth``.

        The payload goes to a hidden temporary file in the same directory; only a complete
        checkpoint that passes the supported contract is renamed over the final name, so a
        failed or interrupted save never exposes partial bytes to checkpoint selection.
        """
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        path = self.checkpoints_dir / f"ep{epoch:03d}.pth"
        identity = self.identity()
        payload = build_checkpoint_payload(
            identity,
            epoch=epoch,
            state=self.method.state_dict(),
            config_hash=self.config_hash,
        )
        fd, temp_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=self.checkpoints_dir
        )
        temp_path = Path(temp_name)
        try:
            with os.fdopen(fd, "wb") as handle:
                torch.save(payload, handle)
                handle.flush()
                os.fsync(handle.fileno())
            validate_checkpoint(read_checkpoint(temp_path), identity, temp_path)
            os.replace(temp_path, path)
        except BaseException:
            temp_path.unlink(missing_ok=True)
            raise
        directory_fd = os.open(self.checkpoints_dir, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        logger.info("Checkpoint saved: %s", path)
        return path

    def load(self, path: Path) -> int:
        """Validate ``path`` and restore method state; a failed restore voids the runtime."""
        checkpoint = validate_checkpoint(read_checkpoint(path), self.identity(), path)
        try:
            self.method.load_state_dict(checkpoint.state)
        except CheckpointCompatibilityError as exc:
            raise CheckpointCompatibilityError(
                f"Checkpoint '{path}' is incompatible: {exc}"
            ) from exc
        start_epoch = checkpoint.epoch + 1
        logger.info("Checkpoint loaded from %s, resuming at epoch %s", path, start_epoch)
        return start_epoch

    def latest(self) -> Path | None:
        return latest_checkpoint_path(self.checkpoints_dir)
