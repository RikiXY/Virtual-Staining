from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler

logger = logging.getLogger(__name__)

CHECKPOINT_FORMAT_VERSION: int = 2
GENERATOR_OUTPUT_ACTIVATION = "tanh"
NORMALIZATION_CONTRACT = {
    "input_range": "[-1, 1]",
    "output_range": "[-1, 1]",
}


@dataclass
class CheckpointState:
    epoch: int
    generator_state_dict: dict[str, Any]
    discriminator_state_dict: dict[str, Any]
    opt_G_state_dict: dict[str, Any]
    opt_D_state_dict: dict[str, Any]
    scaler_G_state_dict: dict[str, Any]
    scaler_D_state_dict: dict[str, Any]


@dataclass(frozen=True)
class BestCheckpointRecord:
    policy: str
    metric: str
    epoch: int
    checkpoint_path: Path
    metric_value: float
    mode: str | None = None


def _make_arch_metadata(
    generator: nn.Module,
    discriminator: nn.Module,
    *,
    model_name: str | None = None,
) -> dict[str, Any]:
    """Build the architecture dict saved inside every checkpoint."""
    return {
        "name": model_name,
        "generator": {
            "class": type(generator).__name__,
            "in_channels": getattr(generator, "in_channels", None),
            "out_channels": getattr(generator, "out_channels", None),
            "base_channels": getattr(generator, "base_channels", None),
            "norm": getattr(generator, "norm", None),
            "dropout": getattr(generator, "dropout", None),
            "bilinear": getattr(generator, "bilinear", None),
            "output_activation": GENERATOR_OUTPUT_ACTIVATION,
        },
        "discriminator": {
            "class": type(discriminator).__name__,
            "in_channels": getattr(discriminator, "in_channels", None),
            "ndf": getattr(discriminator, "ndf", None),
            "norm": getattr(discriminator, "norm", None),
            "use_sigmoid": getattr(discriminator, "use_sigmoid", None),
        },
    }


def _validate_checkpoint_metadata(checkpoint: dict[str, Any], path: Path) -> dict[str, Any]:
    """Return architecture metadata after validating checkpoint format invariants."""
    ckpt_version = checkpoint.get("format_version")
    if ckpt_version != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"Checkpoint format version {ckpt_version!r} does not match current version "
            f"{CHECKPOINT_FORMAT_VERSION}. Re-train from scratch with the current code."
        )

    arch = checkpoint.get("architecture")
    if arch is None:
        raise ValueError(
            f"Checkpoint '{path}' has no architecture metadata. "
            "Only checkpoints saved with the current version are supported."
        )
    if not isinstance(arch, dict):
        raise ValueError("Checkpoint architecture metadata must be a mapping.")

    generator_arch = arch.get("generator", {})
    if not isinstance(generator_arch, dict):
        raise ValueError("Checkpoint generator architecture metadata must be a mapping.")

    ckpt_activation = generator_arch.get("output_activation")
    if ckpt_activation != GENERATOR_OUTPUT_ACTIVATION:
        raise ValueError(
            f"Checkpoint has output_activation={ckpt_activation!r}; current code requires "
            f"{GENERATOR_OUTPUT_ACTIVATION!r}."
        )

    normalization_contract = checkpoint.get("normalization_contract")
    if normalization_contract != NORMALIZATION_CONTRACT:
        raise ValueError(
            "Checkpoint normalization_contract "
            f"{normalization_contract!r} does not match current code."
        )

    return arch


def _check_arch_match(
    checkpoint_arch: dict[str, Any],
    generator: nn.Module,
    discriminator: nn.Module,
) -> None:
    """Raise ValueError if checkpoint architecture does not match the current models."""
    gen_arch = checkpoint_arch.get("generator", {})
    for key in ("in_channels", "out_channels", "base_channels", "norm", "dropout", "bilinear"):
        ckpt_val = gen_arch.get(key)
        curr_val = getattr(generator, key, None)
        if ckpt_val != curr_val:
            raise ValueError(
                f"Architecture mismatch for generator.{key}: "
                f"checkpoint has {ckpt_val!r}, current model has {curr_val!r}. "
                "Instantiate the model with the same parameters used during training."
            )

    disc_arch = checkpoint_arch.get("discriminator", {})
    for key in ("in_channels", "ndf", "norm", "use_sigmoid"):
        ckpt_val = disc_arch.get(key)
        curr_val = getattr(discriminator, key, None)
        if ckpt_val != curr_val:
            raise ValueError(
                f"Architecture mismatch for discriminator.{key}: "
                f"checkpoint has {ckpt_val!r}, current model has {curr_val!r}. "
                "Instantiate the model with the same parameters used during training."
            )


def _check_generator_arch(checkpoint_arch: dict[str, Any], generator: nn.Module) -> None:
    """Raise ValueError if the checkpoint's generator architecture does not match."""
    gen_arch = checkpoint_arch.get("generator", {})
    ckpt_activation = gen_arch.get("output_activation")
    if ckpt_activation != GENERATOR_OUTPUT_ACTIVATION:
        raise ValueError(
            f"Checkpoint has output_activation={ckpt_activation!r}; current code requires "
            f"{GENERATOR_OUTPUT_ACTIVATION!r}."
        )
    for key in ("in_channels", "out_channels", "base_channels", "norm", "dropout", "bilinear"):
        ckpt_val = gen_arch.get(key)
        curr_val = getattr(generator, key, None)
        if ckpt_val != curr_val:
            raise ValueError(
                f"Architecture mismatch for generator.{key}: "
                f"checkpoint has {ckpt_val!r}, inference model has {curr_val!r}. "
                "Instantiate the generator with the same parameters used during training."
            )


class CheckpointManager:
    """Manages saving and loading of Pix2Pix training checkpoints."""

    def __init__(
        self,
        checkpoints_dir: Path,
        generator: nn.Module,
        discriminator: nn.Module,
        opt_G: optim.Optimizer,
        opt_D: optim.Optimizer,
        scaler_G: GradScaler,
        scaler_D: GradScaler,
        image_size: tuple[int, int],
        device: torch.device,
        *,
        model_name: str | None = None,
        lr_g: float | None = None,
        lr_d: float | None = None,
        beta1: float | None = None,
        beta2: float | None = None,
        batch_size: int | None = None,
        num_workers: int | None = None,
        dataset_root: str | None = None,
    ) -> None:
        self.checkpoints_dir = checkpoints_dir
        self.generator = generator
        self.discriminator = discriminator
        self.opt_G = opt_G
        self.opt_D = opt_D
        self.scaler_G = scaler_G
        self.scaler_D = scaler_D
        self.image_size = image_size
        self.device = device
        self.model_name = model_name
        self.lr_g = lr_g
        self.lr_d = lr_d
        self.beta1 = beta1
        self.beta2 = beta2
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_root = dataset_root

    @property
    def best_record_path(self) -> Path:
        return self.checkpoints_dir / "best.json"

    def save(self, epoch: int) -> Path:
        """Save a full training checkpoint. Returns the path."""
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        path = self.checkpoints_dir / f"ep{epoch:03d}.pth"
        checkpoint = {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "epoch": epoch,
            "architecture": _make_arch_metadata(
                self.generator,
                self.discriminator,
                model_name=self.model_name,
            ),
            "normalization_contract": NORMALIZATION_CONTRACT,
            "generator_state_dict": self.generator.state_dict(),
            "discriminator_state_dict": self.discriminator.state_dict(),
            "optimizerG_state_dict": self.opt_G.state_dict(),
            "optimizerD_state_dict": self.opt_D.state_dict(),
            "scalerG_state_dict": self.scaler_G.state_dict(),
            "scalerD_state_dict": self.scaler_D.state_dict(),
            "lr_g": self.lr_g,
            "lr_d": self.lr_d,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "image_size": self.image_size,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "dataset_root": self.dataset_root,
        }
        torch.save(checkpoint, path)
        logger.info("Checkpoint saved: %s", path)
        return path

    def update_selection_records(
        self,
        *,
        metrics: dict[str, float],
        modes: dict[str, str],
        top_k: int,
        epoch: int,
        checkpoint_path: Path,
        config_hash: str | None = None,
        loss_config: dict[str, Any] | None = None,
    ) -> Path:
        """Update checkpoints/best.json with per-metric best and top-k selections."""
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")
        if not metrics:
            raise ValueError("metrics must contain at least one finite metric value")
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Cannot rank missing checkpoint file: {checkpoint_path}")

        payload = self._load_selection_payload()
        payload["schema_version"] = 1
        payload["metrics"] = _selection_metrics(payload)

        for metric, metric_value in sorted(metrics.items()):
            mode = modes[metric]
            if mode not in {"min", "max"}:
                raise ValueError("mode must be one of ['max', 'min']")
            metric_payload = payload["metrics"].get(metric)
            if not isinstance(metric_payload, dict) or metric_payload.get("mode") != mode:
                metric_payload = {
                    "mode": mode,
                    "top_k": top_k,
                    "records": [],
                }

            records = [
                record
                for record in _selection_records(metric_payload, self.best_record_path)
                if record["epoch"] != epoch
                and self._checkpoint_record_path(str(record["checkpoint_path"])).exists()
            ]
            record: dict[str, Any] = {
                "epoch": epoch,
                "checkpoint_path": checkpoint_path.name,
                "metric_value": metric_value,
            }
            if config_hash is not None:
                record["config_hash"] = config_hash
            if loss_config is not None:
                record["loss_config"] = loss_config
            records.append(record)

            ranked_records = _rank_top_k_records(records, mode=mode, top_k=top_k)
            metric_payload["top_k"] = top_k
            metric_payload["records"] = ranked_records
            metric_payload["best"] = ranked_records[0] if ranked_records else None
            payload["metrics"][metric] = metric_payload

        self.best_record_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("Checkpoint selection record saved: %s", self.best_record_path)
        return self.best_record_path

    def _load_selection_payload(self) -> dict[str, Any]:
        if not self.best_record_path.exists():
            return {"schema_version": 1, "metrics": {}}
        try:
            payload = json.loads(self.best_record_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Best checkpoint metadata at {self.best_record_path} is not valid JSON."
            ) from exc
        if not isinstance(payload, dict):
            raise ValueError(
                f"Best checkpoint metadata at {self.best_record_path} must be a JSON object."
            )
        if "metrics" in payload:
            return payload
        raise ValueError(
            f"Best checkpoint metadata at {self.best_record_path} is not a selection record."
        )

    def _checkpoint_record_path(self, checkpoint_value: str) -> Path:
        checkpoint_path = Path(checkpoint_value)
        if not checkpoint_path.is_absolute():
            checkpoint_path = self.checkpoints_dir / checkpoint_path
        return checkpoint_path

    def load(self, path: Path) -> int:
        """Load a training checkpoint. Returns the start_epoch for resuming."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        stored_image_size = checkpoint.get("image_size")
        if stored_image_size is not None and tuple(stored_image_size) != tuple(self.image_size):
            raise ValueError(
                "Image size mismatch between checkpoint and resumed training. "
                f"Checkpoint image_size={tuple(stored_image_size)}, "
                f"current image_size={tuple(self.image_size)}."
            )

        arch = _validate_checkpoint_metadata(checkpoint, path)
        _check_arch_match(arch, self.generator, self.discriminator)

        self.generator.load_state_dict(checkpoint["generator_state_dict"])
        self.discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
        self.opt_G.load_state_dict(checkpoint["optimizerG_state_dict"])
        self.opt_D.load_state_dict(checkpoint["optimizerD_state_dict"])
        self.scaler_G.load_state_dict(checkpoint["scalerG_state_dict"])
        self.scaler_D.load_state_dict(checkpoint["scalerD_state_dict"])

        start_epoch: int = checkpoint["epoch"] + 1
        logger.info("Checkpoint loaded from %s, resuming at epoch %s", path, start_epoch)
        return start_epoch

    def latest(self) -> Path | None:
        """Return the path to the most recent ep*.pth file, or None."""
        candidates = sorted(self.checkpoints_dir.glob("ep*.pth"))
        return candidates[-1] if candidates else None


def load_best_checkpoint_record(
    checkpoints_dir: Path,
    *,
    policy: str,
    metric: str | None = None,
    rank: int = 1,
) -> BestCheckpointRecord:
    """Load and validate checkpoints/best.json for a checkpoint policy."""
    best_path = checkpoints_dir / "best.json"
    if not best_path.exists():
        raise FileNotFoundError(
            f"checkpoint_policy={policy!r} requires {best_path}, but that file does not exist."
        )

    try:
        payload = json.loads(best_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Best checkpoint metadata at {best_path} is not valid JSON.") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"Best checkpoint metadata at {best_path} must be a JSON object.")

    return _load_selection_checkpoint_record(
        payload,
        checkpoints_dir=checkpoints_dir,
        best_path=best_path,
        policy=policy,
        metric=metric,
        rank=rank,
    )


def resolve_best_checkpoint_path(
    checkpoints_dir: Path,
    *,
    policy: str,
    metric: str | None = None,
    rank: int = 1,
) -> Path:
    """Resolve a best-checkpoint policy via checkpoints/best.json."""
    return load_best_checkpoint_record(
        checkpoints_dir,
        policy=policy,
        metric=metric,
        rank=rank,
    ).checkpoint_path


def _load_selection_checkpoint_record(
    payload: dict[str, Any],
    *,
    checkpoints_dir: Path,
    best_path: Path,
    policy: str,
    metric: str | None,
    rank: int,
) -> BestCheckpointRecord:
    if rank <= 0:
        raise ValueError("checkpoint_rank must be greater than 0")
    metrics = _selection_metrics(payload)
    selected_metric = _select_metric_from_policy(payload, policy=policy, metric=metric)
    metric_payload = metrics.get(selected_metric)
    if not isinstance(metric_payload, dict):
        raise ValueError(
            f"Best checkpoint metadata at {best_path} has no records for metric "
            f"{selected_metric!r}."
        )
    records = _selection_records(metric_payload, best_path)
    matching = [record for record in records if record.get("rank") == rank]
    if not matching:
        raise ValueError(
            f"Best checkpoint metadata at {best_path} has no rank {rank} for metric "
            f"{selected_metric!r}."
        )
    record = matching[0]
    checkpoint_path = Path(str(record["checkpoint_path"]))
    if not checkpoint_path.is_absolute():
        checkpoint_path = checkpoints_dir / checkpoint_path
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Best checkpoint metadata at {best_path} points to missing file {checkpoint_path}."
        )
    return BestCheckpointRecord(
        policy=policy,
        metric=selected_metric,
        epoch=int(record["epoch"]),
        checkpoint_path=checkpoint_path,
        metric_value=float(record["metric_value"]),
        mode=str(metric_payload.get("mode")) if metric_payload.get("mode") is not None else None,
    )


def _select_metric_from_policy(
    payload: dict[str, Any],
    *,
    policy: str,
    metric: str | None,
) -> str:
    if metric is not None:
        return metric
    raise ValueError(f"checkpoint_policy={policy!r} requires checkpoint_metric.")


def _selection_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    metrics = payload.get("metrics", {})
    if not isinstance(metrics, dict):
        raise ValueError("Best checkpoint metadata has invalid metrics.")
    return metrics


def _selection_records(metric_payload: dict[str, Any], path: Path) -> list[dict[str, Any]]:
    records = metric_payload.get("records", [])
    if not isinstance(records, list):
        raise ValueError(f"Best checkpoint metadata at {path} has invalid records.")
    return [_normalize_top_k_record(record, path) for record in records]


def _normalize_top_k_record(record: Any, top_k_path: Path) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ValueError(f"Top-k checkpoint metadata at {top_k_path} contains a non-object record.")

    epoch = record.get("epoch")
    if not isinstance(epoch, int):
        raise ValueError(f"Top-k checkpoint metadata at {top_k_path} has invalid record epoch.")

    checkpoint_value = record.get("checkpoint_path")
    if not isinstance(checkpoint_value, str) or not checkpoint_value.strip():
        raise ValueError(
            f"Top-k checkpoint metadata at {top_k_path} has invalid record checkpoint_path."
        )

    metric_value = record.get("metric_value")
    if not isinstance(metric_value, int | float):
        raise ValueError(
            f"Top-k checkpoint metadata at {top_k_path} has invalid record metric_value."
        )

    normalized = {
        "epoch": epoch,
        "checkpoint_path": checkpoint_value,
        "metric_value": float(metric_value),
    }
    if "rank" in record:
        rank = record["rank"]
        if not isinstance(rank, int):
            raise ValueError(f"Top-k checkpoint metadata at {top_k_path} has invalid record rank.")
        normalized["rank"] = rank
    if "config_hash" in record:
        config_hash = record["config_hash"]
        if not isinstance(config_hash, str):
            raise ValueError(
                f"Top-k checkpoint metadata at {top_k_path} has invalid record config_hash."
            )
        normalized["config_hash"] = config_hash
    if "loss_config" in record:
        loss_config = record["loss_config"]
        if not isinstance(loss_config, dict):
            raise ValueError(
                f"Top-k checkpoint metadata at {top_k_path} has invalid record loss_config."
            )
        normalized["loss_config"] = loss_config
    return normalized


def _rank_top_k_records(
    records: list[dict[str, Any]],
    *,
    mode: str,
    top_k: int,
) -> list[dict[str, Any]]:
    reverse = mode == "max"
    sorted_records = sorted(
        records,
        key=lambda record: (
            (float(record["metric_value"]), -int(record["epoch"]))
            if reverse
            else (float(record["metric_value"]), int(record["epoch"]))
        ),
        reverse=reverse,
    )
    ranked_records: list[dict[str, Any]] = []
    for rank, record in enumerate(sorted_records[:top_k], start=1):
        ranked_record = dict(record)
        ranked_record["rank"] = rank
        ranked_records.append(ranked_record)
    return ranked_records
