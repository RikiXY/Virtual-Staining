from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, get_args

from virtual_staining.metrics import VALIDATION_IMAGE_METRIC_NAMES, is_higher_better_metric

logger = logging.getLogger(__name__)

CheckpointMetric = str
CheckpointMode = Literal["min", "max"]
CHECKPOINT_SELECTION_SCHEMA_VERSION = 1
SUPPORTED_CHECKPOINT_METRICS = frozenset(("loss_G_val", *VALIDATION_IMAGE_METRIC_NAMES))
RANKED_CHECKPOINT_POLICIES = frozenset({"best", "top_k"})
SUPPORTED_CHECKPOINT_POLICIES = frozenset({"latest"}) | RANKED_CHECKPOINT_POLICIES
_CHECKPOINT_NAME_PATTERN = re.compile(r"^ep(?P<epoch>\d+)\.pth$")


def default_checkpoint_mode(metric: str) -> CheckpointMode:
    if metric not in SUPPORTED_CHECKPOINT_METRICS:
        raise ValueError(
            f"Unsupported checkpoint_metric {metric!r}. "
            f"Supported metrics: {sorted(SUPPORTED_CHECKPOINT_METRICS)}."
        )
    if metric == "loss_G_val":
        return "min"
    if metric.startswith("val_"):
        return "max" if is_higher_better_metric(metric.removeprefix("val_")) else "min"
    raise AssertionError(f"Unsupported checkpoint_metric slipped through validation: {metric!r}")


@dataclass(frozen=True)
class BestCheckpointRecord:
    policy: str
    metric: str
    epoch: int
    checkpoint_path: Path
    metric_value: float
    mode: str | None = None


def update_checkpoint_selection(
    checkpoints_dir: Path,
    *,
    metrics: dict[str, float],
    modes: dict[str, str],
    top_k: int,
    epoch: int,
    checkpoint_path: Path,
    config_hash: str | None = None,
    loss_config: dict[str, Any] | None = None,
) -> Path:
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")
    if epoch < 0:
        raise ValueError("epoch must be greater than or equal to 0")
    if not metrics:
        raise ValueError("metrics must contain at least one finite metric value")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Cannot rank missing checkpoint file: {checkpoint_path}")

    for metric, metric_value in metrics.items():
        if isinstance(metric_value, bool) or not isinstance(metric_value, int | float):
            raise ValueError(f"Checkpoint metric {metric!r} must be numeric")
        if not math.isfinite(float(metric_value)):
            raise ValueError(f"Checkpoint metric {metric!r} must be finite")
        if metric not in modes:
            raise ValueError(f"Missing checkpoint mode for metric {metric!r}")
        if modes[metric] not in set(get_args(CheckpointMode)):
            raise ValueError("mode must be one of ['max', 'min']")

    best_path = checkpoints_dir / "best.json"
    payload = _load_selection_payload(best_path)
    payload["schema_version"] = CHECKPOINT_SELECTION_SCHEMA_VERSION
    payload["metrics"] = _selection_metrics(payload, best_path)

    for metric, metric_value in sorted(metrics.items()):
        mode = modes[metric]
        metric_payload = payload["metrics"].get(metric)
        if not isinstance(metric_payload, dict) or metric_payload.get("mode") != mode:
            metric_payload = {"mode": mode, "top_k": top_k, "records": []}

        records = [
            record
            for record in _selection_records(metric_payload, best_path)
            if record["epoch"] != epoch
            and _checkpoint_record_path(checkpoints_dir, str(record["checkpoint_path"])).is_file()
        ]
        record: dict[str, Any] = {
            "epoch": epoch,
            "checkpoint_path": checkpoint_path.name,
            "metric_value": float(metric_value),
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

    best_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Checkpoint selection record saved: %s", best_path)
    return best_path


def load_best_checkpoint_record(
    checkpoints_dir: Path,
    *,
    policy: str,
    metric: str | None = None,
    rank: int = 1,
) -> BestCheckpointRecord:
    best_path = checkpoints_dir / "best.json"
    if not best_path.exists():
        raise FileNotFoundError(
            f"checkpoint_policy={policy!r} requires {best_path}, but that file does not exist."
        )
    if rank <= 0:
        raise ValueError("checkpoint_rank must be greater than 0")

    payload = _load_selection_payload(best_path)
    metrics = _selection_metrics(payload, best_path)
    selected_metric = _select_metric_from_policy(policy=policy, metric=metric)
    metric_payload = metrics.get(selected_metric)
    if not isinstance(metric_payload, dict):
        raise ValueError(
            f"Best checkpoint metadata at {best_path} has no records for metric "
            f"{selected_metric!r}."
        )
    mode = _selection_mode(metric_payload, best_path)
    records = _selection_records(metric_payload, best_path)
    matching = [record for record in records if record.get("rank") == rank]
    if not matching:
        raise ValueError(
            f"Best checkpoint metadata at {best_path} has no rank {rank} for metric "
            f"{selected_metric!r}."
        )
    record = matching[0]
    checkpoint_path = _checkpoint_record_path(checkpoints_dir, str(record["checkpoint_path"]))
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Best checkpoint metadata at {best_path} points to missing file {checkpoint_path}."
        )
    return BestCheckpointRecord(
        policy=policy,
        metric=selected_metric,
        epoch=int(record["epoch"]),
        checkpoint_path=checkpoint_path,
        metric_value=float(record["metric_value"]),
        mode=mode,
    )


def _resolve_best_checkpoint_path(
    checkpoints_dir: Path,
    *,
    policy: str,
    metric: str | None = None,
    rank: int = 1,
) -> Path:
    return load_best_checkpoint_record(
        checkpoints_dir,
        policy=policy,
        metric=metric,
        rank=rank,
    ).checkpoint_path


def latest_checkpoint_path(checkpoints_dir: Path) -> Path | None:
    candidates: list[tuple[int, Path]] = []
    for path in checkpoints_dir.glob("ep*.pth"):
        match = _CHECKPOINT_NAME_PATTERN.fullmatch(path.name)
        if match is not None and path.is_file():
            candidates.append((int(match.group("epoch")), path))
    if not candidates:
        return None
    return max(candidates, key=lambda candidate: (candidate[0], candidate[1].name))[1]


def resolve_checkpoint_path(
    checkpoints_dir: Path,
    *,
    policy: str,
    metric: str | None = None,
    rank: int = 1,
) -> Path:
    if policy == "latest":
        path = latest_checkpoint_path(checkpoints_dir)
        if path is None:
            raise FileNotFoundError(
                f"checkpoint_policy='latest' but no checkpoints found in {checkpoints_dir}"
            )
        return path
    if policy in RANKED_CHECKPOINT_POLICIES:
        return _resolve_best_checkpoint_path(
            checkpoints_dir, policy=policy, metric=metric, rank=rank
        )
    raise ValueError(f"Unsupported checkpoint policy: {policy!r}")


def _load_selection_payload(best_path: Path) -> dict[str, Any]:
    if not best_path.exists():
        return _empty_selection_payload()

    payload = _read_payload(best_path)
    schema_version = payload.get("schema_version")
    if schema_version is None:
        raise ValueError(
            f"Checkpoint selection metadata at {best_path} is missing schema_version. "
            f"Legacy/unversioned best.json records are not supported; expected schema version "
            f"{CHECKPOINT_SELECTION_SCHEMA_VERSION}."
        )
    if type(schema_version) is not int:
        raise ValueError(
            f"Checkpoint selection metadata at {best_path} has invalid schema_version."
        )
    if schema_version != CHECKPOINT_SELECTION_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported checkpoint selection schema_version {schema_version} at {best_path}. "
            f"Supported version: {CHECKPOINT_SELECTION_SCHEMA_VERSION}."
        )
    _selection_metrics(payload, best_path)
    return payload


def _empty_selection_payload() -> dict[str, Any]:
    return {"schema_version": CHECKPOINT_SELECTION_SCHEMA_VERSION, "metrics": {}}


def _read_payload(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Best checkpoint metadata at {path} is not valid JSON.") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Best checkpoint metadata at {path} must be a JSON object.")
    return payload


def _checkpoint_record_path(checkpoints_dir: Path, checkpoint_value: str) -> Path:
    checkpoint_path = Path(checkpoint_value)
    return checkpoint_path if checkpoint_path.is_absolute() else checkpoints_dir / checkpoint_path


def _select_metric_from_policy(*, policy: str, metric: str | None) -> str:
    if metric is not None:
        return metric
    raise ValueError(f"checkpoint_policy={policy!r} requires checkpoint_metric.")


def _selection_metrics(payload: dict[str, Any], path: Path) -> dict[str, Any]:
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError(f"Best checkpoint metadata at {path} has invalid metrics.")
    return metrics


def _selection_mode(metric_payload: dict[str, Any], path: Path) -> str:
    mode = metric_payload.get("mode")
    if not isinstance(mode, str) or mode not in set(get_args(CheckpointMode)):
        raise ValueError(f"Best checkpoint metadata at {path} has invalid metric mode.")
    return mode


def _selection_records(metric_payload: dict[str, Any], path: Path) -> list[dict[str, Any]]:
    _selection_mode(metric_payload, path)
    records = metric_payload.get("records", [])
    if not isinstance(records, list):
        raise ValueError(f"Best checkpoint metadata at {path} has invalid records.")
    return [_normalize_top_k_record(record, path) for record in records]


def _normalize_top_k_record(record: Any, top_k_path: Path) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ValueError(f"Top-k checkpoint metadata at {top_k_path} contains a non-object record.")
    epoch = record.get("epoch")
    if type(epoch) is not int or epoch < 0:
        raise ValueError(f"Top-k checkpoint metadata at {top_k_path} has invalid record epoch.")
    checkpoint_value = record.get("checkpoint_path")
    if not isinstance(checkpoint_value, str) or not checkpoint_value.strip():
        raise ValueError(
            f"Top-k checkpoint metadata at {top_k_path} has invalid record checkpoint_path."
        )
    metric_value = record.get("metric_value")
    if (
        isinstance(metric_value, bool)
        or not isinstance(metric_value, int | float)
        or not math.isfinite(float(metric_value))
    ):
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
        if type(rank) is not int or rank <= 0:
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
    metric_direction = -1.0 if mode == "max" else 1.0
    sorted_records = sorted(
        records,
        key=lambda record: (
            metric_direction * float(record["metric_value"]),
            int(record["epoch"]),
            str(record["checkpoint_path"]),
        ),
    )
    ranked_records: list[dict[str, Any]] = []
    for rank, record in enumerate(sorted_records[:top_k], start=1):
        ranked_record = dict(record)
        ranked_record["rank"] = rank
        ranked_records.append(ranked_record)
    return ranked_records
