from __future__ import annotations

import csv
import math
import uuid
from pathlib import Path
from types import TracebackType
from typing import TextIO

from virtual_staining.training.helpers import metrics_fieldnames
from virtual_staining.training.results import EpochMetrics
from virtual_staining.training.validation_metrics import VALIDATION_IMAGE_METRIC_NAMES


class TrainingHistory:
    """Own the canonical one-row-per-epoch CSV history."""

    def __init__(self, path: Path, loss_names: list[str], *, resume_at: int) -> None:
        self._path = path
        self._loss_names = loss_names
        self._resume_at = resume_at
        if resume_at < 0:
            raise ValueError("resume_at must be non-negative")
        self._file: TextIO | None = None
        self._writer: csv.DictWriter[str] | None = None
        self._fieldnames = metrics_fieldnames(loss_names) + VALIDATION_IMAGE_METRIC_NAMES

    def __enter__(self) -> TrainingHistory:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if self._resume_at == 0:
            self._replace_rows([])
        else:
            if not self._path.is_file():
                raise FileNotFoundError(f"Training history not found at {self._path}")
            rows = self._read_rows()
            epochs = [self._parse_epoch(row) for row in rows]
            retained = [
                row for row, epoch in zip(rows, epochs, strict=True) if epoch < self._resume_at
            ]
            retained_epochs = [self._parse_epoch(row) for row in retained]
            if len(set(retained_epochs)) != len(retained_epochs):
                raise ValueError(f"Training history at {self._path} contains duplicate epochs")
            expected = list(range(self._resume_at))
            if sorted(retained_epochs) != expected:
                raise ValueError(
                    f"Training history at {self._path} must contain epochs 0..{self._resume_at - 1}"
                )
            self._replace_rows(retained)

        self._file = self._path.open("a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self._fieldnames)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback
        if self._file is not None:
            self._file.close()
            self._file = None
            self._writer = None

    def write_epoch(
        self,
        epoch: int,
        train_metrics: EpochMetrics,
        val_metrics: EpochMetrics | None,
    ) -> dict[str, float]:
        if self._writer is None or self._file is None:
            raise RuntimeError("TrainingHistory must be entered before writing epochs")

        metrics = _flat_metrics(train_metrics, val_metrics)
        row = {name: _format_metric(metrics.get(name)) for name in self._fieldnames}
        row["epoch"] = str(epoch)
        self._writer.writerow(row)
        self._file.flush()
        return {name: value for name, value in metrics.items() if math.isfinite(value)}

    def _read_rows(self) -> list[dict[str, str]]:
        with self._path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != self._fieldnames:
                raise ValueError(
                    f"Training history header mismatch at {self._path}: "
                    f"expected {self._fieldnames!r}, found {reader.fieldnames!r}"
                )
            rows: list[dict[str, str]] = []
            for row in reader:
                if None in row or any(value is None for value in row.values()):
                    raise ValueError(f"Malformed training history row in {self._path}")
                rows.append({name: row[name] for name in self._fieldnames})
            return rows

    @staticmethod
    def _parse_epoch(row: dict[str, str]) -> int:
        try:
            return int(row["epoch"])
        except (KeyError, ValueError) as exc:
            raise ValueError("Training history contains a malformed epoch") from exc

    def _replace_rows(self, rows: list[dict[str, str]]) -> None:
        temporary = self._path.with_name(f".{self._path.name}.{uuid.uuid4().hex}.tmp")
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self._fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
        temporary.replace(self._path)


def _flat_metrics(
    train_metrics: EpochMetrics,
    val_metrics: EpochMetrics | None,
) -> dict[str, float]:
    metrics: dict[str, float] = {
        "loss_G_train": train_metrics.loss_G,
        "loss_D_train": train_metrics.loss_D,
    }
    _add_components(metrics, "train", train_metrics)
    if val_metrics is not None:
        metrics.update({"loss_G_val": val_metrics.loss_G, "loss_D_val": val_metrics.loss_D})
        _add_components(metrics, "val", val_metrics)
        metrics.update(
            {
                name: val_metrics.image.get(name, float("nan"))
                for name in VALIDATION_IMAGE_METRIC_NAMES
            }
        )
    return metrics


def _add_components(metrics: dict[str, float], stage: str, values: EpochMetrics) -> None:
    if values.raw or values.weighted or values.current_weight:
        metrics[f"loss_{stage}_total_generator"] = values.loss_G
        metrics[f"loss_{stage}_total_discriminator"] = values.loss_D
    for name, value in values.raw.items():
        metrics[f"loss_{stage}_raw_{name}"] = value
    for name, value in values.weighted.items():
        metrics[f"loss_{stage}_weighted_{name}"] = value
    for name, value in values.current_weight.items():
        metrics[f"loss_{stage}_current_weight_{name}"] = value


def _format_metric(value: float | None) -> str:
    return "" if value is None or not math.isfinite(value) else f"{value:.6f}"
