from __future__ import annotations

import csv
import math
from contextlib import ExitStack
from pathlib import Path
from types import TracebackType
from typing import IO, Any

from virtual_staining.training.helpers import component_metric_row, metrics_fieldnames
from virtual_staining.training.results import EpochMetrics
from virtual_staining.training.validation_metrics import VALIDATION_IMAGE_METRIC_NAMES


class TrainingHistory:
    """Owns the three per-epoch training CSV files."""

    def __init__(self, metrics_dir: Path, loss_names: list[str]) -> None:
        self._metrics_dir = metrics_dir
        self._loss_names = loss_names
        self._stack = ExitStack()
        self._files: list[IO[str]] = []
        self._train_writer: csv.DictWriter | None = None
        self._validation_writer: csv.DictWriter | None = None
        self._all_writer: csv.DictWriter | None = None

    def __enter__(self) -> TrainingHistory:
        self._metrics_dir.mkdir(parents=True, exist_ok=True)
        train_file = self._open("train.csv")
        validation_file = self._open("validation.csv")
        all_file = self._open("all.csv")
        self._train_writer = csv.DictWriter(
            train_file,
            fieldnames=metrics_fieldnames(self._loss_names, stage="train"),
        )
        self._validation_writer = csv.DictWriter(
            validation_file,
            fieldnames=metrics_fieldnames(self._loss_names, stage="val")
            + VALIDATION_IMAGE_METRIC_NAMES,
        )
        self._all_writer = csv.DictWriter(
            all_file,
            fieldnames=metrics_fieldnames(self._loss_names) + VALIDATION_IMAGE_METRIC_NAMES,
        )
        for writer in (self._train_writer, self._validation_writer, self._all_writer):
            writer.writeheader()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._stack.__exit__(exc_type, exc_value, traceback)

    def _open(self, filename: str) -> IO[str]:
        file = self._stack.enter_context(
            (self._metrics_dir / filename).open("w", newline="", encoding="utf-8")
        )
        self._files.append(file)
        return file

    def write_epoch(
        self,
        epoch: int,
        train_metrics: EpochMetrics,
        val_metrics: EpochMetrics | None,
    ) -> None:
        assert self._train_writer is not None
        assert self._validation_writer is not None
        assert self._all_writer is not None

        train_row = {
            "epoch": epoch,
            "loss_G_train": f"{train_metrics.loss_G:.6f}",
            "loss_D_train": f"{train_metrics.loss_D:.6f}",
        }
        train_row.update(component_metric_row("train", train_metrics))
        self._train_writer.writerow(train_row)

        if val_metrics is not None:
            validation_row = {
                "epoch": epoch,
                "loss_G_val": f"{val_metrics.loss_G:.6f}",
                "loss_D_val": f"{val_metrics.loss_D:.6f}",
            }
            validation_row.update(component_metric_row("val", val_metrics))
            validation_row.update(_image_metric_row(val_metrics))
            self._validation_writer.writerow(validation_row)

        all_row = {
            "epoch": epoch,
            "loss_G_train": f"{train_metrics.loss_G:.6f}",
            "loss_D_train": f"{train_metrics.loss_D:.6f}",
            "loss_G_val": f"{val_metrics.loss_G:.6f}" if val_metrics else "",
            "loss_D_val": f"{val_metrics.loss_D:.6f}" if val_metrics else "",
        }
        all_row.update(component_metric_row("train", train_metrics))
        all_row.update(component_metric_row("val", val_metrics))
        if val_metrics is not None:
            all_row.update(_image_metric_row(val_metrics))
        self._all_writer.writerow(all_row)

        for file in self._files:
            file.flush()


def _image_metric_row(metrics: EpochMetrics) -> dict[str, Any]:
    return {
        name: f"{value:.6f}" if math.isfinite(value) else ""
        for name in VALIDATION_IMAGE_METRIC_NAMES
        if (value := metrics.image.get(name, float("nan"))) is not None
    }
