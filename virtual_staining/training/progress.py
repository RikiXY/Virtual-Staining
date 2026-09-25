from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TypeAlias


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "--"
    seconds = int(max(seconds, 0))
    if seconds < 60:
        return f"{seconds}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {sec:02d}s"
    hours, minutes = divmod(minutes, 60)
    if hours < 24:
        return f"{hours}h {minutes:02d}m"
    days, hours = divmod(hours, 24)
    return f"{days}d {hours:02d}h"


@dataclass(frozen=True)
class ProgressUpdate:
    progress: float
    epoch_progress: float
    epoch: int
    batch_index: int
    total_epochs: int
    total_batches: int
    step_metrics: Mapping[str, float]
    eval_metrics: Mapping[str, float] | None
    eval_epoch: int | None
    elapsed_str: str
    eta_str: str
    end_time_str: str
    last_checkpoint_name: str
    best_checkpoint_name: str
    best_checkpoint_metric_name: str
    best_checkpoint_metric_value: float | None


ProgressReporter: TypeAlias = Callable[[ProgressUpdate], None]


def _format_metrics(metrics: Mapping[str, float]) -> str:
    return " | ".join(f"{name} {value:.4f}" for name, value in metrics.items()) or "metrics --"


def format_progress_log(update: ProgressUpdate) -> str:
    first_line = (
        f"ep {update.epoch + 1}/{update.total_epochs} "
        f"({update.progress:.2%}) | "
        f"b {update.batch_index + 1}/{update.total_batches} "
        f"({update.epoch_progress:.0%}) | "
        f"{_format_metrics(update.step_metrics)} | "
        f"elapsed {update.elapsed_str} | "
        f"ETA {update.eta_str} | "
        f"end {update.end_time_str} | "
        f"last ckpt {update.last_checkpoint_name.strip()}"
    )
    if update.eval_metrics is None:
        eval_parts = "eval --"
    else:
        eval_epoch = f"ep {update.eval_epoch + 1} | " if update.eval_epoch is not None else ""
        eval_parts = f"eval {eval_epoch}{_format_metrics(update.eval_metrics)}"
    best_value = (
        "--"
        if update.best_checkpoint_metric_value is None
        else f"{update.best_checkpoint_metric_value:.4f}"
    )
    return (
        f"{first_line}\n"
        f"{eval_parts} | best ckpt {update.best_checkpoint_name.strip()} "
        f"({update.best_checkpoint_metric_name} {best_value})"
    )


class ProgressTracker:
    def __init__(
        self,
        total_epochs: int,
        total_batches: int,
        start_epoch: int = 0,
        max_history: int = 300,
        warmup_batches: int = 10,
        min_eta_batches: int = 5,
    ) -> None:
        self.total_epochs = total_epochs
        self.total_batches = total_batches
        self.start_epoch = start_epoch
        self.max_history = max_history
        self.warmup_batches = warmup_batches
        self.min_eta_batches = min_eta_batches

        self.total_steps = total_epochs * total_batches
        self.start_step = start_epoch * total_batches

        self.start_time: float | None = None
        self.last_step_time: float | None = None
        self.step_durations: list[float] = []

    def start(self) -> None:
        now = time.time()
        self.start_time = now
        self.last_step_time = now
        self.step_durations = []

    def calculate_progress(
        self, epoch: int, batch: int
    ) -> tuple[float, float, float | None, float | None]:
        now = time.time()

        current_step = epoch * self.total_batches + batch + 1
        completed_since_start = current_step - self.start_step
        remaining_steps = max(self.total_steps - current_step, 0)

        assert self.last_step_time is not None, "call start() before calculate_progress()"
        step_duration = now - self.last_step_time
        self.last_step_time = now

        if completed_since_start > self.warmup_batches:
            self.step_durations.append(step_duration)
            if len(self.step_durations) > self.max_history:
                self.step_durations.pop(0)

        assert self.start_time is not None, "call start() before calculate_progress()"
        total_elapsed_time = now - self.start_time
        progress = current_step / self.total_steps if self.total_steps > 0 else 1.0

        if len(self.step_durations) < self.min_eta_batches:
            eta = None
            end_time = None
        else:
            avg_step_time = sum(self.step_durations) / len(self.step_durations)
            eta = avg_step_time * remaining_steps
            end_time = now + eta

        return progress, total_elapsed_time, eta, end_time
