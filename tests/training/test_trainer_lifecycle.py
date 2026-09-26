"""Trainer lifecycle orchestration against a scripted method runtime."""

from __future__ import annotations

import csv
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from torch.utils.data import DataLoader

from virtual_staining.config.losses import LossConfig
from virtual_staining.config.project import ProjectConfig
from virtual_staining.config.training import (
    EarlyStoppingConfig,
    LearningRateSchedulerConfig,
    TrainingConfig,
)
from virtual_staining.experiment.run_layout import RunLayout, ensure_run_directories
from virtual_staining.experiment.session import ExperimentSession
from virtual_staining.training import progress as progress_module
from virtual_staining.training.helpers import (
    TrainingEpochAccumulator,
    build_lr_scheduler,
    loss_validation_metric,
    step_lr_schedulers,
)
from virtual_staining.training.progress import ProgressUpdate
from virtual_staining.training.results import TrainingResult
from virtual_staining.training.runtime import MethodMetrics
from virtual_staining.training.trainer import Trainer

_MISSING = object()


class _FakeMethod:
    """Scripted runtime: step loss is the batch mean; validation replays ``val_losses``."""

    name = "fake"
    pairing = "paired"
    input_names = ("LF",)
    output_names = ("stained",)
    prediction_directions = ("LF->stained",)
    default_checkpoint_metric = "loss_G_val"
    metric_names = ("loss_G", "loss_D")
    component_total_names = ("generator",)
    loss_names = ["generator_l1"]
    loss_config = LossConfig()

    def __init__(self, training: TrainingConfig, val_losses: Sequence[Any] = ()) -> None:
        self.training = training
        self.val_losses = list(val_losses)
        self.calls: list[tuple[Any, ...]] = []
        self.clock: _Clock | None = None
        self.optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
        self.scheduler = build_lr_scheduler(training, self.optimizer)
        if self.scheduler is not None:
            original_step = self.scheduler.step

            def counting_step(*args: Any) -> None:
                self.calls.append(("scheduler_step", *args))
                original_step(*args)

            self.scheduler.step = counting_step  # type: ignore[method-assign]

    def train_mode(self) -> None:
        pass

    def batch_size(self, batch: object) -> int:
        return len(cast(torch.Tensor, batch))

    def step(self, batch: object, *, epoch: int, global_step: int) -> MethodMetrics:
        self.optimizer.step()
        if self.clock is not None:
            self.clock.mono += 1.0 + global_step % 3
        value = float(cast(torch.Tensor, batch).mean())
        return MethodMetrics(
            losses={"loss_G": value, "loss_D": 2 * value},
            component_totals={"generator": 3 * value},
            raw={"generator_l1": 4 * value},
            weighted={"generator_l1": 5 * value},
            current_weight={"generator_l1": value},
        )

    def validate(self, loader: object, *, epoch: int, preview_sink: object = None) -> MethodMetrics:
        self.calls.append(("validate", epoch))
        value = self.val_losses.pop(0) if self.val_losses else 1.0
        losses = {"loss_D": 0.0} if value is _MISSING else {"loss_G": value, "loss_D": 0.0}
        return MethodMetrics(losses=losses)

    def validation_metric(self, metrics: MethodMetrics, name: str) -> float | None:
        return loss_validation_metric(metrics, name)

    def checkpoint_selection_metrics(self, metrics: MethodMetrics) -> dict[str, float]:
        value = metrics.losses.get("loss_G")
        return {} if value is None or not math.isfinite(value) else {"loss_G_val": value}

    def checkpoint_selection_modes(self) -> dict[str, str]:
        return {"loss_G_val": "min"}

    def step_schedulers(self, *, epoch: int, validation_metrics: MethodMetrics | None) -> bool:
        self.calls.append(("step_schedulers", epoch, validation_metrics is not None))
        return step_lr_schedulers(
            self.training.scheduler,
            (self.scheduler,),
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
        return {}

    def component_metadata(self) -> Mapping[str, object]:
        return {}

    def state_dict(self) -> dict[str, Any]:
        return {"weight": torch.zeros(1)}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        pass


class _Clock:
    def __init__(self) -> None:
        self.mono = 0.0

    def monotonic(self) -> float:
        return self.mono

    def time(self) -> float:
        return 1_900_000_000.0


def _training(**overrides: Any) -> TrainingConfig:
    values: dict[str, Any] = {
        "batch_size": 2,
        "epochs": 3,
        "lr_g": 1.0,
        "lr_d": 1.0,
        "beta1": 0.5,
        "beta2": 0.999,
        "seed": 0,
        "num_workers": 0,
        "validate_rate": 1,
        "checkpoint_rate": 1,
        "log_rate": 100,
    }
    values.update(overrides)
    return TrainingConfig(**values)


def _run(
    tmp_path: Path,
    training: TrainingConfig,
    *,
    samples: Sequence[float] = (1.0, 1.0, 1.0, 1.0),
    val_losses: Sequence[Any] = (),
    clock: _Clock | None = None,
) -> tuple[TrainingResult, list[ProgressUpdate], _FakeMethod, RunLayout]:
    project = ProjectConfig(
        dataset_root=tmp_path / "dataset",
        results_path=tmp_path / "results",
        run_name="run",
        image_size=(8, 8),
    )
    paths = RunLayout.from_project(project)
    ensure_run_directories(paths)
    method = _FakeMethod(training, val_losses)
    method.clock = clock
    loader = DataLoader(list(samples), batch_size=training.batch_size)  # pyright: ignore[reportArgumentType]
    updates: list[ProgressUpdate] = []
    trainer = Trainer(
        training,
        paths,
        method,  # pyright: ignore[reportArgumentType]
        loader,
        loader,
        torch.device("cpu"),
        train_dir=tmp_path / "train",
        val_dir=tmp_path / "val",
        experiment_session=cast(
            ExperimentSession, SimpleNamespace(log_metrics=lambda *_args, **_kwargs: None)
        ),
        config_hash="sha256:test",
        image_size=(8, 8),
        progress_reporter=updates.append,
    )
    return trainer.train(seed=0), updates, method, paths


def _rows(paths: RunLayout) -> list[dict[str, str]]:
    with paths.epochs_csv.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _lifecycle_updates(updates: list[ProgressUpdate], epoch: int) -> list[ProgressUpdate]:
    """Updates for ``epoch`` beyond the per-batch cadence (batch 0 and last batch)."""
    epoch_updates = [update for update in updates if update.epoch == epoch]
    return epoch_updates[2:]


def test_validation_and_scheduled_checkpoint_emit_one_epoch_final_update(tmp_path: Path) -> None:
    _, updates, _, _ = _run(tmp_path, _training(), val_losses=[3.0, 2.0, 1.0])

    for epoch in range(3):
        (final,) = _lifecycle_updates(updates, epoch)
        assert final.eval_epoch == epoch
        assert final.eval_metrics == {"loss_G": 3.0 - epoch, "loss_D": 0.0}
        assert final.last_checkpoint_name == f"ep{epoch:03d}.pth"
        assert final.best_checkpoint_name == f"ep{epoch:03d}.pth"
        assert final.best_checkpoint_metric_value == 3.0 - epoch
        assert final.step_metrics == {"loss_G": 1.0, "loss_D": 2.0}
        assert (final.batch_index, final.epoch_progress) == (1, 1.0)
    assert updates[-1].progress == 1.0
    assert updates[-1].eta_seconds == 0.0


def test_plain_epoch_and_unscheduled_final_checkpoint_progress(tmp_path: Path) -> None:
    training = _training(validate_rate=2, checkpoint_rate=2)
    _, updates, _, paths = _run(tmp_path, training, val_losses=[5.0])

    assert _lifecycle_updates(updates, 0) == []
    (epoch_1,) = _lifecycle_updates(updates, 1)
    assert (epoch_1.last_checkpoint_name, epoch_1.eval_epoch) == ("ep001.pth", 1)
    assert epoch_1.progress == pytest.approx(2 / 3)
    (completion,) = _lifecycle_updates(updates, 2)
    assert completion.last_checkpoint_name == "ep002.pth"
    assert completion.best_checkpoint_name == "ep001.pth"
    assert completion.eval_epoch == 1
    assert (completion.progress, completion.eta_seconds) == (1.0, 0.0)
    assert completion.estimated_end is not None
    assert sorted(p.name for p in paths.checkpoints_dir.glob("*.pth")) == [
        "ep001.pth",
        "ep002.pth",
    ]


def test_early_stop_completion_update_is_final_and_complete(tmp_path: Path) -> None:
    early = EarlyStoppingConfig(monitor="loss_G_val", mode="min", patience=1)
    training = _training(epochs=5, checkpoint_rate=10, early_stopping=early)
    result, updates, _, paths = _run(tmp_path, training, val_losses=[1.0, 2.0])

    assert (result.stopped_early, result.stop_epoch, result.final_epoch) == (True, 1, 1)
    (completion,) = _lifecycle_updates(updates, 1)
    assert updates[-1] is completion
    assert (completion.progress, completion.eta_seconds) == (1.0, 0.0)
    assert completion.last_checkpoint_name == "ep001.pth"
    assert [row["epoch"] for row in _rows(paths)] == ["0", "1"]


def test_eta_observes_every_batch_independent_of_log_rate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def timings(log_rate: int) -> dict[tuple[int, int], tuple[float, float | None]]:
        clock = _Clock()
        monkeypatch.setattr(progress_module, "time", clock)
        _, updates, _, _ = _run(
            tmp_path / str(log_rate),
            _training(batch_size=1, epochs=2, log_rate=log_rate, validate_rate=10),
            samples=[1.0] * 24,
            clock=clock,
        )
        return {  # updates[-1] is the completion event
            (u.epoch, u.batch_index): (u.elapsed_seconds, u.eta_seconds) for u in updates[:-1]
        }

    every_batch = timings(1)
    sparse = timings(12)
    assert len(every_batch) == 48
    assert sorted(sparse) == [(0, 0), (0, 12), (0, 23), (1, 0), (1, 12), (1, 23)]
    assert {key: every_batch[key] for key in sparse} == sparse
    assert sparse[(0, 23)][1] is not None


def test_uneven_final_batch_keeps_equal_weight_step_mean(tmp_path: Path) -> None:
    training = _training(batch_size=4, epochs=1, validate_rate=10)
    _, _, _, paths = _run(tmp_path, training, samples=[1.0, 1.0, 1.0, 1.0, 3.0])

    (row,) = _rows(paths)
    # (1 + 3) / 2 per optimization step, not the sample-weighted (4 * 1 + 3) / 5.
    assert row["loss_G_train"] == "2.000000"
    assert row["loss_D_train"] == "4.000000"
    assert row["loss_train_total_generator"] == "6.000000"
    assert row["loss_train_raw_generator_l1"] == "8.000000"
    assert row["loss_train_weighted_generator_l1"] == "10.000000"
    assert row["loss_train_current_weight_generator_l1"] == "2.000000"


def test_epoch_accumulator_counts_samples_without_weighting() -> None:
    accumulator = TrainingEpochAccumulator(
        metric_names=("loss_G",), component_total_names=("generator",), loss_names=["l1"]
    )
    for value, samples in ((1.0, 4), (3.0, 1)):
        accumulator.add(
            MethodMetrics(
                losses={"loss_G": value},
                component_totals={"generator": value},
                raw={"l1": value},
                weighted={"l1": value},
                current_weight={"l1": value},
            ),
            samples=samples,
        )

    assert (accumulator.steps, accumulator.samples) == (2, 5)
    assert accumulator.step_mean() == MethodMetrics(
        losses={"loss_G": 2.0},
        component_totals={"generator": 2.0},
        raw={"l1": 2.0},
        weighted={"l1": 2.0},
        current_weight={"l1": 2.0},
    )


def test_empty_training_loader_fails_before_publishing_epoch(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="Training loader was empty"):
        _run(tmp_path, _training(), samples=[])

    paths = RunLayout.from_project(
        ProjectConfig(
            dataset_root=tmp_path / "dataset",
            results_path=tmp_path / "results",
            run_name="run",
            image_size=(8, 8),
        )
    )
    assert _rows(paths) == []
    assert list(paths.checkpoints_dir.iterdir()) == []


@pytest.mark.parametrize("value", [math.nan, _MISSING])
def test_unavailable_checkpoint_metric_skips_ranking(tmp_path: Path, value: Any) -> None:
    training = _training(epochs=2, checkpoint_rate=10)
    result, updates, _, paths = _run(tmp_path, training, val_losses=[value, value])

    assert not (paths.checkpoints_dir / "best.json").exists()
    assert sorted(p.name for p in paths.checkpoints_dir.iterdir()) == ["ep001.pth"]
    assert result.best_checkpoint_path == paths.checkpoints_dir / "ep001.pth"
    assert all(update.best_checkpoint_metric_value is None for update in updates)
    assert all(row["loss_G_val"] == "" for row in _rows(paths))


def test_early_stopping_counts_only_finite_validation_events(tmp_path: Path) -> None:
    early = EarlyStoppingConfig(monitor="loss_G_val", mode="min", patience=2, min_delta=0.1)
    training = _training(epochs=10, checkpoint_rate=100, early_stopping=early)
    # improve, stale(1), NaN, missing, improve (reset), stale(1), stale(2) -> stop.
    values = [1.0, 0.95, math.nan, _MISSING, 0.5, 0.45, 0.44, 0.1]
    result, _, method, _ = _run(tmp_path, training, val_losses=values)

    assert result.stopped_early
    assert (result.stop_epoch, result.final_epoch) == (6, 6)
    assert (result.early_stopping_best_epoch, result.early_stopping_best_value) == (4, 0.5)
    assert result.stop_reason is not None and "loss_G_val" in result.stop_reason
    assert [call[1] for call in method.calls if call[0] == "validate"] == list(range(7))


def test_early_stopping_ignores_epochs_without_validation(tmp_path: Path) -> None:
    early = EarlyStoppingConfig(monitor="loss_G_val", mode="min", patience=1)
    training = _training(epochs=6, validate_rate=2, checkpoint_rate=100, early_stopping=early)
    result, _, _, _ = _run(tmp_path, training, val_losses=[1.0, 0.5, 0.7])

    assert (result.stop_epoch, result.early_stopping_best_epoch) == (5, 3)


def test_linear_decay_steps_once_per_epoch_after_validation(tmp_path: Path) -> None:
    scheduler = LearningRateSchedulerConfig(name="linear_decay", decay_start_epoch=0)
    training = _training(epochs=4, validate_rate=2, checkpoint_rate=100, scheduler=scheduler)
    _, _, method, _ = _run(tmp_path, training)

    assert method.calls == [
        ("step_schedulers", 0, False),
        ("scheduler_step",),
        ("validate", 1),
        ("step_schedulers", 1, True),
        ("scheduler_step",),
        ("step_schedulers", 2, False),
        ("scheduler_step",),
        ("validate", 3),
        ("step_schedulers", 3, True),
        ("scheduler_step",),
    ]


def test_plateau_steps_only_on_finite_validation_events(tmp_path: Path) -> None:
    scheduler = LearningRateSchedulerConfig(name="reduce_on_plateau", monitor="loss_G_val")
    training = _training(epochs=8, validate_rate=2, checkpoint_rate=100, scheduler=scheduler)
    _, _, method, _ = _run(tmp_path, training, val_losses=[1.0, math.nan, _MISSING, 0.5])

    assert [call for call in method.calls if call[0] == "scheduler_step"] == [
        ("scheduler_step", 1.0),
        ("scheduler_step", 0.5),
    ]
    assert [call for call in method.calls if call[0] == "step_schedulers"] == [
        ("step_schedulers", epoch, epoch % 2 == 1) for epoch in range(8)
    ]
