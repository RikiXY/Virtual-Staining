"""Trainer lifecycle orchestration against a scripted method runtime."""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import pytest
import torch
from torch.utils.data import DataLoader

from tests.checkpoint_helpers import assert_nested_equal
from virtual_staining.checkpoint_contract import CheckpointIdentity, read_checkpoint
from virtual_staining.checkpoint_selection import resolve_checkpoint_path
from virtual_staining.config.project import ProjectConfig
from virtual_staining.config.scheduler import LearningRateSchedulerConfig, LearningRateSchedulerName
from virtual_staining.config.training import EarlyStoppingConfig, TrainingConfig
from virtual_staining.experiment.run_layout import RunLayout, ensure_run_directories
from virtual_staining.training import checkpoints as checkpoint_module
from virtual_staining.training import progress as progress_module
from virtual_staining.training.checkpoints import MethodCheckpointManager
from virtual_staining.training.helpers import (
    TrainingEpochAccumulator,
    build_lr_scheduler,
    loss_validation_metric,
    step_lr_schedulers,
)
from virtual_staining.training.progress import ProgressUpdate
from virtual_staining.training.results import TrainingResult
from virtual_staining.training.runtime import MethodMetrics
from virtual_staining.training.trainer import Trainer, _TrainingSession

_MISSING = object()


class _FakeMethod:
    """Scripted runtime: step loss is the batch mean; validation replays ``val_losses``."""

    name = "fake"
    default_checkpoint_metric = "loss_G_val"
    metric_names = ("loss_G", "loss_D")
    component_total_names = ("generator",)
    loss_names = ["generator_l1"]
    validation_metric_names = ("val_ssim",)

    def __init__(
        self,
        training: TrainingConfig,
        val_losses: Sequence[Any] = (),
        scheduler: LearningRateSchedulerConfig | None = None,
    ) -> None:
        self.training = training
        self.scheduler_config = scheduler or LearningRateSchedulerConfig()
        self.val_losses = list(val_losses)
        self.calls: list[tuple[Any, ...]] = []
        self.clock: _Clock | None = None
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.steps = 0
        self.optimizer = torch.optim.SGD([self.weight], lr=1.0, momentum=0.9)
        self.scheduler = build_lr_scheduler(self.scheduler_config, training.epochs, self.optimizer)
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
        self.weight.grad = torch.ones_like(self.weight)
        self.optimizer.step()
        self.steps += 1
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
        monitor = self.scheduler_config.monitor
        return step_lr_schedulers(
            self.scheduler_config,
            (self.scheduler,),
            epoch=epoch,
            monitor_value=(
                None
                if validation_metrics is None or monitor is None
                else lambda: self.validation_metric(validation_metrics, monitor)
            ),
        )

    def learning_rates(self) -> Mapping[str, float]:
        return {}

    def checkpoint_identity(self) -> CheckpointIdentity:
        return CheckpointIdentity(
            method=self.name,
            implementation={"version": "1", "source": "tests"},
            pairing="paired",
            inputs=("LF",),
            outputs=("stained",),
            prediction_directions=("forward",),
            options={},
            components={},
            image_size=(8, 8),
        )

    def objective_metadata(self) -> dict[str, Any]:
        return {"objective": "scripted"}

    def state_dict(self) -> dict[str, Any]:
        return {
            "weight": self.weight.detach(),
            "steps": self.steps,
            "optimizer": self.optimizer.state_dict(),
            # Exclude the test's call-counting wrapper from scheduler state.
            "scheduler": None
            if self.scheduler is None
            else {k: v for k, v in self.scheduler.state_dict().items() if k != "step"},
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        with torch.no_grad():
            self.weight.copy_(state["weight"])
        self.steps = state["steps"]
        self.optimizer.load_state_dict(state["optimizer"])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(state["scheduler"])


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
    scheduler: LearningRateSchedulerConfig | None = None,
    resume: bool = False,
) -> tuple[TrainingResult, list[ProgressUpdate], _FakeMethod, RunLayout]:
    project = ProjectConfig(
        dataset_root=tmp_path / "dataset",
        results_path=tmp_path / "results",
        run_name="run",
        image_size=(8, 8),
    )
    paths = RunLayout.from_project(project)
    ensure_run_directories(paths)
    method = _FakeMethod(training, val_losses, scheduler)
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
        config_hash="sha256:test",
        progress_reporter=updates.append,
    )
    start_epoch = trainer.resume("latest") if resume else 0
    return trainer.train(seed=0, start_epoch=start_epoch), updates, method, paths


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
    training = _training(epochs=4, validate_rate=2, checkpoint_rate=100)
    _, _, method, _ = _run(tmp_path, training, scheduler=scheduler)

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
    training = _training(epochs=8, validate_rate=2, checkpoint_rate=100)
    _, _, method, _ = _run(
        tmp_path, training, val_losses=[1.0, math.nan, _MISSING, 0.5], scheduler=scheduler
    )

    assert [call for call in method.calls if call[0] == "scheduler_step"] == [
        ("scheduler_step", 1.0),
        ("scheduler_step", 0.5),
    ]
    assert [call for call in method.calls if call[0] == "step_schedulers"] == [
        ("step_schedulers", epoch, epoch % 2 == 1) for epoch in range(8)
    ]


@pytest.fixture
def saved_epochs(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    epochs: list[int] = []
    original = MethodCheckpointManager.save

    def save(manager: MethodCheckpointManager, epoch: int) -> Path:
        path = original(manager, epoch)
        epochs.append(epoch)
        return path

    monkeypatch.setattr(MethodCheckpointManager, "save", save)
    return epochs


@pytest.mark.parametrize(
    ("validate_rate", "checkpoint_rate", "values", "expected"),
    [
        (1, 2, [1.0, 2.0, 3.0], [0, 1, 2]),
        (1, 1, [1.0, 2.0, 3.0], [0, 1, 2]),
        (2, 2, [1.0], [1, 2]),
        (10, 3, [], [2]),
        (1, 10, [1.0, 2.0, 3.0], [0, 1, 2]),
        (1, 2, [1.0, 2.0, math.nan], [0, 1, 2]),
        (1, 2, [1.0, 2.0, _MISSING], [0, 1, 2]),
        (1, 10, [math.nan] * 3, [2]),
        (1, 10, [_MISSING] * 3, [2]),
    ],
)
def test_final_publication_uses_actual_save_evidence(
    tmp_path: Path,
    saved_epochs: list[int],
    validate_rate: int,
    checkpoint_rate: int,
    values: list[Any],
    expected: list[int],
) -> None:
    with (
        patch.object(torch, "save", wraps=torch.save) as serialize,
        patch.object(checkpoint_module, "read_checkpoint", wraps=read_checkpoint) as readback,
        patch.object(
            checkpoint_module, "validate_checkpoint", wraps=checkpoint_module.validate_checkpoint
        ) as validate,
    ):
        result, updates, method, paths = _run(
            tmp_path,
            _training(validate_rate=validate_rate, checkpoint_rate=checkpoint_rate),
            val_losses=values,
        )
    assert saved_epochs == expected
    assert serialize.call_count == readback.call_count == validate.call_count == len(expected)
    assert sorted(p.name for p in paths.checkpoints_dir.glob("*.pth")) == [
        f"ep{epoch:03d}.pth" for epoch in expected
    ]
    payload = cast(dict, read_checkpoint(paths.checkpoints_dir / "ep002.pth"))
    assert payload["epoch"] == result.final_epoch == 2
    assert payload["format_version"] == 4
    assert payload["config_hash"] == "sha256:test"
    assert_nested_equal(payload["state"], method.state_dict())
    assert updates[-1].last_checkpoint_name == "ep002.pth"
    assert resolve_checkpoint_path(paths.checkpoints_dir, policy="latest").name == "ep002.pth"
    assert len(_lifecycle_updates(updates, 2)) == 1
    assert method.steps == 6


@pytest.mark.parametrize("scheduler_name", ["none", "linear_decay", "reduce_on_plateau"])
def test_reused_payload_and_history_match_previous_final_save(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    saved_epochs: list[int],
    scheduler_name: LearningRateSchedulerName,
) -> None:
    scheduler = LearningRateSchedulerConfig(
        name=scheduler_name, decay_start_epoch=0, monitor="loss_G_val", patience=0, factor=0.5
    )
    training = _training(checkpoint_rate=2, checkpoint_top_k=2)
    result, updates, method, paths = _run(
        tmp_path / "reuse", training, val_losses=[1.0, 2.0, 3.0], scheduler=scheduler
    )
    assert saved_epochs == [0, 1, 2]
    retained = read_checkpoint(paths.checkpoints_dir / "ep002.pth")

    def previous_final_save(trainer: Trainer, session: _TrainingSession) -> None:
        path = trainer._save_checkpoint(session.final_epoch)
        session.last_checkpoint = path.name

    monkeypatch.setattr(Trainer, "_save_final_checkpoint_if_needed", previous_final_save)
    saved_epochs.clear()
    old_result, old_updates, old_method, old_paths = _run(
        tmp_path / "previous", training, val_losses=[1.0, 2.0, 3.0], scheduler=scheduler
    )
    assert saved_epochs == [0, 1, 2, 2]
    assert_nested_equal(retained, read_checkpoint(old_paths.checkpoints_dir / "ep002.pth"))
    assert_nested_equal(method.state_dict(), old_method.state_dict())
    assert _rows(paths) == _rows(old_paths)
    assert method.calls == old_method.calls
    assert result.best_checkpoint_path == paths.checkpoints_dir / "ep000.pth"
    assert old_result.best_checkpoint_path == old_paths.checkpoints_dir / "ep000.pth"
    assert [
        (
            u.epoch,
            u.step_metrics,
            u.eval_metrics,
            u.last_checkpoint_name,
            u.best_checkpoint_name,
            u.progress,
        )
        for u in updates
    ] == [
        (
            u.epoch,
            u.step_metrics,
            u.eval_metrics,
            u.last_checkpoint_name,
            u.best_checkpoint_name,
            u.progress,
        )
        for u in old_updates
    ]
    catalog = json.loads((paths.checkpoints_dir / "best.json").read_text())
    assert catalog == json.loads((old_paths.checkpoints_dir / "best.json").read_text())
    records = catalog["metrics"]["loss_G_val"]["records"]
    assert [record["epoch"] for record in records] == [0, 1]
    assert all(record["objective_metadata"] == {"objective": "scripted"} for record in records)
    state = cast(dict, retained)["state"]
    if scheduler_name == "linear_decay":
        assert state["scheduler"]["last_epoch"] == 3
        assert state["optimizer"]["param_groups"][0]["lr"] == 0.0
    elif scheduler_name == "reduce_on_plateau":
        assert state["scheduler"]["last_epoch"] == 3
        assert state["scheduler"]["best"] == 1.0
        assert state["optimizer"]["param_groups"][0]["lr"] == 0.25


@pytest.mark.parametrize("rank", [True, False])
def test_early_stop_reuses_only_published_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, saved_epochs: list[int], rank: bool
) -> None:
    if not rank:
        monkeypatch.setattr(_FakeMethod, "checkpoint_selection_metrics", lambda *args: {})
    early = EarlyStoppingConfig(monitor="loss_G_val", mode="min", patience=1)
    result, updates, _, paths = _run(
        tmp_path,
        _training(epochs=5, checkpoint_rate=10, early_stopping=early),
        val_losses=[1.0, 2.0],
    )
    assert saved_epochs == ([0, 1] if rank else [1])
    assert (result.final_epoch, result.stop_epoch, result.stopped_early) == (1, 1, True)
    assert (result.early_stopping_best_epoch, result.early_stopping_best_value) == (0, 1.0)
    assert result.stop_reason is not None and "1 validation event(s)" in result.stop_reason
    assert result.best_checkpoint_path == paths.checkpoints_dir / (
        "ep000.pth" if rank else "ep001.pth"
    )
    assert updates[-1].last_checkpoint_name == "ep001.pth"
    assert updates[-1].progress == 1.0


@pytest.mark.parametrize("new_epochs", [False, True])
@pytest.mark.parametrize("early_stop", [False, True])
def test_resume_does_not_reuse_previous_execution_publication(
    tmp_path: Path, saved_epochs: list[int], new_epochs: bool, early_stop: bool
) -> None:
    _, _, _, paths = _run(tmp_path, _training(epochs=1, checkpoint_rate=2))
    original = (paths.checkpoints_dir / "ep000.pth").read_bytes()
    saved_epochs.clear()
    early = (
        EarlyStoppingConfig(monitor="loss_G_val", mode="min", patience=1) if early_stop else None
    )
    training = _training(
        epochs=(5 if early_stop else 3) if new_epochs else 1,
        checkpoint_rate=2,
        resume="latest",
        early_stopping=early,
    )
    result, updates, method, _ = _run(tmp_path, training, val_losses=[2.0, 3.0], resume=True)
    assert saved_epochs == ([1, 2] if new_epochs else [])
    assert result.final_epoch == (2 if new_epochs else 0)
    assert result.stopped_early == (new_epochs and early_stop)
    assert result.best_checkpoint_path == paths.checkpoints_dir / "ep000.pth"
    assert (paths.checkpoints_dir / "ep000.pth").read_bytes() == original
    assert method.steps == (6 if new_epochs else 2)
    if new_epochs:
        assert updates[-1].last_checkpoint_name == "ep002.pth"
        assert updates[-1].progress == 1.0
        assert_nested_equal(
            cast(dict, read_checkpoint(paths.checkpoints_dir / "ep002.pth"))["state"],
            method.state_dict(),
        )
    else:
        assert updates == []


@pytest.mark.parametrize("checkpoint_rate", [1, 2])
@pytest.mark.parametrize("scheduler_name", ["linear_decay", "reduce_on_plateau"])
def test_scheduler_operation_after_publication_requires_new_final_save(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    saved_epochs: list[int],
    checkpoint_rate: int,
    scheduler_name: LearningRateSchedulerName,
) -> None:
    original = Trainer._save_final_checkpoint_if_needed

    def finalize(trainer: Trainer, session: _TrainingSession) -> None:
        # Exercise a late scheduler operation, even on the regular checkpoint cadence.
        trainer._step_lr_schedulers(
            epoch=session.final_epoch, val_metrics=session.latest_eval_metrics
        )
        original(trainer, session)
        original(trainer, session)  # Repeated finalization must reuse the new publication.

    monkeypatch.setattr(Trainer, "_save_final_checkpoint_if_needed", finalize)
    scheduler = LearningRateSchedulerConfig(
        name=scheduler_name, decay_start_epoch=0, monitor="loss_G_val"
    )
    _, _, method, paths = _run(
        tmp_path, _training(checkpoint_rate=checkpoint_rate), scheduler=scheduler
    )
    assert saved_epochs == [0, 1, 2, 2]
    payload = cast(dict, read_checkpoint(paths.checkpoints_dir / "ep002.pth"))
    assert payload["state"]["scheduler"]["last_epoch"] == 4
    assert_nested_equal(payload["state"], method.state_dict())


@pytest.mark.parametrize("mutation", ["train", "validate", "resume", "other_epoch"])
def test_late_method_operations_invalidate_final_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, saved_epochs: list[int], mutation: str
) -> None:
    original = Trainer._save_final_checkpoint_if_needed

    def finalize(trainer: Trainer, session: _TrainingSession) -> None:
        if mutation == "train":
            trainer._train_epoch(session.final_epoch, session)
        elif mutation == "validate":
            # A method is allowed to maintain serialized validation state.
            def validate(method: _FakeMethod, *args: Any, **kwargs: Any) -> MethodMetrics:
                method.steps += 1
                return MethodMetrics(losses={"loss_G": 1.0, "loss_D": 0.0})

            monkeypatch.setattr(_FakeMethod, "validate", validate)
            trainer._validate(session.final_epoch)
        elif mutation == "resume":
            trainer.resume("ep000.pth")
        else:
            trainer._save_checkpoint(session.final_epoch - 1)
        original(trainer, session)

    monkeypatch.setattr(Trainer, "_save_final_checkpoint_if_needed", finalize)
    _, _, method, paths = _run(tmp_path, _training(checkpoint_rate=2))
    assert saved_epochs == ([0, 1, 2, 1, 2] if mutation == "other_epoch" else [0, 1, 2, 2])
    assert_nested_equal(
        cast(dict, read_checkpoint(paths.checkpoints_dir / "ep002.pth"))["state"],
        method.state_dict(),
    )


@pytest.mark.parametrize("operation", ["serialize", "readback", "validate", "publish"])
def test_checkpoint_failure_propagates_without_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    target, name = {
        "serialize": (torch, "save"),
        "readback": (checkpoint_module, "read_checkpoint"),
        "validate": (checkpoint_module, "validate_checkpoint"),
        "publish": (checkpoint_module.os, "replace"),
    }[operation]
    original_save = Trainer._save_checkpoint

    def checked_save(trainer: Trainer, epoch: int) -> Path:
        if epoch < 2:
            return original_save(trainer, epoch)
        with monkeypatch.context() as failure_patch:
            failure_patch.setattr(target, name, fail)
            with pytest.raises(RuntimeError, match="publication failed"):
                original_save(trainer, epoch)
        assert trainer._published_checkpoint is None
        raise RuntimeError("publication failed")

    def fail(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("publication failed")

    monkeypatch.setattr(Trainer, "_save_checkpoint", checked_save)
    with pytest.raises(RuntimeError, match="publication failed"):
        _run(tmp_path, _training())
    assert sorted(path.name for path in tmp_path.rglob("*.pth")) == ["ep000.pth", "ep001.pth"]
    assert list(tmp_path.rglob("*.tmp")) == []
    (catalog_path,) = tmp_path.rglob("best.json")
    catalog = json.loads(catalog_path.read_text())
    assert [record["epoch"] for record in catalog["metrics"]["loss_G_val"]["records"]] == [0, 1]


def test_final_reuse_keeps_independent_metric_rankings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, saved_epochs: list[int]
) -> None:
    monkeypatch.setattr(
        _FakeMethod,
        "checkpoint_selection_metrics",
        lambda self, metrics: {
            "loss_G_val": metrics.losses["loss_G"],
            "val_ssim": metrics.losses["loss_G"],
        },
    )
    monkeypatch.setattr(
        _FakeMethod,
        "checkpoint_selection_modes",
        lambda self: {"loss_G_val": "min", "val_ssim": "max"},
    )
    result, updates, _, paths = _run(
        tmp_path, _training(checkpoint_rate=2, checkpoint_top_k=1), val_losses=[1.0, 2.0, 3.0]
    )
    assert saved_epochs == [0, 1, 2]
    assert result.best_checkpoint_path == paths.checkpoints_dir / "ep000.pth"
    assert updates[-1].best_checkpoint_name == "ep000.pth"
    assert (
        resolve_checkpoint_path(paths.checkpoints_dir, policy="best", metric="val_ssim").name
        == "ep002.pth"
    )
    catalog = json.loads((paths.checkpoints_dir / "best.json").read_text())
    assert {
        name: [r["epoch"] for r in metric["records"]] for name, metric in catalog["metrics"].items()
    } == {"loss_G_val": [0], "val_ssim": [2]}
