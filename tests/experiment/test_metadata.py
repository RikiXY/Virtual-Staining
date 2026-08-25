from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from virtual_staining.config.project import ProjectConfig
from virtual_staining.experiment.run_layout import RunLayout
from virtual_staining.experiment.session import ExperimentSession, LocalRunStore


class Reporter:
    def __init__(self, events: list[tuple[str, object]]) -> None:
        self.events = events

    def start(self, run: dict[str, object]) -> None:
        self.events.append(("start", run))

    def log_metrics(self, metrics: dict[str, float], *, step: int | None = None) -> None:
        self.events.append(("metrics", (metrics, step)))

    def finish(self, status: str) -> None:
        self.events.append(("finish", status))


class BrokenReporter(Reporter):
    def start(self, run: dict[str, object]) -> None:
        super().start(run)
        raise RuntimeError("reporter start")

    def log_metrics(self, metrics: dict[str, float], *, step: int | None = None) -> None:
        super().log_metrics(metrics, step=step)
        raise RuntimeError("reporter metrics")

    def finish(self, status: str) -> None:
        super().finish(status)
        raise RuntimeError("reporter finish")


def _config(tmp_path: Path) -> tuple[Any, Path]:
    dataset_root = tmp_path / "dataset"
    manifest = dataset_root / "manifests" / "manifest.csv"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("sample_id\n", encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text("project: test\n", encoding="utf-8")
    project = ProjectConfig(
        dataset_root=dataset_root,
        results_path=tmp_path,
        run_name="run",
        image_size=(256, 256),
    )
    config = SimpleNamespace(project=project, to_dict=lambda: {"project": "test"})
    return config, config_path


def _events(paths: RunLayout) -> list[dict[str, object]]:
    return [json.loads(line) for line in paths.events.read_text(encoding="utf-8").splitlines()]


def test_run_identity_is_stable_and_fingerprint_conflicts_fail(tmp_path: Path) -> None:
    paths = RunLayout(tmp_path / "run")
    first = LocalRunStore(paths, run_name="demo", dataset_fingerprint="sha256:a").ensure_run()
    second = LocalRunStore(paths, run_name="demo", dataset_fingerprint="sha256:a").ensure_run()
    assert first["run_id"] == second["run_id"]
    with pytest.raises(ValueError, match="conflicts"):
        LocalRunStore(paths, run_name="demo", dataset_fingerprint="sha256:b").ensure_run()
    with pytest.raises(ValueError, match="mismatch"):
        LocalRunStore(paths, run_name="other", dataset_fingerprint="sha256:a").ensure_run()


def test_session_overwrites_current_stage_and_appends_events(tmp_path: Path) -> None:
    config, config_path = _config(tmp_path)
    with ExperimentSession.open(config=config, config_path=config_path, stage="infer") as run:
        run.result(attempt=1, inferred_count=1)
    with ExperimentSession.open(config=config, config_path=config_path, stage="infer") as run:
        run.result(attempt=2, inferred_count=2)

    paths = RunLayout.from_project(config.project)
    stage = json.loads((paths.metadata_dir / "stages" / "infer.json").read_text())
    assert stage["status"] == "completed"
    assert stage["details"] == {"attempt": 2, "inferred_count": 2}
    assert [event["event_type"] for event in _events(paths)] == [
        "stage_started",
        "stage_completed",
        "stage_started",
        "stage_completed",
    ]


def test_session_records_failure_and_reraises(tmp_path: Path) -> None:
    config, config_path = _config(tmp_path)
    with (
        pytest.raises(RuntimeError, match="boom"),
        ExperimentSession.open(config=config, config_path=config_path, stage="infer") as run,
    ):
        run.result(inferred_count=1)
        raise RuntimeError("boom")
    paths = RunLayout.from_project(config.project)
    stage = json.loads((paths.metadata_dir / "stages" / "infer.json").read_text())
    assert stage["status"] == "failed"
    assert stage["error_type"] == "RuntimeError"
    assert stage["error"] == "boom"


def test_reporters_run_after_local_writes_and_fail_independently(tmp_path: Path) -> None:
    config, config_path = _config(tmp_path)
    events: list[tuple[str, object]] = []
    reporters = (BrokenReporter(events), Reporter(events))
    with ExperimentSession.open(
        config=config,
        config_path=config_path,
        stage="train",
        reporters=cast(Any, reporters),
    ) as run:
        run.log_metrics({"loss": 1.0, "bad": float("nan")}, step=3)

    assert [name for name, _ in events] == [
        "start",
        "start",
        "metrics",
        "metrics",
        "finish",
        "finish",
    ]
    stage = json.loads((tmp_path / "run" / "metadata" / "stages" / "train.json").read_text())
    assert stage["status"] == "completed"


def test_session_records_missing_manifest_bootstrap_failure(tmp_path: Path) -> None:
    config, config_path = _config(tmp_path)
    manifest = config.project.dataset_root / "manifests" / "manifest.csv"
    manifest.unlink()
    events: list[tuple[str, object]] = []

    with (
        pytest.raises(FileNotFoundError, match="Manifest not found at"),
        ExperimentSession.open(
            config=config,
            config_path=config_path,
            stage="infer",
            reporters=cast(Any, (Reporter(events),)),
        ),
    ):
        raise AssertionError("bootstrap should fail before entering the body")

    paths = RunLayout.from_project(config.project)
    stage = json.loads((paths.metadata_dir / "stages" / "infer.json").read_text())
    assert stage["status"] == "failed"
    assert stage["error_type"] == "FileNotFoundError"
    assert "Manifest not found at" in stage["error"]
    assert [event["event_type"] for event in _events(paths)] == [
        "stage_started",
        "stage_failed",
    ]
    assert "FileNotFoundError" in paths.run_log.read_text(encoding="utf-8")
    start_view = events[0][1]
    assert isinstance(start_view, dict)
    assert start_view["config_hash"] == ""
    assert start_view["manifest_sha256"] == ""
    assert start_view["dataset_fingerprint"] is None
    assert [name for name, _ in events] == ["start", "finish"]
    assert events[-1] == ("finish", "failed")
