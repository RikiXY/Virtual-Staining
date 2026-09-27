from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from virtual_staining.config.project import ProjectConfig
from virtual_staining.data.consumption import AssetRow, DataSnapshot, build_snapshot
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


def _snapshot(config: Any, content: bytes = b"a") -> DataSnapshot:
    root = config.project.dataset_root
    (root / "x.png").write_bytes(content)
    return build_snapshot(
        [AssetRow(root="dataset", locator="x.png", role="input", domain="A", split="train")],
        kind="consumed",
        adapter="test/1",
        roots={"dataset": root},
        hash_policy="content",
    )


def _events(paths: RunLayout) -> list[dict[str, object]]:
    return [json.loads(line) for line in paths.events.read_text(encoding="utf-8").splitlines()]


def test_run_identity_is_stable_and_rejects_superseded_schema(tmp_path: Path) -> None:
    paths = RunLayout(tmp_path / "run")
    first = LocalRunStore(paths, run_name="demo").ensure_run()
    second = LocalRunStore(paths, run_name="demo").ensure_run()
    assert first["run_id"] == second["run_id"]
    assert first["schema_version"] == 2
    assert first["training_data"] is None
    assert "dataset_fingerprint" not in first
    with pytest.raises(ValueError, match="mismatch"):
        LocalRunStore(paths, run_name="other").ensure_run()
    paths.run_metadata.write_text(
        json.dumps({**first, "schema_version": 1, "dataset_fingerprint": None}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="unsupported schema"):
        LocalRunStore(paths, run_name="demo").ensure_run()


def test_training_identity_conflict_fails_without_replacing_bound_snapshot(
    tmp_path: Path,
) -> None:
    config, config_path = _config(tmp_path)
    with ExperimentSession.open(config=config, config_path=config_path, stage="train") as run:
        first = run.bind_inputs(_snapshot(config, b"a"))
    paths = RunLayout.from_project(config.project)
    run_data = json.loads(paths.run_metadata.read_text())
    assert run_data["training_data"]["snapshot_id"] == first["snapshot_id"]

    with (
        pytest.raises(ValueError, match="conflicts with the existing run identity"),
        ExperimentSession.open(config=config, config_path=config_path, stage="train") as run,
    ):
        run.bind_inputs(_snapshot(config, b"b"))
    stage = json.loads(paths.stage_record("train").read_text())
    assert stage["status"] == "failed"
    assert stage["consumed_data"] is None
    stored = json.loads(paths.consumed_data("train").metadata.read_text())
    assert stored["snapshot_id"] == first["snapshot_id"]


def test_session_overwrites_current_stage_and_appends_events(tmp_path: Path) -> None:
    config, config_path = _config(tmp_path)
    with ExperimentSession.open(config=config, config_path=config_path, stage="infer") as run:
        run.bind_inputs(_snapshot(config))
        run.result(attempt=1, inferred_count=1)
    with ExperimentSession.open(config=config, config_path=config_path, stage="infer") as run:
        reference = run.bind_inputs(_snapshot(config))
        run.result(attempt=2, inferred_count=2)

    paths = RunLayout.from_project(config.project)
    stage = json.loads(paths.stage_record("infer").read_text())
    assert stage["schema_version"] == 2
    assert stage["status"] == "completed"
    assert stage["details"] == {"attempt": 2, "inferred_count": 2}
    assert stage["consumed_data"] == reference
    assert stage["consumed_data"]["metadata_path"] == str(paths.consumed_data("infer").metadata)
    assert stage["produced_data"] is None
    assert [event["event_type"] for event in _events(paths)] == [
        "stage_started",
        "stage_completed",
        "stage_started",
        "stage_completed",
    ]


def test_session_records_failure_after_binding_and_keeps_snapshot(tmp_path: Path) -> None:
    config, config_path = _config(tmp_path)
    with (
        pytest.raises(RuntimeError, match="boom"),
        ExperimentSession.open(config=config, config_path=config_path, stage="infer") as run,
    ):
        reference = run.bind_inputs(_snapshot(config))
        run.result(inferred_count=1)
        raise RuntimeError("boom")
    paths = RunLayout.from_project(config.project)
    stage = json.loads(paths.stage_record("infer").read_text())
    assert stage["status"] == "failed"
    assert stage["error_type"] == "RuntimeError"
    assert stage["error"] == "boom"
    assert stage["consumed_data"] == reference
    assert paths.consumed_data("infer").metadata.is_file()


def test_reporters_start_after_binding_with_resolved_identity(tmp_path: Path) -> None:
    config, config_path = _config(tmp_path)
    events: list[tuple[str, object]] = []
    reporters = (BrokenReporter(events), Reporter(events))
    with ExperimentSession.open(
        config=config,
        config_path=config_path,
        stage="train",
        reporters=cast(Any, reporters),
    ) as run:
        assert events == []
        reference = run.bind_inputs(_snapshot(config))
        run.log_metrics({"loss": 1.0, "bad": float("nan")}, step=3)

    assert [name for name, _ in events] == [
        "start",
        "start",
        "metrics",
        "metrics",
        "finish",
        "finish",
    ]
    start_view = events[0][1]
    assert isinstance(start_view, dict)
    assert start_view["consumed_data_snapshot_id"] == reference["snapshot_id"]
    assert str(start_view["config_hash"]).startswith("sha256:")
    stage = json.loads((tmp_path / "run" / "metadata" / "stages" / "train.json").read_text())
    assert stage["status"] == "completed"


def test_failed_input_resolution_records_attempt_without_reporters(tmp_path: Path) -> None:
    config, config_path = _config(tmp_path)
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
        raise FileNotFoundError("Manifest not found at /missing. Run 'vs prepare'.")

    paths = RunLayout.from_project(config.project)
    stage = json.loads(paths.stage_record("infer").read_text())
    assert stage["status"] == "failed"
    assert stage["error_type"] == "FileNotFoundError"
    assert "Manifest not found at" in stage["error"]
    assert stage["consumed_data"] is None
    assert stage["config"]["sha256"].startswith("sha256:")
    assert [event["event_type"] for event in _events(paths)] == ["stage_failed"]
    assert "FileNotFoundError" in paths.run_log.read_text(encoding="utf-8")
    assert not paths.consumed_data("infer").metadata.exists()
    assert json.loads(paths.run_metadata.read_text())["training_data"] is None
    assert events == []


def test_stage_cannot_complete_without_binding_inputs(tmp_path: Path) -> None:
    config, config_path = _config(tmp_path)
    with (
        pytest.raises(RuntimeError, match="without binding consumed inputs"),
        ExperimentSession.open(config=config, config_path=config_path, stage="evaluate"),
    ):
        pass
    paths = RunLayout.from_project(config.project)
    assert json.loads(paths.stage_record("evaluate").read_text())["status"] == "failed"
