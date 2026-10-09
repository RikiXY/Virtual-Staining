from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Protocol, cast

from virtual_staining.config.run import RunConfig
from virtual_staining.config.stages import RunStageName
from virtual_staining.data.consumption import DataSnapshot, write_snapshot
from virtual_staining.experiment.run_layout import RunLayout, ensure_run_directories
from virtual_staining.experiment.snapshots import (
    save_environment_snapshot,
    save_stage_config_snapshots,
)

logger = logging.getLogger(__name__)
_PACKAGE_LOGGER = logging.getLogger("virtual_staining")
_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s - %(message)s"


Stage = RunStageName


class Reporter(Protocol):
    def start(self, run: Mapping[str, object]) -> None: ...

    def log_metrics(self, metrics: Mapping[str, float], *, step: int | None = None) -> None: ...

    def finish(self, status: str) -> None: ...


RUN_METADATA_SCHEMA_VERSION = 2
STAGE_RECORD_SCHEMA_VERSION = 2
_RUN_KEYS = frozenset(
    {
        "schema_version",
        "run_id",
        "run_name",
        "created_at",
        "training_data",
        "last_event_at",
        "stages_present",
        "last_completed_stage",
    }
)


class LocalRunStore:
    """Local single-writer run metadata store; concurrent writers are not coordinated."""

    def __init__(self, paths: RunLayout, *, run_name: str) -> None:
        self.paths = paths
        self.run_name = run_name

    def ensure_run(self) -> dict[str, object]:
        path = self.paths.run_metadata
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            data = self._read()
            if data.get("run_name") != self.run_name:
                raise ValueError(
                    f"Run name mismatch for {path}: expected {self.run_name!r}, "
                    f"found {data.get('run_name')!r}"
                )
            return data

        data: dict[str, object] = {
            "schema_version": RUN_METADATA_SCHEMA_VERSION,
            "run_id": str(uuid.uuid4()),
            "run_name": self.run_name,
            "created_at": datetime.now(UTC).isoformat(),
            "training_data": None,
            "last_event_at": None,
            "stages_present": [],
            "last_completed_stage": None,
        }
        _replace_json(path, data)
        return data

    def _read(self) -> dict[str, object]:
        path = self.paths.run_metadata
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"Run metadata at {path} must be an object")
        if data.get("schema_version") != RUN_METADATA_SCHEMA_VERSION or set(data) != _RUN_KEYS:
            raise ValueError(
                f"Run metadata at {path} has an unsupported schema; expected schema_version "
                f"{RUN_METADATA_SCHEMA_VERSION}. Start a new run directory."
            )
        if not isinstance(data.get("run_id"), str) or not data["run_id"]:
            raise ValueError(f"Run metadata at {path} has an invalid run_id")
        return data

    def training_data(self) -> dict[str, object] | None:
        value = self._read()["training_data"]
        return value if isinstance(value, dict) else None

    def bind_training_data(self, reference: Mapping[str, object]) -> None:
        """Make the training consumed-data identity part of run identity; conflicts fail."""
        data = self._read()
        identity = {
            key: reference[key] for key in ("snapshot_id", "membership_sha256", "hash_policy")
        }
        existing = data["training_data"]
        if isinstance(existing, dict) and existing.get("snapshot_id") != identity["snapshot_id"]:
            raise ValueError(
                "Training consumed-data identity conflicts with the existing run identity: "
                f"{existing.get('snapshot_id')!r} != {identity['snapshot_id']!r}. "
                "Train, validation, split, role, group, or content membership changed; "
                "use a new run_name instead of resuming or overwriting this run."
            )
        if existing != identity:
            data["training_data"] = identity
            _replace_json(self.paths.run_metadata, data)

    def record_stage(
        self,
        *,
        stage_record: Mapping[str, object],
        event: Mapping[str, object],
    ) -> None:
        event_data = dict(event)
        with self.paths.events.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event_data) + "\n")
            handle.flush()

        stage = cast(Stage, stage_record["stage"])
        _replace_json(self.paths.stage_record(stage), dict(stage_record))

        run_data = self._read()
        timestamp = event_data.get("timestamp")
        if timestamp is not None:
            run_data["last_event_at"] = timestamp
        stages_present = run_data.get("stages_present")
        if not isinstance(stages_present, list):
            stages_present = []
        if stage not in stages_present:
            stages_present.append(stage)
        run_data["stages_present"] = stages_present
        if event_data.get("status") == "completed":
            run_data["last_completed_stage"] = stage
        _replace_json(self.paths.run_metadata, run_data)


class _StageResult:
    def __init__(self) -> None:
        self.details: dict[str, object] = {}

    def result(self, **fields: object) -> None:
        self.details.update(fields)


class ExperimentSession:
    """Bootstrap one run-stage attempt, then bind the inputs the application resolved.

    Lifecycle: ``__enter__`` initializes the run, config, and environment snapshots;
    the application resolves its exact inputs and calls :meth:`bind_inputs`, which persists
    the consumed-data snapshot and only then publishes ``stage_started`` and starts reporters.
    A failure before binding is recorded as a failed attempt with ``consumed_data: null``
    and never starts reporters. The session never guesses what a stage consumes.
    """

    def __init__(
        self,
        *,
        config: RunConfig,
        config_path: Path,
        stage: Stage,
        reporters: Sequence[Reporter],
    ) -> None:
        self.config = config
        self.config_path = config_path
        self.stage: Stage = stage
        self.reporters = tuple(reporters)
        self.paths = RunLayout.from_project(config.project)
        self.config_hash: str | None = None
        self.consumed_data: dict[str, object] | None = None
        self.produced_data: dict[str, object] | None = None
        self._store: LocalRunStore | None = None
        self._run: dict[str, object] | None = None
        self._stage = _StageResult()
        self._started_at = ""
        self._file_handler: logging.FileHandler | None = None
        self._stage_started = False

    @classmethod
    def open(
        cls,
        *,
        config: RunConfig,
        config_path: Path,
        stage: Stage,
        reporters: Sequence[Reporter] = (),
    ) -> ExperimentSession:
        return cls(config=config, config_path=config_path, stage=stage, reporters=reporters)

    def __enter__(self) -> ExperimentSession:
        try:
            ensure_run_directories(self.paths)
            stage_layout = self.paths.stage(self.stage)
            assert self.config.project.run_name is not None
            self._store = LocalRunStore(self.paths, run_name=self.config.project.run_name)
            self._run = self._store.ensure_run()
            self._attach_file_handler()
            self._started_at = datetime.now(UTC).isoformat()
            self.config_hash = save_stage_config_snapshots(
                self.config,
                self.config_path,
                input_dest=stage_layout.input_config,
                resolved_dest=stage_layout.resolved_config,
            )
            save_environment_snapshot(stage_layout.environment)
            return self
        except Exception as error:
            if self._run is not None:
                self._complete(error)
            else:
                self._close_file_handler()
            raise

    def bind_inputs(self, snapshot: DataSnapshot) -> dict[str, object]:
        """Persist the stage's consumed-data snapshot, then start the stage normally."""
        if self._stage_started:
            raise RuntimeError(f"Stage {self.stage} inputs are already bound")
        if snapshot.kind != "consumed":
            raise ValueError("bind_inputs requires a consumed-data snapshot")
        assert self._store is not None and self._run is not None
        paths = self.paths.consumed_data(self.stage)
        if self.stage == "train":
            # Check before publishing so a conflicting attempt never replaces the snapshot
            # that backs the run's recorded training identity.
            self._store.bind_training_data(snapshot.reference(paths))
        self.paths.produced_data(self.stage).remove()
        self.consumed_data = write_snapshot(snapshot, paths)
        logger.info(
            "Stage %s consumed-data snapshot %s (%s rows, hash_policy=%s)",
            self.stage,
            snapshot.snapshot_id,
            len(snapshot.rows),
            snapshot.hash_policy,
        )
        self._store.record_stage(
            stage_record=self._stage_record(status="running", completed_at=None),
            event=self._event(
                timestamp=self._started_at, event_type="stage_started", status="running"
            ),
        )
        self._stage_started = True
        logger.info("Stage attempt started: %s", self.stage)
        run_view = {
            "run_id": self._run["run_id"],
            "run_name": self.config.project.run_name,
            "stage": self.stage,
            "config_hash": self.config_hash,
            "consumed_data_snapshot_id": snapshot.snapshot_id,
            "consumed_data_hash_policy": snapshot.hash_policy,
            "run_root": str(self.paths.root),
        }
        for reporter in self.reporters:
            try:
                reporter.start(run_view)
            except Exception as exc:
                logger.warning("Reporter start failed: %s", exc)
        return self.consumed_data

    def record_outputs(self, snapshot: DataSnapshot) -> dict[str, object]:
        """Persist the identity of files this stage produced (never called consumed)."""
        if not self._stage_started:
            raise RuntimeError("Stage outputs cannot be recorded before inputs are bound")
        if snapshot.kind != "produced":
            raise ValueError("record_outputs requires a produced-data snapshot")
        self.produced_data = write_snapshot(snapshot, self.paths.produced_data(self.stage))
        return self.produced_data

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, traceback
        if exc is None and not self._stage_started:
            error = RuntimeError(f"Stage {self.stage} finished without binding consumed inputs")
            self._complete(error)
            raise error
        self._complete(exc)

    def _complete(self, exc: BaseException | None) -> None:
        status = "failed" if exc is not None else "completed"
        completed_at = datetime.now(UTC).isoformat()
        if exc is not None:
            logger.error(
                "Stage %s failed: %s: %s",
                self.stage,
                type(exc).__name__,
                exc,
            )
        error_type = type(exc).__name__ if exc is not None else None
        error = str(exc) if exc is not None else None
        persistence_error: Exception | None = None
        try:
            assert self._store is not None
            self._store.record_stage(
                stage_record=self._stage_record(
                    status=status, completed_at=completed_at, error_type=error_type, error=error
                ),
                event=self._event(
                    timestamp=completed_at,
                    event_type=f"stage_{status}",
                    status=status,
                    error_type=error_type,
                    error=error,
                ),
            )
        except Exception as persistence_failure:
            persistence_error = persistence_failure
        if self._stage_started:
            for reporter in self.reporters:
                try:
                    reporter.finish(status)
                except Exception as reporter_error:
                    logger.warning("Reporter finish failed: %s", reporter_error)
        self._close_file_handler()

        if persistence_error is not None:
            if exc is not None:
                raise persistence_error from exc
            raise persistence_error

    def result(self, **fields: object) -> None:
        self._stage.result(**fields)

    @property
    def details(self) -> Mapping[str, object]:
        return self._stage.details

    def log_metrics(self, metrics: Mapping[str, float], *, step: int) -> None:
        clean = {
            name: float(value)
            for name, value in metrics.items()
            if isinstance(value, int | float) and _is_finite(float(value))
        }
        for reporter in self.reporters:
            try:
                reporter.log_metrics(clean, step=step)
            except Exception as exc:
                logger.warning("Reporter metrics failed: %s", exc)

    def _provenance(self) -> dict[str, object]:
        stage_layout = self.paths.stage(self.stage)
        return {
            "config": {
                "input_path": str(stage_layout.input_config),
                "resolved_path": str(stage_layout.resolved_config),
                "sha256": self.config_hash,
            },
            "environment_path": str(stage_layout.environment),
            "consumed_data": self.consumed_data,
            "produced_data": self.produced_data,
            "details": dict(self._stage.details),
        }

    def _stage_record(
        self,
        *,
        status: str,
        completed_at: str | None,
        error_type: str | None = None,
        error: str | None = None,
    ) -> dict[str, object]:
        record: dict[str, object] = {
            "schema_version": STAGE_RECORD_SCHEMA_VERSION,
            "stage": self.stage,
            "status": status,
            "started_at": self._started_at or None,
            "completed_at": completed_at,
            "entrypoint": f"vs {self.stage}",
            **self._provenance(),
        }
        if error_type is not None:
            record["error_type"] = error_type
        if error is not None:
            record["error"] = error
        return record

    def _event(
        self,
        *,
        timestamp: str,
        event_type: str,
        status: str,
        error_type: str | None = None,
        error: str | None = None,
    ) -> dict[str, object]:
        assert self._run is not None
        event: dict[str, object] = {
            "schema_version": STAGE_RECORD_SCHEMA_VERSION,
            "timestamp": timestamp,
            "run_id": self._run["run_id"],
            "run_name": self.config.project.run_name,
            "stage": self.stage,
            "event_type": event_type,
            "status": status,
            **self._provenance(),
        }
        if error_type is not None:
            event["error_type"] = error_type
        if error is not None:
            event["error"] = error
        return event

    def _attach_file_handler(self) -> None:
        handler = logging.FileHandler(self.paths.run_log, mode="a", encoding="utf-8")
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        _PACKAGE_LOGGER.addHandler(handler)
        _PACKAGE_LOGGER.setLevel(logging.DEBUG)
        self._file_handler = handler

    def _close_file_handler(self) -> None:
        if self._file_handler is None:
            return
        _PACKAGE_LOGGER.removeHandler(self._file_handler)
        self._file_handler.close()
        self._file_handler = None


def _replace_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _is_finite(value: float) -> bool:
    return value == value and value not in {float("inf"), float("-inf")}
