from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Literal, Protocol

from virtual_staining.config.run import RunConfig
from virtual_staining.experiment.run_paths import RunPaths
from virtual_staining.experiment.snapshots import (
    compute_manifest_hash,
    resolve_run_snapshot_paths,
    save_environment_snapshot,
    save_stage_config_snapshots,
)

logger = logging.getLogger(__name__)
_PACKAGE_LOGGER = logging.getLogger("virtual_staining")
_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s - %(message)s"
Stage = Literal["train", "infer", "evaluate"]


class Reporter(Protocol):
    def start(self, run: Mapping[str, object]) -> None: ...

    def log_metrics(self, metrics: Mapping[str, float], *, step: int | None = None) -> None: ...

    def finish(self, status: str) -> None: ...


class LocalRunStore:
    def __init__(
        self,
        paths: RunPaths,
        *,
        run_name: str,
        dataset_fingerprint: str | None,
    ) -> None:
        self.paths = paths
        self.run_name = run_name
        self.dataset_fingerprint = dataset_fingerprint

    def ensure_run(self) -> dict[str, object]:
        path = self.paths.run_metadata
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError(f"Run metadata at {path} must be an object")
            required = {
                "schema_version",
                "run_id",
                "run_name",
                "created_at",
                "dataset_fingerprint",
                "last_event_at",
                "stages_present",
                "last_completed_stage",
            }
            if data.get("schema_version") != 1 or not required.issubset(data):
                raise ValueError(f"Run metadata at {path} has an unsupported schema")
            if not isinstance(data.get("run_id"), str) or not data["run_id"]:
                raise ValueError(f"Run metadata at {path} has an invalid run_id")
            if data.get("run_name") != self.run_name:
                raise ValueError(
                    f"Run name mismatch for {path}: expected {self.run_name!r}, "
                    f"found {data.get('run_name')!r}"
                )
            existing_fingerprint = data["dataset_fingerprint"]
            if (
                existing_fingerprint is not None
                and self.dataset_fingerprint is not None
                and existing_fingerprint != self.dataset_fingerprint
            ):
                raise ValueError(
                    "Dataset fingerprint conflicts with the existing run identity: "
                    f"{existing_fingerprint!r} != {self.dataset_fingerprint!r}"
                )
            if existing_fingerprint is None and self.dataset_fingerprint is not None:
                data["dataset_fingerprint"] = self.dataset_fingerprint
                _replace_json(path, data)
            return data

        data: dict[str, object] = {
            "schema_version": 1,
            "run_id": str(uuid.uuid4()),
            "run_name": self.run_name,
            "created_at": datetime.now(UTC).isoformat(),
            "dataset_fingerprint": self.dataset_fingerprint,
            "last_event_at": None,
            "stages_present": [],
            "last_completed_stage": None,
        }
        _replace_json(path, data)
        return data

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

        stage = str(stage_record["stage"])
        _replace_json(self.paths.metadata_dir / "stages" / f"{stage}.json", dict(stage_record))

        run_data = json.loads(self.paths.run_metadata.read_text(encoding="utf-8"))
        if not isinstance(run_data, dict):
            raise ValueError(f"Run metadata at {self.paths.run_metadata} must be an object")
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
        self.paths = RunPaths(config.project.run_root)
        self.config_hash = ""
        self.manifest_hash = ""
        self.dataset_fingerprint: str | None = None
        self._store: LocalRunStore | None = None
        self._run: dict[str, object] | None = None
        self._stage = _StageResult()
        self._started_at = ""
        self._file_handler: logging.FileHandler | None = None

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
            self.paths.create_directories()
            snapshot_paths = resolve_run_snapshot_paths(stage=self.stage, run_paths=self.paths)
            self.config_hash = save_stage_config_snapshots(
                self.config,
                self.config_path,
                input_dest=snapshot_paths.input_config,
                resolved_dest=snapshot_paths.resolved_config,
            )
            save_environment_snapshot(snapshot_paths.environment)

            manifest_path = self.config.project.manifest_path
            if not manifest_path.is_file():
                raise FileNotFoundError(f"Manifest not found at {manifest_path}. Run 'vs prepare'.")
            manifest_hash = compute_manifest_hash(manifest_path)
            self.manifest_hash = manifest_hash
            fingerprint_path = (
                self.config.project.dataset_root / "metadata" / "dataset_fingerprint.json"
            )
            fingerprint = _load_dataset_fingerprint(fingerprint_path)
            self.dataset_fingerprint = fingerprint
            self._store = LocalRunStore(
                self.paths,
                run_name=self.config.project.run_name,
                dataset_fingerprint=fingerprint,
            )
            self._run = self._store.ensure_run()
            self._attach_file_handler()
            self._started_at = datetime.now(UTC).isoformat()
            config_view = {
                "input_path": str(snapshot_paths.input_config),
                "resolved_path": str(snapshot_paths.resolved_config),
                "sha256": self.config_hash,
            }
            dataset_view = {
                "fingerprint": fingerprint,
                "manifest_path": str(manifest_path),
                "manifest_sha256": manifest_hash,
            }
            stage_record = self._stage_record(
                status="running",
                started_at=self._started_at,
                completed_at=None,
                config=config_view,
                environment_path=str(snapshot_paths.environment),
                dataset=dataset_view,
                details={},
            )
            self._store.record_stage(
                stage_record=stage_record,
                event=self._event(
                    timestamp=self._started_at,
                    event_type="stage_started",
                    status="running",
                    config=config_view,
                    environment_path=str(snapshot_paths.environment),
                    dataset=dataset_view,
                    details={},
                ),
            )
            logger.info("Stage attempt started: %s", self.stage)
            run_view = {
                "run_id": self._run["run_id"],
                "run_name": self.config.project.run_name,
                "stage": self.stage,
                "config_hash": self.config_hash,
                "dataset_fingerprint": fingerprint,
                "manifest_sha256": manifest_hash,
                "run_root": str(self.paths.root),
            }
            for reporter in self.reporters:
                try:
                    reporter.start(run_view)
                except Exception as exc:
                    logger.warning("Reporter start failed: %s", exc)
            return self
        except Exception:
            self._close_file_handler()
            raise

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, traceback
        status = "failed" if exc is not None else "completed"
        completed_at = datetime.now(UTC).isoformat()
        persistence_error: Exception | None = None
        try:
            assert self._store is not None
            stage_record = self._stage_record(
                status=status,
                started_at=self._started_at,
                completed_at=completed_at,
                config=self._config_view(),
                environment_path=self._environment_path(),
                dataset=self._dataset_view(),
                details=self._stage.details,
                error_type=type(exc).__name__ if exc is not None else None,
                error=str(exc) if exc is not None else None,
            )
            event = self._event(
                timestamp=completed_at,
                event_type=f"stage_{status}",
                status=status,
                config=self._config_view(),
                environment_path=self._environment_path(),
                dataset=self._dataset_view(),
                details=self._stage.details,
                error_type=type(exc).__name__ if exc is not None else None,
                error=str(exc) if exc is not None else None,
            )
            self._store.record_stage(stage_record=stage_record, event=event)
        except Exception as error:
            persistence_error = error
        else:
            for reporter in self.reporters:
                try:
                    reporter.finish(status)
                except Exception as error:
                    logger.warning("Reporter finish failed: %s", error)
        finally:
            self._close_file_handler()

        if persistence_error is not None:
            if exc is not None:
                raise persistence_error from exc
            raise persistence_error

    def result(self, **fields: object) -> None:
        self._stage.result(**fields)

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

    def _stage_record(
        self,
        *,
        status: str,
        started_at: str,
        completed_at: str | None,
        config: Mapping[str, object],
        environment_path: str,
        dataset: Mapping[str, object],
        details: Mapping[str, object],
        error_type: str | None = None,
        error: str | None = None,
    ) -> dict[str, object]:
        record: dict[str, object] = {
            "schema_version": 1,
            "stage": self.stage,
            "status": status,
            "started_at": started_at,
            "completed_at": completed_at,
            "entrypoint": f"vs {self.stage}",
            "config": dict(config),
            "environment_path": environment_path,
            "dataset": dict(dataset),
            "details": dict(details),
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
        config: Mapping[str, object],
        environment_path: str,
        dataset: Mapping[str, object],
        details: Mapping[str, object],
        error_type: str | None = None,
        error: str | None = None,
    ) -> dict[str, object]:
        assert self._run is not None
        event: dict[str, object] = {
            "schema_version": 1,
            "timestamp": timestamp,
            "run_id": self._run["run_id"],
            "run_name": self.config.project.run_name,
            "stage": self.stage,
            "event_type": event_type,
            "status": status,
            "config": dict(config),
            "environment_path": environment_path,
            "dataset": dict(dataset),
            "details": dict(details),
        }
        if error_type is not None:
            event["error_type"] = error_type
        if error is not None:
            event["error"] = error
        return event

    def _config_view(self) -> dict[str, object]:
        return {
            "input_path": str(self.paths.stage_config_dir(self.stage) / "input.yaml"),
            "resolved_path": str(self.paths.stage_config_dir(self.stage) / "resolved.yaml"),
            "sha256": self.config_hash,
        }

    def _environment_path(self) -> str:
        return str(self.paths.stage_environment(self.stage))

    def _dataset_view(self) -> dict[str, object]:
        return {
            "fingerprint": self.dataset_fingerprint,
            "manifest_path": str(self.config.project.manifest_path),
            "manifest_sha256": self.manifest_hash,
        }

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


def _load_dataset_fingerprint(path: Path) -> str | None:
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Dataset fingerprint at {path} must be an object")
    value = data.get("fingerprint")
    return str(value) if value is not None else None


def _is_finite(value: float) -> bool:
    return value == value and value not in {float("inf"), float("-inf")}
