from __future__ import annotations

import json
import logging
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class _Stage:
    details: dict[str, Any] = field(default_factory=dict)

    def result(self, **fields: Any) -> None:
        """Add fields to the stage's final metadata and event."""
        self.details.update(fields)


@dataclass(frozen=True)
class RunProvenance:
    metadata_dir: Path
    run_name: str
    config_hash: str

    @contextmanager
    def stage(
        self,
        name: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> Iterator[_Stage]:
        """Record one stage's running, completed, or failed lifecycle."""
        started_at = datetime.now(UTC).isoformat()
        base_details = dict(details or {})
        stage = _Stage()
        self._write(name, "running", started_at, None, base_details)
        try:
            yield stage
        except Exception as exc:
            self._write(
                name,
                "failed",
                started_at,
                datetime.now(UTC).isoformat(),
                {**base_details, **stage.details, "error": str(exc)},
            )
            raise
        else:
            self._write(
                name,
                "completed",
                started_at,
                datetime.now(UTC).isoformat(),
                {**base_details, **stage.details},
            )

    def _write(
        self,
        stage: str,
        status: str,
        started_at: str,
        completed_at: str | None,
        details: dict[str, Any],
    ) -> None:
        payload = dict(details)
        payload.update(
            stage=stage,
            status=status,
            started_at=started_at,
            completed_at=completed_at,
            config_hash=self.config_hash,
        )
        save_stage_metadata(stage, payload, self.metadata_dir)
        append_run_event(
            {
                "timestamp": started_at if status == "running" else completed_at,
                "run_name": self.run_name,
                "stage": stage,
                "event_type": f"stage_{'started' if status == 'running' else status}",
                "status": status,
                "config_hash": self.config_hash,
                "details": details,
            },
            self.metadata_dir,
        )


@dataclass
class RunMetadata:
    run_name: str
    started_at: str
    git_commit: str | None = None
    git_dirty: bool | None = None
    config_hash: str | None = None
    manifest_path: str | None = None
    manifest_sha256: str | None = None
    seed: int | None = None
    device: str | None = None
    cuda_device_name: str | None = None
    entrypoint: str | None = None
    package_version: str | None = None
    last_event_at: str | None = None
    stages_present: list[str] = field(default_factory=list)
    last_completed_stage: str | None = None

    @classmethod
    def create(cls, run_name: str, entrypoint: str | None = None, **kwargs: Any) -> RunMetadata:
        git_commit, git_dirty = _capture_git_state()
        package_version = _package_version()
        return cls(
            run_name=run_name,
            started_at=datetime.now(UTC).isoformat(),
            entrypoint=entrypoint,
            git_commit=git_commit,
            git_dirty=git_dirty,
            package_version=package_version,
            **kwargs,
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(asdict(self), handle, indent=2, default=str)


def ensure_run_metadata(
    path: Path,
    *,
    run_name: str,
    entrypoint: str | None = None,
    **kwargs: Any,
) -> dict[str, Any] | None:
    """Create run.json if absent, or merge new stable provenance into the existing record."""
    try:
        existing = (
            json.loads(path.read_text(encoding="utf-8"))
            if path.exists()
            else asdict(RunMetadata.create(run_name=run_name, entrypoint=entrypoint))
        )
        if entrypoint is not None and existing.get("entrypoint") is None:
            existing["entrypoint"] = entrypoint
        for key, value in kwargs.items():
            if value is not None:
                existing[key] = value
        if "stages_present" not in existing or not isinstance(existing["stages_present"], list):
            existing["stages_present"] = []
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
        return existing
    except OSError as exc:
        logger.warning("Failed to write run metadata to %s: %s", path, exc)
        return None


def save_stage_metadata(stage: str, payload: dict[str, Any], metadata_dir: Path) -> Path | None:
    """Write the current state record for a stage under metadata/stages/."""
    stage_path = metadata_dir / "stages" / f"{stage}.json"
    try:
        stage_path.parent.mkdir(parents=True, exist_ok=True)
        stage_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return stage_path
    except OSError as exc:
        logger.warning("Failed to write stage metadata to %s: %s", stage_path, exc)
        return None


def append_run_event(event: dict[str, Any], metadata_dir: Path) -> Path | None:
    """Append a lifecycle event to metadata/events.jsonl and refresh run-level summary fields."""
    events_path = metadata_dir / "events.jsonl"
    try:
        events_path.parent.mkdir(parents=True, exist_ok=True)
        with events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event) + "\n")
        _update_run_summary_from_event(metadata_dir / "run.json", event)
        return events_path
    except OSError as exc:
        logger.warning("Failed to append run event to %s: %s", events_path, exc)
        return None


def _update_run_summary_from_event(path: Path, event: dict[str, Any]) -> None:
    if not path.exists():
        return

    data = json.loads(path.read_text(encoding="utf-8"))
    stage = event.get("stage")
    timestamp = event.get("timestamp")
    status = event.get("status")

    if timestamp is not None:
        data["last_event_at"] = timestamp
    if stage is not None:
        stages_present = data.get("stages_present")
        if not isinstance(stages_present, list):
            stages_present = []
        if stage not in stages_present:
            stages_present.append(stage)
        data["stages_present"] = stages_present
    if status == "completed" and stage is not None:
        data["last_completed_stage"] = stage

    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _capture_git_state() -> tuple[str | None, bool | None]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty_output = subprocess.check_output(
            ["git", "status", "--porcelain"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return commit, bool(dirty_output.strip())
    except Exception:
        return None, None


def _package_version() -> str | None:
    try:
        return version("virtual-staining")
    except PackageNotFoundError:
        return None
