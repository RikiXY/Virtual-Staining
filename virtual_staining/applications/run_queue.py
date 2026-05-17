from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from virtual_staining.applications.run_stages import (
    DEFAULT_FULL_RUN_STAGES,
    VALID_STAGES,
    run_stages,
)
from virtual_staining.config.loader import load_yaml_mapping
from virtual_staining.config.run import RunConfig
from virtual_staining.config.validation import parse_bool_strict, reject_unknown_keys

_QUEUE_KEYS: frozenset[str] = frozenset({"name", "continue_on_failure", "jobs"})
_QUEUE_JOB_KEYS: frozenset[str] = frozenset({"config_path", "label", "notes", "stages"})


@dataclass(frozen=True)
class QueueJob:
    config_path: Path
    label: str | None = None
    notes: str | None = None
    stages: tuple[str, ...] | None = None


@dataclass(frozen=True)
class LocalRunQueue:
    name: str
    continue_on_failure: bool
    jobs: tuple[QueueJob, ...]
    path: Path

    @property
    def state_path(self) -> Path:
        return _resolve_queue_state_path(self.path, self.name)


@dataclass
class QueueJobState:
    index: int
    config_path: str
    label: str | None
    notes: str | None
    stages: list[str] | None = None
    status: str = "pending"
    started_at: str | None = None
    completed_at: str | None = None
    error: str | None = None


@dataclass
class QueueState:
    queue_name: str
    queue_path: str
    continue_on_failure: bool
    status: str
    started_at: str | None
    completed_at: str | None
    current_job_index: int | None
    jobs: list[QueueJobState] = field(default_factory=list)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")


def _resolve_queue_state_path(queue_path: Path, queue_name: str) -> Path:
    parts = queue_path.parts
    for index in range(len(parts) - 1):
        if parts[index] == "config" and parts[index + 1] == "queues":
            repo_root = Path(*parts[:index])
            return repo_root / "local_workspace" / "queues" / f"{queue_name}.state.json"
    return queue_path.with_suffix(".state.json")


def load_local_run_queue(queue_path: Path) -> LocalRunQueue:
    data = load_yaml_mapping(queue_path)
    reject_unknown_keys(data, _QUEUE_KEYS, "queue")

    raw_name = data.get("name")
    if not isinstance(raw_name, str) or not raw_name.strip():
        raise ValueError("Queue file must define a non-empty 'name'.")

    raw_continue = parse_bool_strict(data.get("continue_on_failure", False), "continue_on_failure")

    raw_jobs = data.get("jobs")
    if not isinstance(raw_jobs, list) or not raw_jobs:
        raise ValueError("Queue file must define a non-empty 'jobs' list.")

    jobs: list[QueueJob] = []
    for index, raw_job in enumerate(raw_jobs):
        if not isinstance(raw_job, dict):
            raise ValueError(f"Queue job at index {index} must be a mapping.")

        reject_unknown_keys(raw_job, _QUEUE_JOB_KEYS, f"queue.jobs[{index}]")

        raw_config_path = raw_job.get("config_path")
        if not isinstance(raw_config_path, str) or not raw_config_path.strip():
            raise ValueError(f"Queue job at index {index} must define 'config_path'.")

        label = raw_job.get("label")
        notes = raw_job.get("notes")

        if label is not None and not isinstance(label, str):
            raise ValueError(f"Queue job at index {index} has non-string 'label'.")
        if notes is not None and not isinstance(notes, str):
            raise ValueError(f"Queue job at index {index} has non-string 'notes'.")

        raw_stages = raw_job.get("stages")
        stages: tuple[str, ...] | None = None

        if raw_stages is not None:
            if not isinstance(raw_stages, list) or not raw_stages:
                raise ValueError(
                    f"Queue job at index {index} has invalid 'stages': expected a non-empty list."
                )

            if not all(isinstance(stage, str) for stage in raw_stages):
                raise ValueError(
                    f"Queue job at index {index} has invalid 'stages': all stages must be strings."
                )

            stages = tuple(raw_stages)
            unknown_stages = [stage for stage in stages if stage not in VALID_STAGES]
            if unknown_stages:
                allowed = ", ".join(VALID_STAGES)
                unknown = ", ".join(unknown_stages)
                raise ValueError(
                    f"Queue job at index {index} has unknown stage(s): {unknown}. "
                    f"Allowed stages: {allowed}"
                )

        config_path = Path(raw_config_path)
        if not config_path.is_absolute():
            config_path = (queue_path.parent / config_path).resolve()
        else:
            config_path = config_path.resolve()

        jobs.append(
            QueueJob(
                config_path=config_path,
                label=label,
                notes=notes,
                stages=stages,
            )
        )

    return LocalRunQueue(
        name=raw_name.strip(),
        continue_on_failure=raw_continue,
        jobs=tuple(jobs),
        path=queue_path.resolve(),
    )


def _initial_queue_state(queue: LocalRunQueue) -> QueueState:
    return QueueState(
        queue_name=queue.name,
        queue_path=str(queue.path),
        continue_on_failure=queue.continue_on_failure,
        status="running",
        started_at=datetime.now(UTC).isoformat(),
        completed_at=None,
        current_job_index=None,
        jobs=[
            QueueJobState(
                index=index,
                config_path=str(job.config_path),
                label=job.label,
                notes=job.notes,
                stages=list(job.stages) if job.stages is not None else None,
            )
            for index, job in enumerate(queue.jobs)
        ],
    )


def run_queue(queue_path: Path) -> QueueState:
    queue = load_local_run_queue(queue_path.resolve())
    state = _initial_queue_state(queue)
    state.save(queue.state_path)

    failures = 0
    for index, job in enumerate(queue.jobs):
        job_state = state.jobs[index]
        job_state.status = "running"
        job_state.started_at = datetime.now(UTC).isoformat()
        state.current_job_index = index
        state.save(queue.state_path)

        try:
            config = RunConfig.from_yaml(job.config_path)
            run_stages(
                config=config,
                config_path=job.config_path,
                stages=job.stages or DEFAULT_FULL_RUN_STAGES,
            )
        except Exception as exc:
            failures += 1
            job_state.status = "failed"
            job_state.completed_at = datetime.now(UTC).isoformat()
            job_state.error = str(exc)
            state.status = "failed"
            state.current_job_index = None
            state.save(queue.state_path)
            if not queue.continue_on_failure:
                state.completed_at = datetime.now(UTC).isoformat()
                state.save(queue.state_path)
                return state
        else:
            job_state.status = "completed"
            job_state.completed_at = datetime.now(UTC).isoformat()
            state.save(queue.state_path)

    state.current_job_index = None
    state.completed_at = datetime.now(UTC).isoformat()
    state.status = "failed" if failures else "completed"
    state.save(queue.state_path)
    return state
