from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from virtual_staining.applications.pipeline import (
    DEFAULT_FULL_RUN_STAGES,
    VALID_STAGES,
    run_stages,
)
from virtual_staining.config.loader import load_yaml_mapping
from virtual_staining.config.run import RunConfig
from virtual_staining.config.validation import parse_bool_strict, reject_unknown_keys
from virtual_staining.experiment.snapshots import compute_payload_hash

_QUEUE_KEYS: frozenset[str] = frozenset({"name", "continue_on_failure", "jobs", "ablation"})
_QUEUE_JOB_KEYS: frozenset[str] = frozenset({"config_path", "label", "notes", "stages"})
_ABLATION_KEYS: frozenset[str] = frozenset({"fixed_fields", "variable_fields"})


@dataclass(frozen=True)
class QueueJob:
    config_path: Path
    label: str | None = None
    notes: str | None = None
    stages: tuple[str, ...] | None = None


@dataclass(frozen=True)
class AblationConfig:
    fixed_fields: tuple[str, ...] = ()
    variable_fields: tuple[str, ...] = ()


@dataclass(frozen=True)
class LocalRunQueue:
    name: str
    continue_on_failure: bool
    jobs: tuple[QueueJob, ...]
    path: Path
    ablation: AblationConfig | None = None

    @property
    def state_path(self) -> Path:
        return _resolve_queue_state_path(self.path, self.name)

    @property
    def ablation_summary_path(self) -> Path:
        return self.state_path.with_name(f"{self.name}.ablation.summary.json")


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
    ablation_summary_path: str | None = None
    jobs: list[QueueJobState] = field(default_factory=list)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")


class QueuePreflightError(ValueError):
    def __init__(self, job_index: int, config_path: Path, message: str) -> None:
        super().__init__(
            f"Queue preflight failed for job {job_index} config {config_path}: {message}"
        )
        self.job_index = job_index
        self.config_path = config_path


class QueueAblationError(ValueError):
    pass


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

    ablation = _parse_ablation_config(data.get("ablation"))

    return LocalRunQueue(
        name=raw_name.strip(),
        continue_on_failure=raw_continue,
        jobs=tuple(jobs),
        path=queue_path.resolve(),
        ablation=ablation,
    )


def _parse_ablation_config(raw: object) -> AblationConfig | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise TypeError("queue.ablation must be a YAML mapping")
    reject_unknown_keys(raw, _ABLATION_KEYS, "queue.ablation")
    fixed_fields = _parse_field_list(raw.get("fixed_fields", []), "queue.ablation.fixed_fields")
    variable_fields = _parse_field_list(
        raw.get("variable_fields", []),
        "queue.ablation.variable_fields",
    )
    if not variable_fields:
        raise ValueError("queue.ablation.variable_fields must be a non-empty list")
    overlap = sorted(set(fixed_fields) & set(variable_fields))
    if overlap:
        raise ValueError(
            "queue.ablation fields cannot be both fixed and variable: " + ", ".join(overlap)
        )
    return AblationConfig(fixed_fields=fixed_fields, variable_fields=variable_fields)


def _parse_field_list(raw: object, field_name: str) -> tuple[str, ...]:
    if not isinstance(raw, list):
        raise TypeError(f"{field_name} must be a YAML list")
    fields: list[str] = []
    for index, value in enumerate(raw):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name}[{index}] must be a non-empty string")
        fields.append(value.strip())
    duplicates = sorted({field for field in fields if fields.count(field) > 1})
    if duplicates:
        raise ValueError(f"{field_name} contains duplicate field(s): {', '.join(duplicates)}")
    return tuple(fields)


def _initial_queue_state(queue: LocalRunQueue) -> QueueState:
    return QueueState(
        queue_name=queue.name,
        queue_path=str(queue.path),
        continue_on_failure=queue.continue_on_failure,
        status="running",
        started_at=datetime.now(UTC).isoformat(),
        completed_at=None,
        current_job_index=None,
        ablation_summary_path=(
            str(queue.ablation_summary_path) if queue.ablation is not None else None
        ),
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


def _preflight_run_configs(queue: LocalRunQueue) -> tuple[RunConfig, ...]:
    configs: list[RunConfig] = []
    for index, job in enumerate(queue.jobs):
        try:
            if not job.config_path.is_file():
                raise FileNotFoundError(f"Config file not found: {job.config_path}")
            configs.append(RunConfig.from_yaml(job.config_path))
        except Exception as exc:
            raise QueuePreflightError(index, job.config_path, str(exc)) from exc
    return tuple(configs)


def _preflight_ablation(queue: LocalRunQueue, configs: tuple[RunConfig, ...]) -> None:
    if queue.ablation is None:
        return
    try:
        summary = _build_ablation_summary(queue, configs)
        _write_ablation_summary(queue.ablation_summary_path, summary)
    except Exception as exc:
        first_job = queue.jobs[0]
        raise QueuePreflightError(0, first_job.config_path, str(exc)) from exc


def _build_ablation_summary(
    queue: LocalRunQueue,
    configs: tuple[RunConfig, ...],
) -> dict[str, Any]:
    if queue.ablation is None:
        raise QueueAblationError("queue has no ablation configuration")

    resolved_configs = tuple(
        _canonicalize_resolved_config(config.to_yaml_dict()) for config in configs
    )
    flattened_configs = tuple(_flatten_config(config) for config in resolved_configs)
    _validate_ablation_differences(queue.ablation, flattened_configs)

    baseline = resolved_configs[0]
    fixed_values = {field: _get_dot_path(baseline, field) for field in queue.ablation.fixed_fields}
    jobs: list[dict[str, Any]] = []
    for index, (job, config, resolved_config) in enumerate(
        zip(queue.jobs, configs, resolved_configs, strict=True)
    ):
        jobs.append(
            {
                "index": index,
                "label": job.label,
                "notes": job.notes,
                "config_path": str(job.config_path),
                "run_name": config.project.run_name,
                "config_hash": compute_payload_hash(resolved_config),
                "variable_values": {
                    field: _get_dot_path(resolved_config, field)
                    for field in queue.ablation.variable_fields
                },
            }
        )

    return {
        "queue_name": queue.name,
        "queue_path": str(queue.path),
        "created_at": datetime.now(UTC).isoformat(),
        "fixed_fields": list(queue.ablation.fixed_fields),
        "variable_fields": list(queue.ablation.variable_fields),
        "fixed_values": fixed_values,
        "jobs": jobs,
    }


def _write_ablation_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


def _validate_ablation_differences(
    ablation: AblationConfig,
    flattened_configs: tuple[dict[str, Any], ...],
) -> None:
    paths = sorted({path for config in flattened_configs for path in config})
    violations: list[str] = []
    for path in paths:
        values = [_canonical_compare_value(config.get(path)) for config in flattened_configs]
        if len(set(values)) <= 1:
            continue
        if _path_is_declared_variable(path, ablation.variable_fields):
            continue
        violations.append(path)

    fixed_violations = [
        path
        for path in violations
        if any(
            path == field or path.startswith(f"{field}.") or path.startswith(f"{field}[")
            for field in ablation.fixed_fields
        )
    ]
    if fixed_violations:
        raise QueueAblationError("Ablation fixed field(s) differ: " + ", ".join(fixed_violations))
    if violations:
        raise QueueAblationError(
            "Ablation undeclared config difference(s): "
            + ", ".join(violations)
            + ". Add the intended path(s) to ablation.variable_fields."
        )


def _path_is_declared_variable(path: str, variable_fields: tuple[str, ...]) -> bool:
    return any(
        path == field or path.startswith(f"{field}.") or path.startswith(f"{field}[")
        for field in variable_fields
    )


def _canonicalize_resolved_config(value: Any) -> Any:
    if isinstance(value, dict):
        canonical = {key: _canonicalize_resolved_config(item) for key, item in value.items()}
        losses = canonical.get("losses")
        if isinstance(losses, dict):
            for role in ("generator", "discriminator"):
                terms = losses.get(role)
                if isinstance(terms, list):
                    losses[role] = sorted(
                        terms,
                        key=lambda term: (
                            term.get("name", "") if isinstance(term, dict) else "",
                            json.dumps(term, sort_keys=True),
                        ),
                    )
        return canonical
    if isinstance(value, list):
        return [_canonicalize_resolved_config(item) for item in value]
    return value


def _flatten_config(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        if not value:
            return {prefix: value} if prefix else {}
        flattened: dict[str, Any] = {}
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten_config(item, path))
        return flattened
    if isinstance(value, list):
        if not value:
            return {prefix: value}
        flattened = {}
        for index, item in enumerate(value):
            flattened.update(_flatten_config(item, f"{prefix}[{index}]"))
        return flattened
    return {prefix: value}


def _get_dot_path(value: Any, path: str) -> Any:
    current = value
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _canonical_compare_value(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _mark_preflight_failure(state: QueueState, error: QueuePreflightError) -> None:
    completed_at = datetime.now(UTC).isoformat()
    job_state = state.jobs[error.job_index]
    job_state.status = "failed"
    job_state.completed_at = completed_at
    job_state.error = str(error)
    state.status = "failed"
    state.completed_at = completed_at
    state.current_job_index = None


def run_queue(queue_path: Path) -> QueueState:
    queue = load_local_run_queue(queue_path.resolve())
    state = _initial_queue_state(queue)
    state.save(queue.state_path)
    try:
        configs = _preflight_run_configs(queue)
        _preflight_ablation(queue, configs)
    except QueuePreflightError as exc:
        _mark_preflight_failure(state, exc)
        state.save(queue.state_path)
        return state

    failures = 0
    for index, job in enumerate(queue.jobs):
        job_state = state.jobs[index]
        job_state.status = "running"
        job_state.started_at = datetime.now(UTC).isoformat()
        state.current_job_index = index
        state.save(queue.state_path)

        try:
            run_stages(job.config_path, job.stages or DEFAULT_FULL_RUN_STAGES)
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
