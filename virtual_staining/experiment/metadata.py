from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


@dataclass
class RunMetadata:
    run_name: str
    status: str
    started_at: str
    completed_at: str | None = None
    git_commit: str | None = None
    git_dirty: bool | None = None
    config_hash: str | None = None
    seed: int | None = None
    device: str | None = None
    cuda_device_name: str | None = None
    entrypoint: str | None = None
    package_version: str | None = None

    @classmethod
    def create(cls, run_name: str, entrypoint: str | None = None, **kwargs) -> RunMetadata:
        git_commit, git_dirty = _capture_git_state()
        package_version = _package_version()
        return cls(
            run_name=run_name,
            status="running",
            started_at=datetime.now(UTC).isoformat(),
            entrypoint=entrypoint,
            git_commit=git_commit,
            git_dirty=git_dirty,
            package_version=package_version,
            **kwargs,
        )

    def mark_completed(self) -> None:
        self.status = "completed"
        self.completed_at = datetime.now(UTC).isoformat()

    def mark_failed(self) -> None:
        self.status = "failed"
        self.completed_at = datetime.now(UTC).isoformat()

    def save(self, path: Path) -> None:
        with path.open("w", encoding="utf-8") as handle:
            json.dump(asdict(self), handle, indent=2, default=str)


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
