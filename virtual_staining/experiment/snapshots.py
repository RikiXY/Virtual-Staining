from __future__ import annotations

import json
import shutil
from pathlib import Path

from virtual_staining.config.run import RunConfig
from virtual_staining.experiment.environment import collect_environment
from virtual_staining.utils.hashing import sha256_file


def _save_input_config(src_yaml: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_yaml, dest)


def save_config_hash(hash_str: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(hash_str, encoding="utf-8")


def save_stage_config_snapshots(
    config: RunConfig,
    config_path: Path,
    *,
    input_dest: Path,
    resolved_dest: Path,
) -> str:
    _save_input_config(config_path, input_dest)
    resolved_dest.parent.mkdir(parents=True, exist_ok=True)
    resolved_dest.write_text(config.resolved_yaml(), encoding="utf-8")
    return sha256_file(resolved_dest)


def save_environment_snapshot(dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as handle:
        json.dump(collect_environment(), handle, indent=2, default=str)
