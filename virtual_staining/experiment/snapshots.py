from __future__ import annotations

import json
import shutil
from pathlib import Path

import yaml

from virtual_staining.config.run import RunConfig
from virtual_staining.experiment.environment import collect_environment
from virtual_staining.utils.hashing import sha256_file


def save_input_config(src_yaml: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_yaml, dest)


def save_resolved_config(config_dict: dict[str, object], dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            config_dict,
            handle,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=True,
        )


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
    save_input_config(config_path, input_dest)
    save_resolved_config(config.to_dict(), resolved_dest)
    return sha256_file(resolved_dest)


def save_environment_snapshot(dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as handle:
        json.dump(collect_environment(), handle, indent=2, default=str)
