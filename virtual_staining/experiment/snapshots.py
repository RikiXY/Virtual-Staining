from __future__ import annotations

import hashlib
import shutil
from pathlib import Path


def save_input_config(src_yaml: Path, dest: Path) -> None:
    """Copy the user-supplied YAML into config/input.yaml."""
    shutil.copy2(src_yaml, dest)


def compute_config_hash(yaml_path: Path) -> str:
    """Return sha256:<hex> of the YAML file content."""
    digest = hashlib.sha256(yaml_path.read_bytes()).hexdigest()
    return f"sha256:{digest}"


def save_config_hash(hash_str: str, dest: Path) -> None:
    dest.write_text(hash_str, encoding="utf-8")
