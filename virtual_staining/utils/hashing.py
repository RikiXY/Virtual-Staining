from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

_CHUNK_SIZE = 1024 * 1024


def sha256_bytes(data: bytes) -> str:
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def sha256_file_verified(path: Path) -> tuple[str, int]:
    """Hash ``path`` and fail if the file changed or was replaced while it was read.

    Returns ``(digest, size)``. The open descriptor is fstat-ed before and after reading and
    compared with a fresh ``stat`` of the path, so in-place writes, truncation, and atomic
    replacement during hashing are all rejected instead of yielding inconsistent bytes.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        before = os.fstat(handle.fileno())
        size = 0
        for chunk in iter(lambda: handle.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
            size += len(chunk)
        after = os.fstat(handle.fileno())
    current = os.stat(path)
    signatures = {
        (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns)
        for item in (before, after, current)
    }
    if len(signatures) != 1 or size != before.st_size:
        raise RuntimeError(f"File changed while it was being hashed: {path}")
    return f"sha256:{digest.hexdigest()}", size
