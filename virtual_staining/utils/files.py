from __future__ import annotations

import os
import uuid
from collections.abc import Callable
from pathlib import Path


def publish_file_no_replace(
    data: bytes, destination: Path, *, verify: Callable[[Path], object] | None = None
) -> Path:
    """Publish ``data`` at ``destination``; ``FileExistsError`` if it exists.

    The bytes are written to a sibling temporary file, which ``verify`` may inspect
    (raising to abort), and are then hard-linked into place. This never replaces an
    existing path, never exposes a partial or unverified file, and removes only the
    temporary file it created.
    """
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if verify is not None:
            verify(temporary)
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination
