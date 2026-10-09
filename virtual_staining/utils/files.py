from __future__ import annotations

import ctypes
import os
import sys
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


def publish_directory_no_replace(source: Path, destination: Path) -> None:
    """Atomically rename a staged directory without replacing even an empty destination."""
    # os.rename can overwrite empty directories on POSIX; use the exclusive native flags.
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin":
        rename = libc.renamex_np
        rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        result = rename(os.fsencode(source), os.fsencode(destination), 4)  # RENAME_EXCL
    elif sys.platform.startswith("linux"):
        rename = libc.renameat2
        rename.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename.restype = ctypes.c_int
        result = rename(-100, os.fsencode(source), -100, os.fsencode(destination), 1)
    else:
        raise OSError("Exclusive directory publication requires Linux or macOS")
    if result != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(destination))
