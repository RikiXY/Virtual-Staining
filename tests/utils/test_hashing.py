from __future__ import annotations

import os
from pathlib import Path

import pytest

from virtual_staining.utils import hashing
from virtual_staining.utils.hashing import (
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
    sha256_file_verified,
    sha256_json,
)


def test_sha256_bytes_and_file_use_prefixed_digest(tmp_path: Path) -> None:
    payload = b"hello"
    path = tmp_path / "payload.bin"
    path.write_bytes(payload)

    expected = "sha256:2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    assert sha256_bytes(payload) == expected
    assert sha256_file(path) == expected


def test_canonical_json_preserves_unicode_and_ignores_mapping_order() -> None:
    left = {"beta": "é", "alpha": {"greek": "β"}}
    right = {"alpha": {"greek": "β"}, "beta": "é"}

    assert canonical_json_bytes(left) == '{"alpha":{"greek":"β"},"beta":"é"}'.encode()
    assert sha256_json(left) == sha256_json(right)
    assert sha256_json(left) == sha256_bytes(canonical_json_bytes(left))


def test_verified_hash_rejects_a_file_mutated_while_hashing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "payload.bin"
    path.write_bytes(b"hello")
    assert sha256_file_verified(path) == (sha256_bytes(b"hello"), 5)

    real_fstat = os.fstat
    calls = {"count": 0}

    def mutate_between_reads(fd: int) -> os.stat_result:
        calls["count"] += 1
        if calls["count"] == 2:
            path.write_bytes(b"hello, changed")
        return real_fstat(fd)

    monkeypatch.setattr(hashing.os, "fstat", mutate_between_reads)
    with pytest.raises(RuntimeError, match="changed while it was being hashed"):
        sha256_file_verified(path)
