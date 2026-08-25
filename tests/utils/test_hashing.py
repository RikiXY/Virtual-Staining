from __future__ import annotations

from pathlib import Path

from virtual_staining.utils.hashing import (
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
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
