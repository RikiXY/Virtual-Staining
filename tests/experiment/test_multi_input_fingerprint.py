from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

from virtual_staining.data.pairs import SlidePair
from virtual_staining.experiment.snapshots import build_dataset_fingerprint_metadata


def test_canonical_inventory_hash_ignores_row_order(tmp_path: Path) -> None:
    for name, content in (("s1", b"a"), ("t1", b"b"), ("s2", b"c"), ("t2", b"d")):
        (tmp_path / name).write_bytes(content)
    pairs = (
        SlidePair("P1", Path("s1"), Path("t1"), patient_id="PT1"),
        SlidePair("P2", Path("s2"), Path("t2"), patient_id="PT2"),
    )
    kwargs = {
        "dataset_root": tmp_path,
        "preprocessing_config": {"split": {"unit": "patient"}},
    }
    first = build_dataset_fingerprint_metadata(**kwargs, pairs=pairs)
    second = build_dataset_fingerprint_metadata(**kwargs, pairs=tuple(reversed(pairs)))
    assert first["fingerprint"] == second["fingerprint"]
    assert first["canonical_inventory_sha256"] == second["canonical_inventory_sha256"]


def test_hash_cache_and_force_verification(tmp_path: Path) -> None:
    (tmp_path / "source").write_bytes(b"source")
    (tmp_path / "target").write_bytes(b"target")
    pair = (SlidePair("P1", Path("source"), Path("target")),)
    cache = tmp_path / "metadata/input_hashes.json"
    kwargs = {
        "dataset_root": tmp_path,
        "preprocessing_config": {},
        "pairs": pair,
        "hash_cache_path": cache,
    }
    build_dataset_fingerprint_metadata(**kwargs)
    with patch(
        "virtual_staining.experiment.snapshots.compute_file_sha256",
        side_effect=AssertionError("cache should be reused"),
    ):
        build_dataset_fingerprint_metadata(**kwargs)

    with patch(
        "virtual_staining.experiment.snapshots.compute_file_sha256",
        wraps=lambda path: f"sha256:{Path(path).name}",
    ) as digest:
        build_dataset_fingerprint_metadata(**kwargs, force_hash_verification=True)
    assert digest.call_count == 2

    stat = (tmp_path / "source").stat()
    os.utime(tmp_path / "source", ns=(stat.st_atime_ns, stat.st_mtime_ns + 1))
    with patch(
        "virtual_staining.experiment.snapshots.compute_file_sha256",
        wraps=lambda path: f"sha256:{Path(path).name}",
    ) as digest:
        build_dataset_fingerprint_metadata(**kwargs)
    assert digest.call_count == 1
