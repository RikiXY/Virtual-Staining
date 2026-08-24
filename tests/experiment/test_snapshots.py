from __future__ import annotations

from pathlib import Path

from virtual_staining.data.slide_sets import SlideAsset, SlideSet
from virtual_staining.experiment.snapshots import (
    build_dataset_fingerprint_metadata,
    compute_config_hash,
    resolve_prepare_snapshot_paths,
    save_resolved_config,
)


def _slide_set(root: Path) -> tuple[SlideSet, ...]:
    (root / "source").write_bytes(b"source")
    (root / "target").write_bytes(b"target")
    return (
        SlideSet(
            "S1", (SlideAsset("LF", Path("source")),), SlideAsset("target", Path("target")), "LF"
        ),
    )


def test_prepare_snapshot_paths_and_config_hash(tmp_path: Path) -> None:
    paths = resolve_prepare_snapshot_paths(tmp_path)
    save_resolved_config({"b": 2, "a": 1}, paths.resolved_config)
    assert paths.resolved_config.exists()
    assert compute_config_hash(paths.resolved_config).startswith("sha256:")


def test_fingerprint_changes_when_set_content_changes(tmp_path: Path) -> None:
    sets = _slide_set(tmp_path)
    kwargs = {
        "dataset_root": tmp_path,
        "preprocessing_config": {"inputs": {"modalities": ["LF"]}},
        "slide_sets": sets,
    }
    first = build_dataset_fingerprint_metadata(**kwargs)
    (tmp_path / "target").write_bytes(b"changed")
    second = build_dataset_fingerprint_metadata(**kwargs)
    assert first["fingerprint"] != second["fingerprint"]
