from __future__ import annotations

from pathlib import Path

from virtual_staining.data.provenance import (
    build_dataset_fingerprint_metadata,
    resolve_prepare_snapshot_paths,
)
from virtual_staining.data.slide_sets import SlideAsset, SlideSet
from virtual_staining.experiment.snapshots import compute_config_hash, save_resolved_config


def test_prepare_snapshot_paths_and_config_hash(tmp_path: Path) -> None:
    paths = resolve_prepare_snapshot_paths(tmp_path)
    save_resolved_config({"b": 2, "a": 1}, paths.resolved_config)
    assert paths.resolved_config.exists()
    assert compute_config_hash(paths.resolved_config).startswith("sha256:")


def _sets(root: Path) -> tuple[SlideSet, ...]:
    for name in ("lf", "af", "third", "target", "lf-mask"):
        (root / name).write_bytes(name.encode())
    return (
        SlideSet(
            "S1",
            (
                SlideAsset("LF", Path("lf"), already_aligned=True, mask_path=Path("lf-mask")),
                SlideAsset("AF", Path("af"), already_aligned=False),
                SlideAsset("TH", Path("third"), already_aligned=True),
            ),
            SlideAsset("target", Path("target"), already_aligned=True),
            "LF",
            patient_id="P1",
            specimen_id="SP1",
        ),
    )


def _fingerprint(root: Path, sets: tuple[SlideSet, ...]) -> str:
    return build_dataset_fingerprint_metadata(
        dataset_root=root,
        preprocessing_config={"inputs": {"modalities": ["LF", "AF", "TH"]}},
        slide_sets=sets,
    )["fingerprint"]


def test_fingerprint_is_row_order_independent_and_schema_v3(tmp_path: Path) -> None:
    sets = _sets(tmp_path)
    first = build_dataset_fingerprint_metadata(
        dataset_root=tmp_path,
        preprocessing_config={"inputs": {"modalities": ["LF", "AF", "TH"]}},
        slide_sets=sets,
    )
    reordered = _fingerprint(tmp_path, tuple(reversed(sets)))
    assert first["schema_version"] == "3.0"
    assert first["fingerprint"] == reordered


def test_each_asset_and_mask_changes_fingerprint(tmp_path: Path) -> None:
    sets = _sets(tmp_path)
    baseline = _fingerprint(tmp_path, sets)
    changed = list(sets[0].inputs)
    (tmp_path / "af").write_bytes(b"changed")
    assert _fingerprint(tmp_path, sets) != baseline
    (tmp_path / "af").write_bytes(b"af")
    changed[2] = SlideAsset("TH", Path("third"), already_aligned=False)
    assert (
        _fingerprint(tmp_path, (SlideSet("S1", tuple(changed), sets[0].target, "LF"),)) != baseline
    )
