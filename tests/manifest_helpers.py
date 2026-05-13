from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import cast

from virtual_staining.data.manifest import DatasetManifest, ManifestRecord, Split


def make_manifest_records(
    n: int = 3,
    splits: Sequence[str] | None = None,
) -> tuple[ManifestRecord, ...]:
    """Return synthetic manifest records with deterministic sample IDs and paths."""
    if n <= 0:
        return ()

    if splits is None:
        splits_list = ["train"] * max(0, n - 1) + ["val"]
    else:
        split_cycle = list(splits)
        if not split_cycle:
            raise ValueError("splits must not be empty when provided")
        splits_list = [split_cycle[i % len(split_cycle)] for i in range(n)]

    records: list[ManifestRecord] = []
    for i, split in enumerate(splits_list):
        x = i * 256
        sample_id = f"{x:05}_00000"
        typed_split = cast(Split, split)
        records.append(
            ManifestRecord(
                sample_id=sample_id,
                split=typed_split,
                input_path=Path(f"dataset_{typed_split}/{sample_id}_source.tif"),
                target_path=Path(f"dataset_{typed_split}/{sample_id}_target.tif"),
                input_modality="label_free",
                target_modality="stained",
                x=x,
                y=0,
                width=256,
                height=256,
            )
        )
    return tuple(records)


def write_manifest_csv(
    tmp_path: Path,
    records: Sequence[ManifestRecord],
    *,
    filename: str = "manifest.csv",
) -> Path:
    """Write records to tmp_path/manifests/<filename> and return the CSV path."""
    manifest = DatasetManifest(records=tuple(records), dataset_root=tmp_path)
    csv_path = tmp_path / "manifests" / filename
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(csv_path)
    return csv_path
