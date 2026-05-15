from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import cast

from virtual_staining.data.manifest import DatasetManifest, ManifestRecord, Split


def make_manifest_record(
    sample_id: str = "00000_00000",
    split: str = "train",
    *,
    ext: str = ".tif",
    input_modality: str = "label_free",
    target_modality: str = "stained",
    width: int = 256,
    height: int = 256,
    x: int | None = None,
    y: int | None = None,
    input_path: Path | None = None,
    target_path: Path | None = None,
) -> ManifestRecord:
    """Return one synthetic manifest record using the canonical split path layout."""
    if x is None or y is None:
        x_str, y_str = sample_id.split("_", maxsplit=1)
        x = int(x_str) if x is None else x
        y = int(y_str) if y is None else y

    typed_split = cast(Split, split)
    input_path = input_path or Path(f"splits/{typed_split}/{sample_id}_source{ext}")
    target_path = target_path or Path(f"splits/{typed_split}/{sample_id}_target{ext}")
    return ManifestRecord(
        sample_id=sample_id,
        split=typed_split,
        input_path=input_path,
        target_path=target_path,
        input_modality=input_modality,
        target_modality=target_modality,
        x=x,
        y=y,
        width=width,
        height=height,
    )


def make_manifest_records(
    n: int = 3,
    splits: Sequence[str] | None = None,
    *,
    ext: str = ".tif",
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
        records.append(make_manifest_record(sample_id, split, ext=ext))
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
