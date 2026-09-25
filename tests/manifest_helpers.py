from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from virtual_staining.data.manifest import (
    MANIFEST_SCHEMA_VERSION,
    DatasetManifest,
    ManifestMetadata,
    ManifestRecord,
    Split,
)

INPUT_MODALITIES = ("label_free",)
REFERENCE_MODALITY = "label_free"
TARGET_MODALITY = "stained"


def make_manifest_record(
    sample_id: str = "00000_00000",
    split: str = "train",
    *,
    ext: str = ".tif",
    input_paths: dict[str, Path] | None = None,
    target_path: Path | None = None,
    set_id: str = "P1",
    width: int = 256,
    height: int = 256,
    x: int | None = None,
    y: int | None = None,
    foreground_mask_path: Path | None = None,
) -> ManifestRecord:
    """Return one synthetic v3 manifest record."""
    if x is None or y is None:
        parts = sample_id.split("_", maxsplit=1)
        x = int(parts[0]) if x is None else x
        y = int(parts[1]) if y is None and len(parts) > 1 else (0 if y is None else y)
    typed_split = cast(Split, split)
    input_paths = input_paths or {
        "label_free": Path(f"splits/{typed_split}/{sample_id}_input__label_free{ext}")
    }
    target_path = target_path or Path(f"splits/{typed_split}/{sample_id}__target{ext}")
    return ManifestRecord(
        sample_id=sample_id,
        set_id=set_id,
        split=typed_split,
        input_paths=input_paths,
        target_path=target_path,
        x=x,
        y=y,
        width=width,
        height=height,
        foreground_mask_path=foreground_mask_path,
    )


def manifest_metadata(input_modalities: tuple[str, ...] = INPUT_MODALITIES) -> ManifestMetadata:
    return ManifestMetadata(
        MANIFEST_SCHEMA_VERSION, input_modalities, input_modalities[0], TARGET_MODALITY
    )


def make_manifest_records(
    n: int = 3,
    splits: Sequence[str] | None = None,
    *,
    ext: str = ".tif",
) -> tuple[ManifestRecord, ...]:
    if n <= 0:
        return ()
    if splits is None:
        splits_list = ["train"] * max(0, n - 1) + ["val"]
    else:
        split_cycle = list(splits)
        if not split_cycle:
            raise ValueError("splits must not be empty when provided")
        splits_list = [split_cycle[i % len(split_cycle)] for i in range(n)]
    return tuple(
        make_manifest_record(f"{i * 256:05}_00000", split, ext=ext, set_id=f"P{i}")
        for i, split in enumerate(splits_list)
    )


def write_manifest_csv(
    tmp_path: Path,
    records: Sequence[ManifestRecord],
    *,
    filename: str = "manifest.csv",
    metadata: ManifestMetadata | None = None,
) -> Path:
    manifest = DatasetManifest(
        records=tuple(records), dataset_root=tmp_path, metadata=metadata or manifest_metadata()
    )
    csv_path = tmp_path / "manifests" / filename
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(csv_path)
    return csv_path


def write_aligned_test_manifest(dataset_root: Path, sample_ids: list[str]) -> None:
    """Write an aligned test-split manifest of <id>_source/<id>_target PNGs for set P1."""
    records = tuple(
        make_manifest_record(
            sample_id,
            "test",
            input_paths={"label_free": Path(f"splits/test/{sample_id}_source.png")},
            ext=".png",
            target_path=Path(f"splits/test/{sample_id}_target.png"),
        )
        for sample_id in sample_ids
    )
    write_manifest_csv(dataset_root, records)
    with (dataset_root / "manifests" / "slide_sets.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "set_id",
                "split",
                "patient_id",
                "specimen_id",
                "status",
                "label_free__alignment_method",
                "target__alignment_method",
                "label_free__alignment_metadata",
                "target__alignment_metadata",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "set_id": "P1",
                "split": "test",
                "patient_id": "patient-1",
                "specimen_id": "specimen-1",
                "status": "processed",
                "label_free__alignment_method": "identity",
                "target__alignment_method": "identity",
                "label_free__alignment_metadata": "{}",
                "target__alignment_metadata": "{}",
            }
        )
    (dataset_root / "manifests" / "manifest_metadata.json").write_text(
        json.dumps(manifest_metadata().to_dict()), encoding="utf-8"
    )
