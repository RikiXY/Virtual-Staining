from __future__ import annotations

import csv
import datetime
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from virtual_staining.config.data import PreprocessingConfig
from virtual_staining.data.layout import DatasetLayout
from virtual_staining.data.manifest import (
    MANIFEST_SCHEMA_VERSION,
    DatasetManifest,
    ManifestMetadata,
    ManifestRecord,
    Split,
)
from virtual_staining.data.preprocessing import ensure_clean_directory
from virtual_staining.data.provenance import (
    build_dataset_fingerprint_metadata,
    save_dataset_fingerprint,
)
from virtual_staining.data.slide_set_processor import SlideSetProcessor
from virtual_staining.data.slide_sets import SlideSet
from virtual_staining.data.splitting import (
    assign_group_splits,
    group_id_for_set,
    write_split_assignment,
)


@dataclass(frozen=True)
class DatasetBuildResult:
    train_count: int
    val_count: int
    test_count: int
    skipped_count: int
    output_root: Path
    reused: bool = False

    def save(self, path: Path, *, num_sets: int, num_sets_excluded: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "schema_version": MANIFEST_SCHEMA_VERSION,
                    "num_sets": num_sets,
                    "num_sets_excluded": num_sets_excluded,
                    "patches": {
                        "train": self.train_count,
                        "val": self.val_count,
                        "test": self.test_count,
                        "discarded": self.skipped_count,
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path, *, output_root: Path, reused: bool = False) -> DatasetBuildResult:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid dataset build metadata at {path}") from exc
        if not isinstance(data, dict) or data.get("schema_version") != MANIFEST_SCHEMA_VERSION:
            raise ValueError(f"Invalid dataset build metadata at {path}")
        patches = data.get("patches")
        if not isinstance(patches, dict):
            raise ValueError(f"Invalid dataset build metadata at {path}")
        try:
            counts = tuple(int(patches[name]) for name in ("train", "val", "test", "discarded"))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid dataset build metadata at {path}") from exc
        return cls(
            counts[0],
            counts[1],
            counts[2],
            counts[3],
            output_root,
            reused,
        )


class DatasetBuilder:
    """Coordinate slide-set builds and persist dataset manifests and provenance."""

    def __init__(
        self,
        config: PreprocessingConfig,
        slide_sets: tuple[SlideSet, ...],
        fingerprint_metadata: dict[str, Any] | None = None,
    ) -> None:
        if not slide_sets:
            raise ValueError("DatasetBuilder requires at least one slide set")
        self.config, self.slide_sets, self.fingerprint_metadata = (
            config,
            slide_sets,
            fingerprint_metadata,
        )

    def _records(
        self, set_id: str, rows: tuple[dict[str, Any], ...], *, discarded: bool = False
    ) -> tuple[ManifestRecord, ...]:
        layout = DatasetLayout(Path())
        records = []
        for row in rows:
            split = "discarded" if discarded else row["split"]
            root = layout.discarded_patches_dir if discarded else layout.split_dir(row["split"])
            if discarded:
                inputs = {
                    name: root / set_id / name / filename
                    for name, filename in row["inputs"].items()
                    if name in self.config.inputs.modalities
                }
                target = root / set_id / "target" / row["target"]
                mask = None
            else:
                base = root / set_id
                inputs = {
                    name: base / filename
                    for name, filename in row["inputs"].items()
                    if name in self.config.inputs.modalities
                }
                target = base / row["target"]
                mask = (
                    base / row["foreground_mask"]
                    if row.get("foreground_mask") and self.config.masks.save_patch_masks
                    else None
                )
            records.append(
                ManifestRecord(
                    sample_id=row["sample_id"],
                    set_id=set_id,
                    split=split,
                    input_paths=inputs,
                    target_path=target,
                    foreground_mask_path=mask,
                    x=row["x"],
                    y=row["y"],
                    width=self.config.patching.patch_size[0],
                    height=self.config.patching.patch_size[1],
                )
            )
        return tuple(records)

    def run_all(self) -> DatasetBuildResult:
        layout = DatasetLayout(self.config.dataset_root)
        root = layout.root
        for path in (layout.split_dir(name) for name in ("train", "val", "test")):
            ensure_clean_directory(path)
        layout.manifests_dir.mkdir(parents=True, exist_ok=True)
        layout.metadata_dir.mkdir(parents=True, exist_ok=True)
        assignments = assign_group_splits(
            self.slide_sets,
            unit=self.config.split.unit,
            ratios=(self.config.split.train, self.config.split.val, self.config.split.test),
            seed=self.config.split.seed,
            assignment_file=self.config.split.assignment_file,
            dataset_root=root,
        )
        valid_records: list[ManifestRecord] = []
        discarded_records: list[ManifestRecord] = []
        set_rows: list[dict[str, Any]] = []
        excluded: list[dict[str, str]] = []
        for slide_set in sorted(self.slide_sets, key=lambda item: item.set_id):
            set_result = SlideSetProcessor(
                self.config, slide_set, assignments.get(slide_set.set_id)
            ).process()
            valid_records.extend(self._records(set_result.set_id, set_result.valid_rows))
            discarded_records.extend(
                self._records(set_result.set_id, set_result.discarded_rows, discarded=True)
            )
            set_rows.append(
                {
                    "set_id": set_result.set_id,
                    "split": set_result.split or "",
                    "patient_id": slide_set.patient_id or "",
                    "specimen_id": slide_set.specimen_id or "",
                    "status": "excluded" if set_result.error is not None else "processed",
                    **set_result.metadata,
                }
            )
            if set_result.error is not None:
                excluded.append(
                    {
                        "set_id": set_result.set_id,
                        "split": set_result.split or "",
                        "error": set_result.error,
                    }
                )
        self._write_manifests(layout, valid_records, discarded_records)
        self._write_set_metadata(layout, set_rows, excluded)
        self._write_provenance(layout, assignments, valid_records)
        counts = {
            name: sum(record.split == name for record in valid_records)
            for name in ("train", "val", "test")
        }
        result = DatasetBuildResult(
            counts["train"], counts["val"], counts["test"], len(discarded_records), root
        )
        result.save(
            layout.dataset_build_path,
            num_sets=len(self.slide_sets),
            num_sets_excluded=len(excluded),
        )
        return result

    def _write_manifests(
        self,
        layout: DatasetLayout,
        valid_records: list[ManifestRecord],
        discarded_records: list[ManifestRecord],
    ) -> None:
        metadata = ManifestMetadata(
            MANIFEST_SCHEMA_VERSION,
            cast(tuple[str, ...], self.config.inputs.modalities),
            self.config.inputs.reference,
            self.config.inputs.target_modality,
        )
        manifest = DatasetManifest(tuple(valid_records), layout.root, metadata)
        manifest.validate()
        manifest.to_csv(layout.manifest_path)
        DatasetManifest(tuple(discarded_records), layout.root, metadata).to_csv(
            layout.discarded_manifest_path
        )
        manifest_meta = {
            **metadata.to_dict(),
            "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
            "record_count": len(valid_records),
            "splits": {
                name: sum(record.split == name for record in valid_records)
                for name in ("train", "val", "test")
            },
        }
        layout.manifest_metadata_path.write_text(
            json.dumps(manifest_meta, indent=2), encoding="utf-8"
        )

    def _write_set_metadata(
        self,
        layout: DatasetLayout,
        set_rows: list[dict[str, Any]],
        excluded: list[dict[str, str]],
    ) -> None:
        fields = ["set_id", "split", "patient_id", "specimen_id", "status"]
        fields.extend(
            f"{name}__alignment_method" for name in (*self.config.inputs.modalities, "target")
        )
        fields.extend(
            f"{name}__alignment_metadata" for name in (*self.config.inputs.modalities, "target")
        )
        with layout.slide_sets_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(set_rows)
        with (layout.metadata_dir / "excluded_sets.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=["set_id", "split", "error"])
            writer.writeheader()
            writer.writerows(excluded)

    def _write_provenance(
        self,
        layout: DatasetLayout,
        assignments: dict[str, Split],
        valid_records: list[ManifestRecord],
    ) -> None:
        assignments_out: dict[str, Split] = cast(
            dict[str, Split],
            (
                {record.sample_id: record.split for record in valid_records}
                if self.config.split.unit == "patch"
                else {
                    group_id_for_set(item, self.config.split.unit): assignments[item.set_id]
                    for item in self.slide_sets
                }
            ),
        )
        write_split_assignment(
            layout.split_assignment_path,
            unit=self.config.split.unit,
            assignments=assignments_out,
        )
        fingerprint = self.fingerprint_metadata or build_dataset_fingerprint_metadata(
            dataset_root=layout.root,
            preprocessing_config=self.config.to_dict(),
            slide_sets=self.slide_sets,
        )
        save_dataset_fingerprint(fingerprint, layout.dataset_fingerprint_path)
