from __future__ import annotations

import csv
import datetime
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from virtual_staining.config.data import PreprocessingConfig
from virtual_staining.data.alignment import RegistrationBackend
from virtual_staining.data.layout import DatasetLayout
from virtual_staining.data.manifest import (
    MANIFEST_SCHEMA_VERSION,
    DatasetManifest,
    ManifestMetadata,
    ManifestRecord,
)
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
from virtual_staining.split_contract import (
    DATASET_SPLITS,
    DISCARDED_SPLIT,
    TEST_SPLIT,
    TRAIN_SPLIT,
    VAL_SPLIT,
    DatasetSplit,
)


@dataclass(frozen=True)
class DatasetBuildResult:
    """Committed sample counts per split; every committed sample has every named target."""

    train_count: int
    val_count: int
    test_count: int
    skipped_count: int
    output_root: Path
    reused: bool = False
    input_modalities: tuple[str, ...] = ()
    target_modalities: tuple[str, ...] = ()

    def save(self, path: Path, *, num_sets: int, num_sets_excluded: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "schema_version": MANIFEST_SCHEMA_VERSION,
                    "input_modalities": list(self.input_modalities),
                    "target_modalities": list(self.target_modalities),
                    "num_sets": num_sets,
                    "num_sets_excluded": num_sets_excluded,
                    "patches": {
                        TRAIN_SPLIT: self.train_count,
                        VAL_SPLIT: self.val_count,
                        TEST_SPLIT: self.test_count,
                        DISCARDED_SPLIT: self.skipped_count,
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
        names: list[tuple[str, ...]] = []
        for key in ("input_modalities", "target_modalities"):
            value = data.get(key)
            if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
                raise ValueError(f"Invalid dataset build metadata at {path}")
            names.append(tuple(value))
        if not isinstance(patches, dict):
            raise ValueError(f"Invalid dataset build metadata at {path}")
        try:
            counts = tuple(int(patches[name]) for name in (*DATASET_SPLITS, DISCARDED_SPLIT))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid dataset build metadata at {path}") from exc
        return cls(
            counts[0],
            counts[1],
            counts[2],
            counts[3],
            output_root,
            reused,
            names[0],
            names[1],
        )


def _ensure_clean_directory(directory: Path) -> None:
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)


class DatasetBuilder:
    """Coordinate slide-set builds and persist dataset manifests and provenance."""

    def __init__(
        self,
        config: PreprocessingConfig,
        slide_sets: tuple[SlideSet, ...],
        fingerprint_metadata: dict[str, Any] | None = None,
        *,
        registration_backend: RegistrationBackend | None = None,
    ) -> None:
        if not slide_sets:
            raise ValueError("DatasetBuilder requires at least one slide set")
        if fingerprint_metadata is not None and fingerprint_metadata.get("registration") != (
            registration_backend.metadata if registration_backend else None
        ):
            raise ValueError("Fingerprint registration identity does not match the backend")
        self.registration_backend = registration_backend
        self.config, self.slide_sets, self.fingerprint_metadata = (
            config,
            slide_sets,
            json.loads(json.dumps(fingerprint_metadata))
            if fingerprint_metadata is not None
            else None,
        )

    def _records(
        self, set_id: str, rows: tuple[dict[str, Any], ...], *, discarded: bool = False
    ) -> tuple[ManifestRecord, ...]:
        """Manifest records of one set; discarded rows keep their per-modality layout."""
        layout = DatasetLayout(Path())
        inputs_config = self.config.inputs
        records = []
        for row in rows:
            split = DISCARDED_SPLIT if discarded else cast(DatasetSplit, row["split"])
            base = (
                layout.discarded_patches_dir / set_id
                if discarded
                else layout.split_dir(cast(DatasetSplit, row["split"])) / set_id
            )

            def place(name: str, filename: str, base: Path = base) -> Path:
                return base / name / filename if discarded else base / filename

            records.append(
                ManifestRecord(
                    sample_id=row["sample_id"],
                    set_id=set_id,
                    split=split,
                    input_paths={
                        name: place(name, row["inputs"][name]) for name in inputs_config.modalities
                    },
                    target_paths={
                        name: place(name, row["targets"][name])
                        for name in inputs_config.target_modalities
                    },
                    foreground_mask_paths={
                        name: place(name, row["foreground_masks"][name])
                        if not discarded and row["foreground_masks"]
                        else None
                        for name in inputs_config.target_modalities
                    },
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
        # Withdraw the completion marker and manifests first, so a build that fails part
        # way never leaves an earlier, now inconsistent dataset looking consumable.
        for path in (
            layout.dataset_build_path,
            layout.manifest_path,
            layout.manifest_metadata_path,
            layout.discarded_manifest_path,
        ):
            path.unlink(missing_ok=True)
        for path in (layout.split_dir(name) for name in DATASET_SPLITS):
            _ensure_clean_directory(path)
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
                self.config,
                slide_set,
                assignments.get(slide_set.set_id),
                registration_backend=self.registration_backend,
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
            name: sum(record.split == name for record in valid_records) for name in DATASET_SPLITS
        }
        result = DatasetBuildResult(
            counts[TRAIN_SPLIT],
            counts[VAL_SPLIT],
            counts[TEST_SPLIT],
            len(discarded_records),
            root,
            input_modalities=self.config.inputs.modalities,
            target_modalities=self.config.inputs.target_modalities,
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
            schema_version=MANIFEST_SCHEMA_VERSION,
            input_modalities=self.config.inputs.modalities,
            target_modalities=self.config.inputs.target_modalities,
            reference_modality=self.config.inputs.reference,
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
                for name in DATASET_SPLITS
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
        names = (*self.config.inputs.modalities, *self.config.inputs.target_modalities)
        fields = ["set_id", "split", "patient_id", "specimen_id", "status"]
        fields.extend(f"{name}__alignment_method" for name in names)
        fields.extend(f"{name}__alignment_metadata" for name in names)
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
        assignments: dict[str, DatasetSplit],
        valid_records: list[ManifestRecord],
    ) -> None:
        assignments_out: dict[str, DatasetSplit] = (
            {record.sample_id: cast(DatasetSplit, record.split) for record in valid_records}
            if self.config.split.unit == "patch"
            else {
                group_id_for_set(item, self.config.split.unit): assignments[item.set_id]
                for item in self.slide_sets
            }
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
            registration_backend=self.registration_backend,
        )
        save_dataset_fingerprint(fingerprint, layout.dataset_fingerprint_path)
