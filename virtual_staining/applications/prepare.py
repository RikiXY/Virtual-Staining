from __future__ import annotations

import json
import logging
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

from virtual_staining.config.data import PreprocessingConfig
from virtual_staining.config.run import RunConfig
from virtual_staining.data.alignment import RegistrationBackend
from virtual_staining.data.builder import DatasetBuilder, DatasetBuildResult, build_unpaired_dataset
from virtual_staining.data.consumption import AssetRow, DataSnapshot, build_snapshot, write_snapshot
from virtual_staining.data.layout import DatasetLayout
from virtual_staining.data.provenance import (
    build_dataset_fingerprint_metadata,
    save_dataset_fingerprint,
    unpaired_fingerprint,
)
from virtual_staining.data.slide_sets import (
    RegistrationEvidence,
    SlideSet,
    resolve_registration_evidence,
    resolve_slide_sets,
)
from virtual_staining.data.unpaired_inventory import load_unpaired_inventory, unpaired_assignments
from virtual_staining.experiment.snapshots import (
    save_config_hash,
    save_environment_snapshot,
    save_stage_config_snapshots,
)
from virtual_staining.split_contract import DATASET_SPLITS
from virtual_staining.utils.files import publish_directory_no_replace
from virtual_staining.utils.hashing import sha256_file
from virtual_staining.utils.image_io import detect_openslide_format

logger = logging.getLogger(__name__)

PREPARE_ADAPTER = "slide_set_inventory/1"


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def source_snapshot(config: RunConfig, slide_sets: tuple[SlideSet, ...]) -> DataSnapshot:
    """Snapshot the raw inputs, targets, and supplied masks selected for preparation.

    Splits are assigned by preparation itself, so rows carry no split and no cross-split
    group claim is made here; the prepared stages validate their own split partitions.
    """
    assert config.preprocessing is not None
    preprocessing = config.preprocessing
    rows: list[AssetRow] = []
    for item in slide_sets:
        groups: dict[str, Any] = {
            "set_id": item.set_id,
            "specimen_id": item.specimen_id or "",
            "patient_id": item.patient_id or "",
        }
        for role, asset in (
            *(("input", asset) for asset in item.inputs),
            *(("target", asset) for asset in item.targets),
        ):
            rows.append(
                AssetRow(
                    root="dataset",
                    locator=asset.path.as_posix(),
                    role=role,
                    domain=asset.modality,
                    **groups,
                )
            )
            if asset.mask_path is not None:
                rows.append(
                    AssetRow(
                        root="dataset",
                        locator=asset.mask_path.as_posix(),
                        role="mask",
                        domain=asset.modality,
                        **groups,
                    )
                )
    inventory = preprocessing.inputs.inventory
    inventory_path = (
        inventory if inventory.is_absolute() else preprocessing.dataset_root / inventory
    )
    return build_snapshot(
        rows,
        kind="consumed",
        adapter=PREPARE_ADAPTER,
        roots={"dataset": preprocessing.dataset_root},
        hash_policy=config.data.hash_policy,
        selection={
            "modalities": list(preprocessing.inputs.modalities),
            "reference": preprocessing.inputs.reference,
            "target_modalities": list(preprocessing.inputs.target_modalities),
            "split_unit": preprocessing.split.unit,
        },
        sources={"inventory": str(inventory), "inventory_sha256": sha256_file(inventory_path)},
    )


def _build_current_fingerprint(
    config: RunConfig,
    slide_sets: tuple[SlideSet, ...],
    snapshot: DataSnapshot,
    registration_backend: RegistrationBackend | None = None,
    registration_evidence: RegistrationEvidence | None = None,
) -> dict[str, Any]:
    assert config.preprocessing is not None
    layout = DatasetLayout(config.preprocessing.dataset_root)
    root = layout.root.resolve()
    verified = {
        str((root / row.locator).resolve()): row.sha256
        for row in snapshot.rows
        if row.sha256 is not None
    }
    fingerprint = build_dataset_fingerprint_metadata(
        dataset_root=layout.root,
        preprocessing_config=config.preprocessing.to_dict(),
        slide_sets=slide_sets,
        inventory_path=layout.root / config.preprocessing.inputs.inventory,
        hash_cache_path=layout.input_hashes_path,
        force_hash_verification=config.preprocessing.inputs.hash_verification == "always",
        verified_hashes=verified,
        registration_backend=registration_backend,
        registration_evidence=registration_evidence,
    )
    # Cross-reference only; the fingerprint digest itself stays preparation lineage.
    fingerprint["source_snapshot_id"] = snapshot.snapshot_id
    return fingerprint


def _dataset_outputs_are_complete(dataset_root: Path) -> bool:
    layout = DatasetLayout(dataset_root)
    required = (
        layout.manifest_path,
        layout.discarded_manifest_path,
        layout.slide_sets_path,
        layout.manifest_metadata_path,
        layout.dataset_build_path,
        layout.dataset_fingerprint_path,
        layout.split_assignment_path,
    )
    return all(path.is_file() for path in required) and all(
        layout.split_dir(name).is_dir() for name in DATASET_SPLITS
    )


def _build_reused_result(dataset_root: Path) -> DatasetBuildResult:
    layout = DatasetLayout(dataset_root)
    return DatasetBuildResult.load(layout.dataset_build_path, output_root=layout.root, reused=True)


def _log_prepare_summary(
    preprocessing: PreprocessingConfig, slide_sets: tuple[SlideSet, ...], *, reused: bool
) -> None:
    logger.info(
        "Prepare summary | dataset=%s | sets=%d | action=%s",
        preprocessing.dataset_root,
        len(slide_sets),
        "reuse" if reused else "build",
    )
    for item in slide_sets:
        logger.info(
            "Set %s | inputs=%s | targets=%s | reference=%s",
            item.set_id,
            ",".join(asset.modality for asset in item.inputs),
            ",".join(asset.modality for asset in item.targets),
            item.reference_modality,
        )


def _warn_image_backend(config: RunConfig, slide_sets: tuple[SlideSet, ...]) -> None:
    preprocessing = config.preprocessing
    if preprocessing is None or not preprocessing.io.tiled:
        return
    root = preprocessing.dataset_root
    paths = tuple(
        sorted(
            {
                root / asset.path
                for item in slide_sets
                for asset in item.assets
                if (root / asset.path).is_file()
            }
        )
    )
    if not paths:
        return
    incompatible = tuple(path for path in paths if detect_openslide_format(path) is None)
    if incompatible and preprocessing.io.backend == "openslide":
        logger.warning(
            "Configured slides are not OpenSlide-compatible; tiled preparation "
            "cannot use the requested backend."
        )


def prepare(
    config: RunConfig,
    config_path: Path,
    *,
    registration_backend: RegistrationBackend | None = None,
    registration_evidence: RegistrationEvidence | None = None,
) -> DatasetBuildResult:
    config.validate_stages(("prepare",))
    assert config.preprocessing is not None
    if config.data.pairing == "unpaired":
        if registration_backend is not None or registration_evidence is not None:
            raise ValueError("Registration is unsupported for independent unpaired preparation")
        return _prepare_unpaired(config, config_path)
    root = config.preprocessing.dataset_root
    layout = DatasetLayout(root)
    slide_sets = resolve_slide_sets(config.preprocessing)
    registration_evidence = resolve_registration_evidence(slide_sets, registration_evidence)
    config_hash = save_stage_config_snapshots(
        config,
        config_path,
        input_dest=layout.input_config_path,
        resolved_dest=layout.resolved_config_path,
    )
    save_config_hash(config_hash, layout.config_hash_path)
    save_environment_snapshot(layout.environment_path)

    # Freeze and persist the selected raw assets before any reuse decision or build.
    snapshot = source_snapshot(config, slide_sets)
    write_snapshot(snapshot, layout.source_snapshot)
    fingerprint = _build_current_fingerprint(
        config, slide_sets, snapshot, registration_backend, registration_evidence
    )
    stored = _load_json(layout.dataset_fingerprint_path)
    result = None
    if (
        stored
        and stored.get("fingerprint") == fingerprint.get("fingerprint")
        and _dataset_outputs_are_complete(layout.root)
    ):
        result = _build_reused_result(layout.root)
    _log_prepare_summary(config.preprocessing, slide_sets, reused=result is not None)
    if result is None:
        _warn_image_backend(config, slide_sets)
        result = DatasetBuilder(
            config.preprocessing,
            slide_sets=slide_sets,
            fingerprint_metadata=fingerprint,
            registration_backend=registration_backend,
            registration_evidence=registration_evidence,
        ).run_all()
    return result


def _prepare_unpaired(config: RunConfig, config_path: Path) -> DatasetBuildResult:
    assert config.preprocessing is not None
    preprocessing = config.preprocessing
    root = preprocessing.dataset_root
    items = load_unpaired_inventory(preprocessing)
    assignments = unpaired_assignments(preprocessing, items, config.data.group_validation)
    inventory = root / preprocessing.inputs.inventory

    def observe() -> tuple[DataSnapshot, dict[str, Any]]:
        if (
            load_unpaired_inventory(preprocessing) != items
            or unpaired_assignments(preprocessing, items, config.data.group_validation)
            != assignments
        ):
            raise ValueError("Raw inventory or frozen assignment changed during preparation")
        assignment_file = preprocessing.split.assignment_file
        rows = []
        for item in items:
            split = assignments.get(getattr(item, f"{preprocessing.split.unit}_id", ""), "")
            rows.append(item.row(split=split))
            if item.mask_path:
                rows.append(replace(item.row(split=split, locator=item.mask_path), role="mask"))
        snapshot = build_snapshot(
            rows,
            kind="consumed",
            adapter="unpaired_inventory/1",
            roots={"dataset": root},
            hash_policy=config.data.hash_policy,
            group_validation=config.data.group_validation,
            selection={
                "domains": list(preprocessing.inputs.domains),
                "split_unit": preprocessing.split.unit,
            },
            sources={
                "inventory": str(preprocessing.inputs.inventory),
                "inventory_sha256": sha256_file(inventory),
                "assignment_sha256": sha256_file(root / assignment_file)
                if assignment_file
                else None,
            },
        )
        fingerprint = unpaired_fingerprint(
            dataset_root=root,
            resolved_config=config.resolved_yaml(),
            snapshot=snapshot,
            assignments=dict(assignments),
            inventory_path=inventory,
        )
        return snapshot, fingerprint

    snapshot, fingerprint = observe()
    layout = DatasetLayout(root).unpaired_build(fingerprint["fingerprint"])
    if not layout.root.parent.resolve().is_relative_to(root.resolve()):
        raise ValueError("Unpaired output directory escapes dataset_root")
    if layout.root.exists() or layout.root.is_symlink():
        stored = _load_json(layout.dataset_build_path)
        artifacts = stored.get("artifacts", {}) if stored else {}
        required = {
            path.relative_to(layout.root).as_posix()
            for path in (
                layout.group_metadata_path,
                layout.dataset_fingerprint_path,
                layout.split_assignment_path,
                layout.resolved_config_path,
                layout.source_snapshot.rows,
                layout.source_snapshot.metadata,
            )
        }
        if (
            not any(p.is_symlink() for p in (layout.root, *layout.root.rglob("*")))
            and stored
            and stored.get("fingerprint") == fingerprint["fingerprint"]
            and isinstance(artifacts, dict)
            and required <= artifacts.keys()
            and stored.get("domain_collections")
            == {
                domain: f"splits/{{split}}/{domain}/*.png"
                for domain in preprocessing.inputs.domains
            }
            and all(
                (layout.split_dir(split) / domain).is_dir()
                for split in DATASET_SPLITS
                for domain in preprocessing.inputs.domains
            )
            and all(
                not Path(name).is_absolute()
                and ".." not in Path(name).parts
                and (layout.root / name).is_file()
                and not (layout.root / name).is_symlink()
                and sha256_file(layout.root / name) == digest
                for name, digest in artifacts.items()
            )
            and {
                p.relative_to(layout.root).as_posix() for p in layout.root.rglob("*") if p.is_file()
            }
            == set(artifacts) | {layout.dataset_build_path.relative_to(layout.root).as_posix()}
        ):
            logger.info("Unpaired prepare reuse: %s", layout.root)
            return DatasetBuildResult.load(
                layout.dataset_build_path, output_root=layout.root, reused=True
            )
        raise FileExistsError(
            f"Incomplete, changed, or caller-owned build at {layout.root}; "
            "preserved without overwrite"
        )
    layout.root.parent.mkdir(parents=True, exist_ok=True)
    if not layout.root.parent.resolve().is_relative_to(root.resolve()):
        raise ValueError("Unpaired output directory escapes dataset_root")
    with tempfile.TemporaryDirectory(prefix=".prepare-", dir=layout.root.parent) as temporary:
        staging = DatasetLayout(Path(temporary))
        try:
            result = build_unpaired_dataset(
                preprocessing,
                items,
                assignments,
                staging.root,
                group_validation=config.data.group_validation,
                published_root=layout.root,
            )
            if observe()[1] != fingerprint:
                raise ValueError("Raw source/inventory changed during unpaired preparation")
            config_hash = save_stage_config_snapshots(
                config,
                config_path,
                input_dest=staging.input_config_path,
                resolved_dest=staging.resolved_config_path,
            )
            save_config_hash(config_hash, staging.config_hash_path)
            save_environment_snapshot(staging.environment_path)
            write_snapshot(snapshot, staging.source_snapshot)
            save_dataset_fingerprint(fingerprint, staging.dataset_fingerprint_path)
            result.save(staging.dataset_build_path, num_sets=0, num_sets_excluded=0)
            metadata = json.loads(staging.dataset_build_path.read_text(encoding="utf-8"))
            metadata.update(
                fingerprint=fingerprint["fingerprint"],
                artifacts={
                    path.relative_to(staging.root).as_posix(): sha256_file(path)
                    for path in sorted(staging.root.rglob("*"))
                    if path.is_file() and path != staging.dataset_build_path
                },
            )
            staging.dataset_build_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            if layout.root.exists() or layout.root.is_symlink():
                raise FileExistsError(f"Unpaired publication collision: {layout.root}")
            publish_directory_no_replace(staging.root, layout.root)
        except Exception as exc:
            # Only the owned temporary directory is cleaned; preserve failure evidence.
            failure = layout.root.parent / f"failure-{Path(temporary).name}.json"
            failure.write_text(
                json.dumps(
                    {
                        "fingerprint": fingerprint,
                        "error": str(exc),
                        "images": json.loads((staging.metadata_dir / "images.json").read_text())
                        if (staging.metadata_dir / "images.json").is_file()
                        else [],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            raise
    logger.info(
        "Unpaired prepare output: %s | data.domains=%s | data.group_metadata=metadata/groups.csv",
        layout.root,
        result.domain_collections,
    )
    return replace(result, output_root=layout.root)
