from __future__ import annotations

import json
from dataclasses import replace
from typing import Literal
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from tests.config_helpers import write_config_data
from tests.data.test_injected_registration import (
    _backend,
    _dataset,
    _known_result,
    _manifest,
    _results,
)
from virtual_staining.applications.prepare import prepare
from virtual_staining.config.run import RunConfig
from virtual_staining.data.alignment import (
    AlignmentError,
    AlignmentImage,
    AlignmentTransform,
    GridGeometry,
    ImageGeometry,
    QCPolicy,
    RegistrationBackend,
    SpatialEvidence,
    evaluate_alignment_qc,
)
from virtual_staining.data.builder import DatasetBuilder
from virtual_staining.data.layout import DatasetLayout
from virtual_staining.data.provenance import build_dataset_fingerprint_metadata
from virtual_staining.data.slide_set_processor import SlideSetProcessor
from virtual_staining.utils.image_io import PillowRegionImageReader


def _map(
    name,
    kind: Literal["tissue_support", "observation_validity"] = "tissue_support",
    *,
    grid=None,
    values=None,
    source="observations/1",
):
    geometry = ImageGeometry(name, (8, 8) if name == "LF" else (34, 20))
    grid = grid or GridGeometry(geometry.shape, np.eye(3))
    if values is None:
        y, x = np.indices(grid.shape)
        values = (x + y) % 2 == 0
    return SpatialEvidence(geometry, grid, values, kind, source=source)


@pytest.mark.parametrize("tiled", [False, True])
@pytest.mark.parametrize("block_size", [3, 256])
def test_builder_transports_distinct_evidence_to_backend_and_each_patch(
    tmp_path, monkeypatch, tiled, block_size
):
    monkeypatch.setattr("virtual_staining.data.alignment.registration._QC_BLOCK_SIZE", block_size)
    config, slide_set = _dataset(tmp_path, tiled=tiled, masks=True)
    supplied = {
        ("S1", "LF"): (_map("LF"), _map("LF", "observation_validity")),
        ("S1", "AF"): (
            _map("AF", grid=GridGeometry.resized_crop((17, 10), origin=(0, 0), scale=(2, 2))),
        ),
        ("S1", "HE"): (
            _map(
                "HE",
                "observation_validity",
                grid=GridGeometry.resized_crop((8, 8), origin=(2, 1), scale=(2, 3)),
                values=np.zeros((8, 8), bool),
            ),
        ),
        ("S1", "PAS"): (_map("PAS"), _map("PAS", "observation_validity")),
    }
    seen = []

    def register(reference, moving, request):
        seen.append(moving.geometry.name)
        for image in (reference, moving):
            by_kind = {item.kind: item for item in supplied[("S1", image.geometry.name)]}
            assert image.tissue_support is by_kind.get("tissue_support")
            assert image.observation_validity is by_kind.get("observation_validity")
        result = _known_result(reference, moving, request)
        assert result.candidate is not None
        qc = evaluate_alignment_qc(
            result.candidate,
            request,
            QCPolicy(),
            moving_support=moving.tissue_support,
            reference_support=reference.tissue_support,
            moving_validity=moving.observation_validity,
            reference_validity=reference.observation_validity,
        )
        return replace(result, qc=qc)

    checked = []
    original = SlideSetProcessor._is_valid_patch

    def inspect_pipeline(processor, patches, masks):
        for name, state in (processor.inputs | processor.targets).items():
            warped = state.warped_patch
            assert warped is not None and warped.image is patches[name]
            kinds = {item.kind for item in supplied[("S1", name)]}
            assert (warped.tissue_support is not None) == ("tissue_support" in kinds)
            assert (warped.observation_validity is not None) == ("observation_validity" in kinds)
            assert (warped.tissue_known is not None) == ("tissue_support" in kinds)
            assert (warped.observation_known is not None) == ("observation_validity" in kinds)
        assert not processor.targets["HE"].warped_patch.observation_validity.any()
        checked.append(True)
        return original(processor, patches, masks)

    monkeypatch.setattr(SlideSetProcessor, "_is_valid_patch", inspect_pipeline)
    backend = _backend(register)
    assert (
        DatasetBuilder(
            config, (slide_set,), registration_backend=backend, registration_evidence=supplied
        )
        .run_all()
        .train_count
        == 4
    )
    assert seen == ["AF", "HE", "PAS"] and len(checked) == 4
    manifest = _manifest(tmp_path)
    manifest.validate(check_files_exist=True)
    assert all(
        set(row.input_paths) == {"LF", "AF"} and set(row.target_paths) == {"HE", "PAS"}
        for row in manifest.records
    )
    results = _results(tmp_path)
    assert results["LF"].qc is None
    for name in ("AF", "HE", "PAS"):
        result = results[name]
        assert result.candidate is not None and result.candidate.moving.name == name
        assert result.candidate.reference.name == "LF"
        assert result.qc is not None and result.qc.status == "insufficient_evidence"
    af, he, pas = (results[name].qc for name in ("AF", "HE", "PAS"))
    assert af is not None and he is not None and pas is not None
    assert af.metrics["observation_valid_fraction"] is None
    assert he.metrics["support_iou"] is None
    assert pas.metrics["support_iou"] is not None


@pytest.mark.parametrize("tiled", [False, True])
def test_affine_patch_evidence_has_analytic_known_invalid_and_unknown_regions(tmp_path, tiled):
    config, slide_set = _dataset(tmp_path, tiled=tiled, masks=True)
    support = _map(
        "AF",
        grid=GridGeometry((2, 2), np.array([[2, 0, 1], [0, 3, 0], [0, 0, 1]])),
        values=np.array([[False, True], [True, False]]),
    )
    validity = _map(
        "AF",
        "observation_validity",
        grid=GridGeometry((2, 3), np.array([[1, 0, 1], [0, 1, 0], [0, 0, 1]])),
        values=np.array([[True, False, True], [True, True, True]]),
    )
    mask_path = slide_set.inputs[1].mask_path
    assert mask_path is not None
    assert cv2.imwrite(str(tmp_path / mask_path), np.full((34, 20), 255, np.uint8))

    def register(reference, moving, request):
        result = _known_result(reference, moving, request)
        return replace(
            result,
            candidate=AlignmentTransform(
                moving.geometry,
                reference.geometry,
                "affine",
                np.array([[1, 0, 0.5], [0, 1, 0], [0, 0, 1]]),
            ),
        )

    processor = SlideSetProcessor(
        config,
        slide_set,
        registration_backend=_backend(register),
        registration_evidence={("S1", "AF"): (support, validity)},
    )
    try:
        processor.compute_masks()
        processor.align()
        state = processor.inputs["AF"]
        image, foreground = processor.extract_asset_patch(state, x=0, y=0, width=4, height=2)
        patch = state.warped_patch
        assert patch is not None and patch.image is image
        np.testing.assert_array_equal(patch.geometric_validity, [[False, True, True, True]] * 2)
        np.testing.assert_array_equal(
            patch.observation_validity, [[False, False, False, False], [False, False, True, True]]
        )
        np.testing.assert_array_equal(patch.observation_known, [[False, False, True, True]] * 2)
        np.testing.assert_array_equal(patch.tissue_support, [[False, False, False, True]] * 2)
        np.testing.assert_array_equal(patch.tissue_known, [[False, True, True, True]] * 2)
        # Nearest foreground sampling remains independent of continuous image contributors.
        np.testing.assert_array_equal(foreground, np.full((2, 4), 255, np.uint8))
        processor.extract_asset_patch(processor.targets["HE"], x=0, y=0, width=4, height=2)
        absent = processor.targets["HE"].warped_patch
        assert absent is not None
        assert absent.observation_validity is absent.observation_known is None
        assert absent.tissue_support is absent.tissue_known is None
    finally:
        processor.close()


def test_reference_evidence_keeps_direct_image_extraction(tmp_path, monkeypatch):
    config, slide_set = _dataset(tmp_path, tiled=True)
    support, validity = _map("LF"), _map("LF", "observation_validity")
    processor = SlideSetProcessor(
        config,
        slide_set,
        registration_backend=_backend(),
        registration_evidence={("S1", "LF"): (support, validity)},
    )
    try:
        processor.compute_masks()
        processor.align()
        read = PillowRegionImageReader.read_region
        calls = []

        def region(reader, x, y, width, height):
            calls.append((x, y, width, height))
            return read(reader, x, y, width, height)

        monkeypatch.setattr(PillowRegionImageReader, "read_region", region)
        monkeypatch.setattr(
            "virtual_staining.data.slide_set_processor.warp_aligned_patch",
            Mock(side_effect=AssertionError("reference resampling")),
        )
        image, _ = processor.extract_asset_patch(processor.reference, x=2, y=1, width=4, height=3)
        patch = processor.reference.warped_patch
        assert patch is not None and patch.image is image and calls == [(2, 1, 4, 3)]
        assert patch.geometric_validity.all()
        np.testing.assert_array_equal(patch.tissue_support, support.values[1:4, 2:6])
        np.testing.assert_array_equal(patch.observation_validity, validity.values[1:4, 2:6])
        assert patch.tissue_known is not None and patch.observation_known is not None
        assert patch.tissue_known.all() and patch.observation_known.all()
    finally:
        processor.close()


@pytest.mark.parametrize("mismatch", ["set", "modality", "asset", "shape", "mpp", "duplicate"])
def test_preparation_rejects_misbound_evidence(tmp_path, mismatch):
    config, slide_set = _dataset(tmp_path)
    item = _map("AF")
    key = ("other" if mismatch == "set" else "S1", "other" if mismatch == "modality" else "AF")
    if mismatch == "asset":
        item = replace(item, asset=replace(item.asset, name="HE"))
    if mismatch == "shape":
        item = replace(item, asset=replace(item.asset, shape=(20, 34)))
    if mismatch == "mpp":
        item = replace(item, asset=replace(item.asset, mpp=(0.25, 0.5)))
    maps = (item, item) if mismatch == "duplicate" else (item,)
    with pytest.raises(AlignmentError, match="evidence|Evidence"):
        DatasetBuilder(
            config, (slide_set,), registration_backend=_backend(), registration_evidence={key: maps}
        ).run_all()
    assert not DatasetLayout(tmp_path).manifest_path.exists()


def test_spatial_evidence_validation_and_immutable_values():
    original = np.array([[True, False], [False, True]])
    item = _map("AF", grid=GridGeometry((2, 2), np.eye(3)), values=original)
    assert original.flags.writeable and not item.values.flags.writeable
    original[:] = False
    np.testing.assert_array_equal(item.values, [[True, False], [False, True]])
    with pytest.raises(ValueError):
        item.values[0, 0] = False
    with pytest.raises(AlignmentError, match="Unsupported evidence semantics"):
        replace(item, kind="foreground")
    with pytest.raises(AlignmentError, match="boolean array"):
        replace(item, values=np.zeros((1, 2), bool))
    with pytest.raises(AlignmentError, match="boolean array"):
        replace(item, values=np.zeros((2, 2), np.uint8))
    with pytest.raises(AlignmentError, match="source"):
        replace(item, source="")
    with pytest.raises(AlignmentError, match="kind/asset"):
        AlignmentImage(np.zeros((2, 2), np.uint8), item.asset, item.grid, observation_validity=item)


def test_evidence_reuse_tracks_values_geometry_source_and_preserves_runtime_identity(tmp_path):
    config, slide_set = _dataset(tmp_path)
    path = write_config_data(
        tmp_path / "run.yaml",
        {
            "dataset_root": str(tmp_path),
            "results_path": str(tmp_path / "results"),
            "run_name": "prepare",
            "model": {"inputs": ["LF", "AF"], "outputs": ["HE", "PAS"]},
            "preprocessing": config.to_dict(),
        },
    )
    run = RunConfig.from_yaml(path)
    calls = Mock(side_effect=_known_result)
    # Configured adapter/policy identity differs explicitly from its underlying engine.
    backend = RegistrationBackend(
        calls,
        "evidence_adapter",
        "2",
        options={"engine": "analytic_fixture", "engine_version": "1"},
    )
    support = _map("AF")
    validity = _map("AF", "observation_validity")
    supplied = {("S1", "AF"): (support, validity)}
    assert not prepare(
        run, path, registration_backend=backend, registration_evidence=supplied
    ).reused
    baseline = json.loads(DatasetLayout(tmp_path).dataset_fingerprint_path.read_text())
    # Reordered maps and independently constructed arrays retain the same identity.
    equivalent = {
        ("S1", "AF"): (
            replace(validity, values=validity.values.copy(order="F")),
            replace(support, values=support.values.copy()),
        )
    }
    assert prepare(run, path, registration_backend=backend, registration_evidence=equivalent).reused
    assert calls.call_count == 3
    for changed in (
        replace(support, values=~support.values),
        replace(
            support,
            grid=GridGeometry(support.grid.shape, np.array([[1, 0, 1], [0, 1, 0], [0, 0, 1]])),
        ),
        replace(support, source="observations/2"),
    ):
        maps = {("S1", "AF"): (changed, validity)}
        assert not prepare(
            run, path, registration_backend=backend, registration_evidence=maps
        ).reused
        assert prepare(run, path, registration_backend=backend, registration_evidence=maps).reused
        metadata = json.loads(DatasetLayout(tmp_path).dataset_fingerprint_path.read_text())
        assert metadata["fingerprint"] != baseline["fingerprint"]
        assert metadata["registration"]["identifier"] == "evidence_adapter"
        assert _results(tmp_path)["AF"].attempt.runtime.backend == "analytic_fixture"
        assert _results(tmp_path)["AF"].qc is None  # Supplying evidence does not run QC.
    assert calls.call_count == 12
    records = baseline["registration_evidence"]
    assert len(records) == 2
    assert all(
        item["set_id"] == "S1"
        and item["modality"] == "AF"
        and item["values_sha256"].startswith("sha256:")
        for item in records
    )
    encoded = json.dumps(baseline)
    assert all(
        word not in encoded
        for word in ('"values"', '"attempt"', '"qc_status"', "array(", "object at", "function")
    )
    with pytest.raises(ValueError, match="Fingerprint registration evidence"):
        DatasetBuilder(config, (slide_set,), baseline, registration_backend=backend)
    with pytest.raises(ValueError, match="Fingerprint registration evidence"):
        DatasetBuilder(
            config,
            (slide_set,),
            baseline,
            registration_backend=backend,
            registration_evidence={("S1", "AF"): (replace(support, source="other"),)},
        )
    builder = DatasetBuilder(
        config, (slide_set,), baseline, registration_backend=backend, registration_evidence=supplied
    )
    supplied.clear()
    assert builder.registration_evidence  # Caller mapping mutations cannot change a queued build.


def test_evidence_binding_isolated_between_sets_with_same_modalities(tmp_path):
    config, first = _dataset(tmp_path)
    second = replace(first, set_id="S2")
    supplied = {
        ("S1", "AF"): (_map("AF", source="first"),),
        ("S2", "AF"): (_map("AF", source="second", values=np.zeros((34, 20), bool)),),
    }
    seen = []

    def register(reference, moving, request):
        assert reference.tissue_support is None
        if moving.geometry.name == "AF":
            seen.append(moving.tissue_support)
        else:
            assert moving.tissue_support is None
        return _known_result(reference, moving, request)

    result = DatasetBuilder(
        config,
        (first, second),
        registration_backend=_backend(register),
        registration_evidence=supplied,
    ).run_all()
    assert result.train_count == 8
    assert seen[0] is supplied[("S1", "AF")][0]
    assert seen[1] is supplied[("S2", "AF")][0]
    assert {row.set_id for row in _manifest(tmp_path).records} == {"S1", "S2"}


def test_evidence_fingerprint_is_independent_of_execution_and_map_order(tmp_path):
    config, slide_set = _dataset(tmp_path)
    evidence = {("S1", "AF"): (_map("AF"),), ("S1", "HE"): (_map("HE", "observation_validity"),)}
    fingerprints = []
    for timing in (1, 2):

        def register(reference, moving, request, timing=timing):
            result = _known_result(reference, moving, request)
            return replace(result, attempt=replace(result.attempt, duration_seconds=timing))

        backend = _backend(register)
        DatasetBuilder(
            config, (slide_set,), registration_backend=backend, registration_evidence=evidence
        ).run_all()
        metadata = json.loads(DatasetLayout(tmp_path).dataset_fingerprint_path.read_text())
        fingerprints.append(metadata["fingerprint"])
        assert _results(tmp_path)["AF"].attempt.duration_seconds == timing
        evidence = dict(reversed(list(evidence.items())))
        assert (
            build_dataset_fingerprint_metadata(
                dataset_root=tmp_path,
                preprocessing_config=config.to_dict(),
                slide_sets=(slide_set,),
                registration_backend=backend,
                registration_evidence=evidence,
            )["fingerprint"]
            == fingerprints[-1]
        )
    assert fingerprints[0] == fingerprints[1]
