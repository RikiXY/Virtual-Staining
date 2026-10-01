from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from tests.config_helpers import write_config_data
from virtual_staining.applications.prepare import prepare
from virtual_staining.config.data import (
    AlignmentConfig,
    FilteringConfig,
    ForegroundFilterConfig,
    InputConfig,
    IOConfig,
    MaskConfig,
    PatchingConfig,
    PreprocessingConfig,
    SplitConfig,
)
from virtual_staining.config.run import RunConfig
from virtual_staining.data.alignment import (
    AlignmentError,
    AlignmentResult,
    AlignmentTransform,
    QCPolicy,
    RegistrationAttempt,
    RegistrationBackend,
    RegistrationFailure,
    RegistrationRuntime,
    SpatialEvidence,
    evaluate_alignment_qc,
    identity_alignment,
)
from virtual_staining.data.builder import DatasetBuilder
from virtual_staining.data.layout import DatasetLayout
from virtual_staining.data.manifest import DatasetManifest, ManifestMetadata
from virtual_staining.data.provenance import build_dataset_fingerprint_metadata
from virtual_staining.data.slide_set_processor import AlignmentQCError, SlideSetProcessor
from virtual_staining.data.slide_sets import SlideAsset, SlideSet
from virtual_staining.utils.image_io import (
    ImageMetadata,
    OpenSlideRegionImageReader,
    PillowRegionImageReader,
    convert_to_pyramidal_tiff,
)

MATRICES = {
    "AF": [[1, 0, -2], [0, 1, -1], [0, 0, 1]],
    # 90 degrees about (4.5, 4.5), followed by a y translation of -2.
    "HE": [[0, -1, 9], [1, 0, -2], [0, 0, 1]],
    "PAS": [[0.5, 0, -0.25], [0, 0.25, -0.125], [0, 0, 1]],
}


def _pixels(x, y):
    return np.stack((20 + 2 * x + 2 * y, 40 + 2 * x, 60 + 2 * y), axis=-1).astype(np.uint8)


def _mask(x, y):
    return (((x // 2 + y // 3) % 2) * 255).astype(np.uint8)


def _dataset(root, *, tiled=False, masks=False, targets=("HE", "PAS")):
    assets = {}
    inventory = {"set_id": "S1"}
    for name in ("LF", "AF", *targets):
        shape = (8, 8) if name == "LF" else (34, 20)
        y, x = np.indices(shape)
        path = Path(f"{name}.png")
        assert cv2.imwrite(str(root / path), _pixels(x, y))
        mask_path = Path(f"{name}-mask.png") if masks else None
        if mask_path is not None:
            assert cv2.imwrite(str(root / mask_path), _mask(x, y))
        assets[name] = SlideAsset(name, path, name == "LF", mask_path)
        prefix = f"{'input' if name in {'LF', 'AF'} else 'target'}__{name}"
        inventory[f"{prefix}_path"] = str(path)
        inventory[f"{prefix}_aligned"] = "true" if name == "LF" else "false"
    with (root / "inventory.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(inventory))
        writer.writeheader()
        writer.writerow(inventory)
    config = PreprocessingConfig(
        dataset_root=root,
        inputs=InputConfig(Path("inventory.csv"), ("LF", "AF"), "LF", targets),
        patching=PatchingConfig((4, 4), (4, 4), margin=0),
        masks=MaskConfig(generation="never", scale=0.5, save_patch_masks=masks),
        filtering=FilteringConfig(foreground=ForegroundFilterConfig(enabled=False)),
        split=SplitConfig(unit="set", train=1, val=0, test=0),
        io=IOConfig(tiled=tiled, backend="pillow"),
    )
    return config, SlideSet(
        "S1", (assets["LF"], assets["AF"]), tuple(assets[n] for n in targets), "LF"
    )


def _known_result(reference, moving, request):
    assert moving.tissue_support is moving.observation_validity is None
    assert reference.tissue_support is reference.observation_validity is None
    return AlignmentResult(
        "succeeded",
        "analytic_fixture",
        request,
        AlignmentTransform(
            moving.geometry, reference.geometry, "affine", np.array(MATRICES[moving.geometry.name])
        ),
        reason="analytic coordinates",
        attempt=RegistrationAttempt(
            stage="supplied",
            outcome="succeeded",
            runtime=RegistrationRuntime(backend="analytic_fixture", backend_version="1", seed=0),
        ),
    )


def _backend(register=_known_result, **kwargs):
    return RegistrationBackend(
        register, "analytic_fixture", "1", options={"matrices": MATRICES}, **kwargs
    )


def _manifest(root):
    layout = DatasetLayout(root)
    return DatasetManifest.from_csv(
        layout.manifest_path,
        root,
        ManifestMetadata.from_mapping(json.loads(layout.manifest_metadata_path.read_text())),
    )


def _results(root):
    with DatasetLayout(root).slide_sets_path.open(newline="") as handle:
        row = next(csv.DictReader(handle))
    return {
        key.removesuffix("__alignment_metadata"): AlignmentResult.from_dict(json.loads(value))
        for key, value in row.items()
        if key.endswith("__alignment_metadata") and value
    }


@pytest.mark.parametrize("tiled", [False, True])
@pytest.mark.parametrize("masks", [False, True])
@pytest.mark.parametrize("known_mpp", [False, True])
@pytest.mark.parametrize("targets", [("HE",), ("HE", "PAS")])
def test_known_transforms_materialize_on_reference_grid(
    tmp_path, monkeypatch, tiled, masks, known_mpp, targets
):
    config, slide_set = _dataset(tmp_path, tiled=tiled, masks=masks, targets=targets)
    mpp = (0.25, 0.5) if known_mpp else (None, None)
    monkeypatch.setattr(
        PillowRegionImageReader,
        "metadata",
        property(lambda reader: ImageMetadata(*reader.size, mpp_x=mpp[0], mpp_y=mpp[1])),
    )
    reads = []
    read_region = PillowRegionImageReader.read_region

    def bounded_read(reader, x, y, width, height):
        assert width * height <= 16
        assert min(x, y) >= 0
        assert x + width <= reader.size[0] and y + height <= reader.size[1]
        reads.append((width, height))
        return read_region(reader, x, y, width, height)

    if tiled:
        monkeypatch.setattr(PillowRegionImageReader, "read_region", bounded_read)
        monkeypatch.setattr(
            PillowRegionImageReader, "read_full", Mock(side_effect=AssertionError("full read"))
        )
    calls, returned = [], {}

    def register(reference, moving, request):
        calls.append((moving.geometry.name, reference.geometry.name))
        result = _known_result(reference, moving, request)
        # The transform, not a backend's method label, must choose resampling.
        returned[moving.geometry.name] = replace(result, method="identity")
        return returned[moving.geometry.name]

    backend = _backend(register)
    built = DatasetBuilder(config, (slide_set,), registration_backend=backend).run_all()
    assert built.train_count == 4
    assert calls == [(name, "LF") for name in ("AF", *targets)]
    manifest = _manifest(tmp_path)
    manifest.validate(check_files_exist=True)
    for record in manifest.records:
        assert set(record.input_paths) == {"LF", "AF"}
        assert set(record.target_paths) == set(targets)
        y, x = np.mgrid[record.y : record.y + 4, record.x : record.x + 4]
        # Analytic inverse coordinates, independent of the production warper.
        coordinates = {
            "LF": (x, y),
            "AF": (x + 2, y + 1),
            "HE": (y + 2, 9 - x),
            "PAS": (2 * x + 0.5, 4 * y + 0.5),
        }
        for name, path in {**record.input_paths, **record.target_paths}.items():
            sx, sy = coordinates[name]
            np.testing.assert_array_equal(cv2.imread(str(tmp_path / path)), _pixels(sx, sy))
        for name, path in record.foreground_mask_paths.items():
            if masks:
                assert path is not None
                sx, sy = coordinates[name]
                expected = _mask(np.floor(sx + 0.5), np.floor(sy + 0.5))
                np.testing.assert_array_equal(
                    cv2.imread(str(tmp_path / path), cv2.IMREAD_GRAYSCALE), expected
                )
            else:
                assert path is None
    results = _results(tmp_path)
    assert results["LF"].reason == "reference" and results["LF"].qc is None
    assert results["LF"].candidate is not None and results["LF"].candidate.family == "identity"
    for name, result in results.items():
        assert result.candidate is not None
        assert result.candidate.moving.mpp == result.candidate.reference.mpp == mpp
        assert result.qc is None
        if name != "LF":
            assert result.metadata == returned[name].metadata
            assert result.candidate.moving.shape != result.candidate.reference.shape
    if tiled:
        assert reads
        if "PAS" in targets:
            assert len(reads) > 4 * 4  # The anisotropic inverse forces source subdivision.
    fingerprint = json.loads(DatasetLayout(tmp_path).dataset_fingerprint_path.read_text())
    assert fingerprint["registration"] == backend.metadata
    assert "attempt" not in json.dumps(fingerprint)
    assert "object at" not in json.dumps(fingerprint) and "function" not in json.dumps(fingerprint)


@pytest.mark.parametrize("status", ["unassessed", "accepted", "rejected", "insufficient_evidence"])
@pytest.mark.parametrize("action", [None, "continue", "skip_set", "error"])
def test_independent_qc_disposition_is_explicit(tmp_path, status, action):
    config, slide_set = _dataset(tmp_path, masks=True)
    # Explicit QC error must not be swallowed by the general skip policy (or vice versa).
    config = replace(
        config, alignment=AlignmentConfig(on_failure="skip_set" if action == "error" else "error")
    )
    returned = {}

    def register(reference, moving, request):
        result = _known_result(reference, moving, request)
        assert result.candidate is not None
        if status != "unassessed":
            points = mapped = None
            if status != "insufficient_evidence":
                points = np.array([[3.0, 3.0]])
                mapped = result.candidate.map_points(points)
                mapped = mapped + (0.5 if status == "rejected" else 0)
            qc = evaluate_alignment_qc(
                result.candidate,
                request,
                QCPolicy({"landmark_rms": (0, 0)}),
                moving_landmarks=points,
                reference_landmarks=mapped,
            )
            result = replace(result, qc=qc, attempt=replace(result.attempt, qc_status=qc.status))
            assert qc.status == status
        returned[moving.geometry.name] = result
        return result

    backend = _backend(register, qc_disposition={} if action is None else {status: action})
    builder = DatasetBuilder(config, (slide_set,), registration_backend=backend)
    if action == "error":
        with pytest.raises(AlignmentQCError, match=status) as caught:
            builder.run_all()
        assert caught.value.result is returned["AF"]
        assert not DatasetLayout(tmp_path).manifest_path.exists()
    else:
        result = builder.run_all()
        assert result.train_count == (0 if action == "skip_set" else 4)
        saved = _results(tmp_path)["AF"]
        assert saved.metadata == returned["AF"].metadata
        assert saved.backend_status == "succeeded" and saved.attempt.failure is None
        if saved.qc is not None:
            assert saved.qc.status == status
            assert "support_iou" in saved.qc.missing_evidence  # Supplied masks are not evidence.
        else:
            assert status == "unassessed"


def test_supplied_spatial_evidence_uses_existing_qc_contract(tmp_path):
    config, slide_set = _dataset(tmp_path)

    def register(reference, moving, request):
        result = _known_result(reference, moving, request)
        assert result.candidate is not None

        def evidence(image, kind):
            return SpatialEvidence(
                image.geometry, image.grid, np.ones(image.grid.shape, dtype=bool), kind
            )

        qc = evaluate_alignment_qc(
            result.candidate,
            request,
            QCPolicy({"support_iou": (1, 1), "observation_valid_fraction": (1, 1)}),
            moving_support=evidence(moving, "tissue_support"),
            reference_support=evidence(reference, "tissue_support"),
            moving_validity=evidence(moving, "observation_validity"),
            reference_validity=evidence(reference, "observation_validity"),
        )
        return replace(result, qc=qc)

    DatasetBuilder(config, (slide_set,), registration_backend=_backend(register)).run_all()
    for name, result in _results(tmp_path).items():
        if name != "LF":
            assert result.qc is not None and result.qc.status == "accepted"
            assert (
                result.qc.metrics["support_iou"]
                == result.qc.metrics["observation_valid_fraction"]
                == 1
            )


@pytest.mark.parametrize("on_failure", ["error", "skip_set"])
@pytest.mark.parametrize("category", ["optimizer_failed", "interrupted"])
def test_typed_failure_and_interruption(tmp_path, on_failure, category):
    config, slide_set = _dataset(tmp_path)
    config = replace(config, alignment=AlignmentConfig(on_failure=on_failure))
    failure = RegistrationFailure(category, "fixture failure")

    def register(reference, moving, request):
        if moving.geometry.name != "PAS":
            return _known_result(reference, moving, request)
        return AlignmentResult(
            "failed",
            "analytic_fixture",
            request,
            None,
            attempt=RegistrationAttempt(
                stage="optimizer",
                outcome="failed",
                runtime=RegistrationRuntime(backend="analytic_fixture"),
                failure=failure,
            ),
        )

    builder = DatasetBuilder(config, (slide_set,), registration_backend=_backend(register))
    if on_failure == "error":
        with pytest.raises(AlignmentError, match="fixture failure"):
            builder.run_all()
        assert not DatasetLayout(tmp_path).manifest_path.exists()
        assert not DatasetLayout(tmp_path).dataset_build_path.exists()
    else:
        assert builder.run_all().train_count == 0
        assert not _manifest(tmp_path).records
        results = _results(tmp_path)
        assert set(results) == {"LF", "AF", "HE", "PAS"}
        assert results["PAS"].attempt.failure == failure and results["PAS"].qc is None


@pytest.mark.parametrize("interrupt", [False, True])
def test_injected_preparation_cleans_failed_owned_writes(tmp_path, monkeypatch, interrupt):
    config, slide_set = _dataset(tmp_path)
    config = replace(config, alignment=AlignmentConfig(on_failure="skip_set"))
    write = cv2.imwrite

    def fail(path, image):
        if "__target__PAS" in path:
            Path(path).write_bytes(b"partial")
            if interrupt:
                raise KeyboardInterrupt("write interrupted")
            return False
        return write(path, image)

    monkeypatch.setattr(cv2, "imwrite", fail)
    builder = DatasetBuilder(config, (slide_set,), registration_backend=_backend())
    if interrupt:
        with pytest.raises(KeyboardInterrupt):
            builder.run_all()
        assert not DatasetLayout(tmp_path).manifest_path.exists()
    else:
        assert builder.run_all().train_count == 0
        assert not _manifest(tmp_path).records
    assert list((tmp_path / "splits").rglob("*.png")) == []


def test_raised_backend_interruption_closes_owned_readers(tmp_path, monkeypatch):
    config, slide_set = _dataset(tmp_path, tiled=True)
    config = replace(config, alignment=AlignmentConfig(on_failure="skip_set"))
    close = Mock()
    monkeypatch.setattr(PillowRegionImageReader, "close", close)
    backend = _backend(Mock(side_effect=KeyboardInterrupt("cancelled")))
    with pytest.raises(KeyboardInterrupt, match="cancelled"):
        DatasetBuilder(config, (slide_set,), registration_backend=backend).run_all()
    assert close.call_count == 4
    assert not DatasetLayout(tmp_path).manifest_path.exists()
    assert not DatasetLayout(tmp_path).dataset_build_path.exists()


def test_injected_preparation_reads_native_wsi_with_bounded_regions(tmp_path, monkeypatch):
    config, slide_set = _dataset(tmp_path, tiled=True)
    assets = []
    for asset in slide_set.assets:
        size = 256 if asset.modality == "LF" else 512
        y, x = np.indices((size, size))
        assert cv2.imwrite(str(tmp_path / asset.path), _pixels(x, y))
        path = asset.path.with_suffix(".tif")
        convert_to_pyramidal_tiff(tmp_path / asset.path, tmp_path / path)
        assets.append(replace(asset, path=path))
    slide_set = replace(slide_set, inputs=tuple(assets[:2]), targets=tuple(assets[2:]))
    config = replace(
        config,
        io=IOConfig(tiled=True, backend="openslide"),
        patching=PatchingConfig((64, 64), (64, 64), margin=0),
        masks=replace(config.masks, scale=0.125),
    )
    matrices = {
        "AF": MATRICES["AF"],
        "HE": [[0, -1, 255], [1, 0, 0], [0, 0, 1]],
        "PAS": [[0.5, 0, 0], [0, 0.5, 0], [0, 0, 1]],
    }

    def register(reference, moving, request):
        result = _known_result(reference, moving, request)
        return replace(
            result,
            candidate=AlignmentTransform(
                moving.geometry,
                reference.geometry,
                "affine",
                np.array(matrices[moving.geometry.name]),
            ),
        )

    backend = RegistrationBackend(
        register, "native_wsi_fixture", "1", options={"matrices": matrices}
    )
    read = OpenSlideRegionImageReader.read_region
    reads = []

    def bounded_read(reader, x, y, width, height):
        assert width * height <= 64 * 64
        reads.append((x, y, width, height))
        return read(reader, x, y, width, height)

    monkeypatch.setattr(OpenSlideRegionImageReader, "read_region", bounded_read)
    monkeypatch.setattr(
        OpenSlideRegionImageReader, "read_full", Mock(side_effect=AssertionError("full slide"))
    )
    assert (
        DatasetBuilder(config, (slide_set,), registration_backend=backend).run_all().train_count
        == 16
    )
    assert len(reads) > 16 * 4
    for record in _manifest(tmp_path).records:
        y, x = np.mgrid[record.y : record.y + 64, record.x : record.x + 64]
        for name, (sx, sy) in {"HE": (y, 255 - x), "PAS": (2 * x, 2 * y)}.items():
            np.testing.assert_array_equal(
                cv2.imread(str(tmp_path / record.target_paths[name])), _pixels(sx, sy)
            )


@pytest.mark.parametrize("family", ["identity", "affine"])
def test_injected_out_of_bounds_pixels_and_masks_use_shared_resampling(tmp_path, family):
    config, slide_set = _dataset(tmp_path, masks=True)
    config = replace(
        config,
        filtering=replace(config.filtering, max_white_ratio=1, max_largest_white_component_ratio=1),
    )
    # An identity candidate under affine permission can have a smaller native extent.
    y, x = np.indices((4, 4))
    for asset in slide_set.assets[1:]:
        assert cv2.imwrite(str(tmp_path / asset.path), _pixels(x, y))
        assert asset.mask_path is not None
        assert cv2.imwrite(str(tmp_path / asset.mask_path), np.full((4, 4), 255, np.uint8))

    def register(reference, moving, request):
        result = _known_result(reference, moving, request)
        matrix = np.array([[1, 0, 2 if family == "affine" else 0], [0, 1, 0], [0, 0, 1]])
        return replace(
            result,
            candidate=AlignmentTransform(moving.geometry, reference.geometry, family, matrix),
        )

    assert (
        DatasetBuilder(config, (slide_set,), registration_backend=_backend(register))
        .run_all()
        .train_count
        == 4
    )
    for record in _manifest(tmp_path).records:
        y, x = np.mgrid[record.y : record.y + 4, record.x : record.x + 4]
        sx = x - (2 if family == "affine" else 0)
        inside = (sx >= 0) & (sx < 4) & (y < 4)
        expected = np.full((4, 4, 3), 255, np.uint8)
        expected[inside] = _pixels(sx, y)[inside]
        for name, path in record.target_paths.items():
            np.testing.assert_array_equal(cv2.imread(str(tmp_path / path)), expected)
            mask_path = record.foreground_mask_paths[name]
            assert mask_path is not None
            np.testing.assert_array_equal(
                cv2.imread(str(tmp_path / mask_path), cv2.IMREAD_GRAYSCALE),
                inside.astype(np.uint8) * 255,
            )


def test_backend_identity_is_frozen_and_controls_actual_reuse(tmp_path):
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
    options = {"matrices": MATRICES, "qc": {"policy": "none"}}
    backend = RegistrationBackend(calls, "analytic_fixture", "1", options=options)
    options["qc"]["policy"] = "changed"
    assert backend.metadata["options"]["qc"]["policy"] == "none"
    detached = backend.metadata
    detached["options"]["qc"]["policy"] = "also changed"
    assert backend.metadata != detached
    assert not prepare(run, path, registration_backend=backend).reused
    assert calls.call_count == 3
    equivalent = RegistrationBackend(
        calls, "analytic_fixture", "1", options={"qc": {"policy": "none"}, "matrices": MATRICES}
    )
    assert prepare(run, path, registration_backend=equivalent).reused
    assert calls.call_count == 3
    baseline = json.loads(DatasetLayout(tmp_path).dataset_fingerprint_path.read_text())
    for changed in (
        RegistrationBackend(
            calls, "different_fixture", "1", options=equivalent.metadata["options"]
        ),
        RegistrationBackend(calls, "analytic_fixture", "2", options=equivalent.metadata["options"]),
        RegistrationBackend(calls, "analytic_fixture", "1", options=options),
        RegistrationBackend(
            calls, "analytic_fixture", "1", options=options, qc_disposition={"rejected": "skip_set"}
        ),
    ):
        assert not prepare(run, path, registration_backend=changed).reused
        current = json.loads(DatasetLayout(tmp_path).dataset_fingerprint_path.read_text())
        assert current["fingerprint"] != baseline["fingerprint"]
        assert current["registration"] == changed.metadata
    assert calls.call_count == 15
    with pytest.raises(ValueError, match="Fingerprint registration identity"):
        DatasetBuilder(config, (slide_set,), baseline, registration_backend=_backend())
    with pytest.raises(ValueError, match="Fingerprint registration identity"):
        DatasetBuilder(config, (slide_set,), baseline)


def test_execution_results_do_not_change_policy_fingerprint(tmp_path):
    config, slide_set = _dataset(tmp_path)
    baseline = None
    for timing in (1, 2):

        def register(reference, moving, request, timing=timing):
            result = _known_result(reference, moving, request)
            return replace(result, attempt=replace(result.attempt, duration_seconds=timing))

        backend = _backend(register)
        DatasetBuilder(config, (slide_set,), registration_backend=backend).run_all()
        metadata = json.loads(DatasetLayout(tmp_path).dataset_fingerprint_path.read_text())
        fingerprint = metadata["fingerprint"]
        assert baseline is None or baseline == fingerprint
        baseline = fingerprint
        assert _results(tmp_path)["AF"].attempt.duration_seconds == timing
        assert (
            build_dataset_fingerprint_metadata(
                dataset_root=tmp_path,
                preprocessing_config=config.to_dict(),
                slide_sets=(slide_set,),
                registration_backend=backend,
            )["fingerprint"]
            == fingerprint
        )


@pytest.mark.parametrize(
    "options", [{"bad": float("nan")}, {"bad": object()}, {1: "bad"}, {"nested": {"bad": (1, 2)}}]
)
def test_unstable_backend_options_are_rejected(options):
    with pytest.raises(ValueError, match="JSON"):
        RegistrationBackend(_known_result, "fixture", "1", options=options)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"identifier": ""},
        {"version": ""},
        {"register": None},
        {"qc_disposition": {"rejected": "accept"}},
        {"qc_disposition": {"unknown": "error"}},
        {"qc_disposition": {"rejected": []}},
    ],
)
def test_backend_requires_explicit_valid_identity_and_disposition(kwargs):
    with pytest.raises(ValueError):
        RegistrationBackend(
            **({"register": _known_result, "identifier": "fixture", "version": "1"} | kwargs)
        )


@pytest.mark.parametrize("invalid", ["frame", "request", "family", "return_type"])
def test_injected_result_boundary_rejects_invalid_contract(tmp_path, invalid):
    config, slide_set = _dataset(tmp_path)

    def register(reference, moving, request):
        result = _known_result(reference, moving, request)
        assert result.candidate is not None
        if invalid == "return_type":
            return result.candidate
        if invalid == "request":
            return replace(result, request=replace(request, relationship="same_section_restained"))
        if invalid == "frame":
            return replace(result, candidate=replace(result.candidate, moving=reference.geometry))
        # A true inventory declaration requests identity, which cannot be escalated.
        return result

    if invalid == "family":
        slide_set = replace(
            slide_set,
            inputs=(slide_set.inputs[0], replace(slide_set.inputs[1], already_aligned=True)),
        )
        config = replace(config, alignment=AlignmentConfig(validate_declared=False))
    with pytest.raises(AlignmentError, match="Injected"):
        DatasetBuilder(config, (slide_set,), registration_backend=_backend(register)).run_all()


@pytest.mark.parametrize("mismatch", ["shape", "mpp"])
def test_injected_identity_does_not_bypass_shared_frame_geometry(tmp_path, monkeypatch, mismatch):
    config, slide_set = _dataset(tmp_path)
    if mismatch == "mpp":
        for asset in slide_set.assets:
            assert cv2.imwrite(str(tmp_path / asset.path), np.full((8, 8, 3), 100, np.uint8))
        monkeypatch.setattr(
            PillowRegionImageReader,
            "metadata",
            property(
                lambda reader: ImageMetadata(
                    *reader.size, mpp_x=0.25 if reader.path.name == "LF.png" else 0.5, mpp_y=0.5
                )
            ),
        )
    slide_set = replace(
        slide_set, inputs=tuple(replace(a, already_aligned=True) for a in slide_set.inputs)
    )
    config = replace(config, alignment=AlignmentConfig(validate_declared=False))

    def register(reference, moving, request):
        return identity_alignment(reference.geometry, moving.geometry, request)

    with pytest.raises(AlignmentError, match="identity alignment"):
        DatasetBuilder(config, (slide_set,), registration_backend=_backend(register)).run_all()


@pytest.mark.parametrize("mode", ["auto", "always", "never"])
@pytest.mark.parametrize("declared", [None, False, True])
def test_builtin_alignment_modes_keep_identity_and_real_sift(tmp_path, mode, declared):
    config, slide_set = _dataset(tmp_path, targets=("HE",))
    config = replace(config, alignment=AlignmentConfig(mode=mode))
    pixels = np.random.default_rng(9).integers(10, 200, (200, 200, 3), dtype=np.uint8)
    for asset in slide_set.assets:
        assert cv2.imwrite(str(tmp_path / asset.path), pixels)
    slide_set = replace(
        slide_set,
        inputs=(slide_set.inputs[0], replace(slide_set.inputs[1], already_aligned=declared)),
        targets=tuple(replace(a, already_aligned=declared) for a in slide_set.targets),
    )
    processor = SlideSetProcessor(config, slide_set)
    try:
        processor.compute_masks()
        if mode == "never" and declared is False:
            with pytest.raises(AlignmentError, match="contradicts"):
                processor.align()
        else:
            processor.align()
            for state in (processor.inputs["AF"], processor.targets["HE"]):
                result = state.alignment
                assert result is not None and result.candidate is not None
                assert result.qc is None
                assert result.method == (
                    "identity" if declared is True or mode == "never" else "affine_sift"
                )
                np.testing.assert_allclose(result.candidate.matrix, np.eye(3), atol=0.01)
    finally:
        processor.close()
