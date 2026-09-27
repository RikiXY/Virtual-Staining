"""Reusable stage primitives run from their natural inputs alone (see docs/library_api.md).

Each test supplies only what the primitive consumes, plus a sentinel tree holding the
predecessor artifacts a tracked workflow would have (manifest, run metadata, checkpoint,
inventory). A process-wide audit hook records file opens and directory listings so the
tests assert the sentinel is never read, not merely that the call happened to succeed.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from tests.image_helpers import write_rgb_image
from tests.training.test_trainer_lifecycle import _FakeMethod, _training
from virtual_staining.config.data import (
    InputConfig,
    PatchingConfig,
    PreprocessingConfig,
    SplitConfig,
)
from virtual_staining.data import manifest as manifest_module
from virtual_staining.data.builder import DatasetBuilder
from virtual_staining.data.layout import DatasetLayout
from virtual_staining.data.manifest import (
    MANIFEST_SCHEMA_VERSION,
    DatasetManifest,
    ManifestMetadata,
)
from virtual_staining.data.slide_sets import SlideAsset, SlideSet
from virtual_staining.evaluation.evaluator import (
    EvaluationSample,
    evaluate_pair,
    evaluate_samples,
)
from virtual_staining.experiment import session as session_module
from virtual_staining.experiment.run_layout import RunLayout
from virtual_staining.inference.single import (
    InferenceRuntime,
    PredictionContract,
    SingleInferenceResult,
    run_image_path_inference,
)
from virtual_staining.models.generator import ConcatUNetGenerator
from virtual_staining.training.trainer import Trainer

_ACCESSED: list[Path] | None = None


def _audit(event: str, args: tuple[object, ...]) -> None:
    if _ACCESSED is None or event not in {"open", "os.listdir", "os.scandir"} or not args:
        return
    path = args[0]
    if isinstance(path, str | bytes | os.PathLike):
        _ACCESSED.append(Path(os.path.abspath(os.fsdecode(path))))


sys.addaudithook(_audit)


@contextmanager
def _recorded_access() -> Iterator[list[Path]]:
    global _ACCESSED
    _ACCESSED = []
    try:
        yield _ACCESSED
    finally:
        _ACCESSED = None


def _sentinel(root: Path) -> Path:
    """Predecessor artifacts a standalone call must neither read nor modify."""
    for relative in (
        "dataset/manifests/manifest.csv",
        "dataset/manifests/manifest_metadata.json",
        "dataset/metadata/dataset_fingerprint.json",
        "run/metadata/run.json",
        "run/config/train/resolved.yaml",
        "run/checkpoints/ep000.pth",
        "inputs.csv",
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("sentinel: must not be read\n", encoding="utf-8")
    return root


def _tree(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _assert_untouched(accessed: list[Path], sentinel: Path, before: dict[str, bytes]) -> None:
    root = Path(os.path.abspath(sentinel))
    assert [path for path in accessed if path.is_relative_to(root)] == []
    assert _tree(sentinel) == before


@pytest.fixture
def no_tracked_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail loudly if a standalone call opens a tracked session or a prepared manifest."""

    def forbidden(*_args: object, **_kwargs: object) -> None:
        pytest.fail("standalone stage primitive reached tracked/manifest machinery")

    monkeypatch.setattr(session_module.ExperimentSession, "__init__", forbidden)
    monkeypatch.setattr(manifest_module, "load_manifest_or_raise", forbidden)
    monkeypatch.setattr(DatasetManifest, "from_csv", forbidden)


def test_dataset_builder_needs_only_preprocessing_config_and_slide_sets(tmp_path: Path) -> None:
    sentinel = _sentinel(tmp_path / "sentinel")
    before = _tree(sentinel)
    sources = tmp_path / "sources"
    image = np.full((8, 16, 3), 100, dtype=np.uint8)
    slide_sets = []
    for set_id in ("set-1", "set-2"):
        (sources / set_id).mkdir(parents=True)
        for name in ("lf", "af", "target"):
            Image.fromarray(image).save(sources / set_id / f"{name}.png")
        slide_sets.append(
            SlideSet(
                set_id,
                (
                    SlideAsset("LF", sources / set_id / "lf.png", already_aligned=True),
                    SlideAsset("AF", sources / set_id / "af.png", already_aligned=True),
                ),
                SlideAsset("target", sources / set_id / "target.png", already_aligned=True),
                "LF",
            )
        )
    dataset_root = tmp_path / "prepared"
    config = PreprocessingConfig(
        dataset_root=dataset_root,
        # The inventory is only read when resolving SlideSets from YAML; here they are given.
        inputs=InputConfig(sentinel / "inputs.csv", ("LF", "AF"), "LF", "target"),
        patching=PatchingConfig(patch_size=(8, 8), grid_movement=(8, 8), margin=0),
        split=SplitConfig(unit="set", train=0.5, val=0.5, test=0.0),
    )

    with _recorded_access() as accessed:
        result = DatasetBuilder(config, tuple(slide_sets)).run_all()

    _assert_untouched(accessed, sentinel, before)
    layout = DatasetLayout(dataset_root)
    metadata = ManifestMetadata(MANIFEST_SCHEMA_VERSION, ("LF", "AF"), "LF", "target")
    manifest = DatasetManifest.from_csv(layout.manifest_path, dataset_root, metadata)
    manifest.validate(check_files_exist=True, require_splits={"train", "val"})
    assert result.train_count + result.val_count == len(manifest) == 4
    assert layout.manifest_metadata_path.is_file()
    assert layout.dataset_fingerprint_path.is_file()
    # Every write lands under the configured dataset root; sources stay read-only.
    assert sorted(path.name for path in tmp_path.iterdir()) == ["prepared", "sentinel", "sources"]


def test_trainer_runs_from_loaders_and_run_layout_without_session(
    tmp_path: Path, no_tracked_run: None
) -> None:
    sentinel = _sentinel(tmp_path / "sentinel")
    before = _tree(sentinel)
    training = _training(epochs=2)
    layout = RunLayout(tmp_path / "run")
    loader = DataLoader([1.0, 1.0, 1.0, 1.0], batch_size=training.batch_size)  # pyright: ignore[reportArgumentType]

    def build() -> Trainer:
        return Trainer(
            training,
            layout,
            _FakeMethod(training),  # pyright: ignore[reportArgumentType]
            loader,
            loader,
            torch.device("cpu"),
        )

    with _recorded_access() as accessed:
        result = build().train(seed=0)
        resumed_epoch = build().resume("latest")

    _assert_untouched(accessed, sentinel, before)
    assert result.final_epoch == 1
    assert result.best_checkpoint_path is not None and result.best_checkpoint_path.is_file()
    assert len(layout.epochs_csv.read_text(encoding="utf-8").splitlines()) == 3
    assert sorted(path.name for path in layout.checkpoints_dir.iterdir()) == [
        "best.json",
        "ep000.pth",
        "ep001.pth",
    ]
    assert resumed_epoch == 2
    # No tracked-run layer: no run.json, stage records, events, or config snapshots.
    assert not layout.metadata_dir.exists()
    assert not layout.config_dir.exists()


def test_image_path_inference_runs_from_runtime_factory_and_named_paths(
    tmp_path: Path, no_tracked_run: None
) -> None:
    sentinel = _sentinel(tmp_path / "sentinel")
    before = _tree(sentinel)
    generator = ConcatUNetGenerator(("LF",), base_channels=4).eval()
    source = write_rgb_image(tmp_path / "images" / "sample.png", size=(32, 32))
    output = tmp_path / "predictions" / "sample_generated.png"

    def runtime_factory() -> InferenceRuntime:
        # checkpoint_path is identity metadata carried by the injected runtime, never read.
        return InferenceRuntime(
            predictor=generator,
            contract=PredictionContract(("LF",), (32, 32)),
            device=torch.device("cpu"),
            checkpoint_path=sentinel / "run/checkpoints/ep000.pth",
            default_single_output_dir=sentinel / "absent_single",
            default_directory_output_dir=sentinel / "absent_directory",
        )

    with _recorded_access() as accessed:
        result = run_image_path_inference(runtime_factory, {"LF": source}, output, mode="resize")

    _assert_untouched(accessed, sentinel, before)
    assert isinstance(result, SingleInferenceResult)
    assert result.output_path == output
    with Image.open(output) as image:
        assert image.size == (32, 32)


def test_evaluation_runs_from_explicit_records_and_output_dir(
    tmp_path: Path, no_tracked_run: None
) -> None:
    sentinel = _sentinel(tmp_path / "sentinel")
    before = _tree(sentinel)
    target = write_rgb_image(tmp_path / "pairs" / "a_target.png", color=(10, 20, 30))
    generated = write_rgb_image(tmp_path / "pairs" / "a_generated.png", color=(12, 20, 30))
    samples = [
        EvaluationSample("a", "S1", target, generated),
        EvaluationSample("b", "S1", target, tmp_path / "pairs" / "b_missing.png"),
    ]
    output_dir = tmp_path / "evaluation"

    with _recorded_access() as accessed:
        metrics, shape = evaluate_pair(target, generated)
        result = evaluate_samples(samples, output_dir, input_failures="permissive")

    _assert_untouched(accessed, sentinel, before)
    assert shape == (16, 16, 3)
    assert metrics["mae"].status == "finite" and (metrics["mae"].value or 0) > 0
    assert (result.num_evaluated, result.num_excluded) == (1, 1)
    assert result.coverage_rows[1]["reason"] == "missing_generated"
    assert {path.name for path in output_dir.iterdir()} == {
        "per_image_metrics.csv",
        "summary.csv",
        "coverage.csv",
        "evaluation_result.json",
    }


def test_stage_primitive_imports_have_no_runtime_side_effects(tmp_path: Path) -> None:
    code = """
import os, sys
import torch
import openslide

def forbidden(*args, **kwargs):
    raise AssertionError("import side effect")

torch.load = forbidden
openslide.OpenSlide = forbidden
opened = []
sys.addaudithook(
    lambda event, args: opened.append(args[0])
    if event in {"open", "os.listdir", "os.scandir"}
    and isinstance(args[0], str) and os.path.abspath(args[0]).startswith(os.getcwd())
    else None
)

from virtual_staining.config.data import PreprocessingConfig
from virtual_staining.config.training import TrainingConfig
from virtual_staining.data.builder import DatasetBuilder
from virtual_staining.data.slide_sets import SlideAsset, SlideSet
from virtual_staining.evaluation.evaluator import EvaluationSample, evaluate_pair, evaluate_samples
from virtual_staining.experiment.run_layout import RunLayout
from virtual_staining.inference.single import InferenceRuntime, run_image_path_inference
from virtual_staining.training.trainer import Trainer

assert not torch.cuda.is_initialized()
assert not {"nicegui", "marimo"} & {name.split(".")[0] for name in sys.modules}
assert "virtual_staining.experiment.session" not in sys.modules
assert opened == [], opened
assert os.listdir(".") == []
"""
    completed = subprocess.run(
        [sys.executable, "-P", "-c", code],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
