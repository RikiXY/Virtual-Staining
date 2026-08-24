from __future__ import annotations

from pathlib import Path

import numpy as np

from virtual_staining.data import builder as builder_module
from virtual_staining.data.builder import AlignmentResult, DatasetBuilder
from virtual_staining.data.config import (
    InputConfig,
    PatchingConfig,
    PreprocessingConfig,
    SplitConfig,
)
from virtual_staining.data.slide_sets import SlideAsset, SlideSet


def _config(root: Path) -> PreprocessingConfig:
    return PreprocessingConfig(
        dataset_root=root,
        inputs=InputConfig(root / "inputs.csv", ("LF", "AF"), "LF", "target"),
        patching=PatchingConfig(patch_size=(8, 8), grid_movement=(8, 8), margin=0),
        split=SplitConfig(unit="set", train=1.0, val=0.0, test=0.0),
    )


def _slide_set(root: Path) -> SlideSet:
    (root / "raw").mkdir(parents=True)
    for name in ("lf.png", "af.png", "target.png"):
        (root / "raw" / name).write_bytes(b"image")
    return SlideSet(
        "set-1",
        (
            SlideAsset("LF", Path("raw/lf.png"), already_aligned=True),
            SlideAsset("AF", Path("raw/af.png"), already_aligned=True),
        ),
        SlideAsset("target", Path("raw/target.png"), already_aligned=True),
        "LF",
    )


def test_builder_emits_dynamic_manifest_and_set_metadata(tmp_path, monkeypatch) -> None:
    config = _config(tmp_path)
    slide_set = _slide_set(tmp_path)
    identity = AlignmentResult("identity", np.eye(2, 3), {"method": "identity"})

    class FakeProcessor:
        def __init__(self, config, slide_set, assigned_split=None):
            self.inputs = {
                name: type("State", (), {"alignment": identity})()
                for name in config.inputs.modalities
            }
            self.target = type("State", (), {"alignment": identity})()

        def compute_masks(self):
            pass

        def align(self):
            pass

        def stream_patches(self):
            return (
                [
                    {
                        "sample_id": "set-1__x00000000_y00000000",
                        "split": "train",
                        "inputs": {"LF": "lf.png", "AF": "af.png"},
                        "target": "target.png",
                        "foreground_mask": None,
                        "x": 0,
                        "y": 0,
                    }
                ],
                [],
            )

        def close(self):
            pass

    monkeypatch.setattr(builder_module, "SlideSetProcessor", FakeProcessor)
    result = DatasetBuilder(config, (slide_set,), {"schema_version": "3.0"}).run_all()
    assert result.train_count == 1
    header = (tmp_path / "manifests" / "manifest.csv").read_text(encoding="utf-8").splitlines()[0]
    assert header.split(",")[:6] == [
        "sample_id",
        "set_id",
        "split",
        "input__LF",
        "input__AF",
        "target_path",
    ]
    assert (tmp_path / "manifests" / "slide_sets.csv").exists()
    assert (tmp_path / "metadata" / "excluded_sets.csv").exists()


def test_discarded_records_use_modality_paths(tmp_path) -> None:
    config = _config(tmp_path)
    builder = DatasetBuilder(config, (_slide_set(tmp_path),), {"schema_version": "3.0"})
    builder._current_set_id = "set-1"
    records = builder._records(
        [
            {
                "sample_id": "x",
                "split": "discarded",
                "inputs": {"LF": "x.png", "AF": "y.png"},
                "target": "t.png",
                "foreground_mask": None,
                "x": 0,
                "y": 0,
            }
        ],
        discarded=True,
    )
    assert records[0].input_paths["LF"] == Path("discarded_patches/set-1/LF/x.png")
    assert records[0].target_path == Path("discarded_patches/set-1/target/t.png")
