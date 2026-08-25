from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.config_helpers import write_yaml
from tests.manifest_helpers import make_manifest_record
from virtual_staining.applications import train as train_app
from virtual_staining.applications.train import _requires_foreground_masks
from virtual_staining.config.run import RunConfig
from virtual_staining.data.manifest import DatasetManifest, ManifestMetadata


class _InertSession:
    def __enter__(self) -> SimpleNamespace:
        return SimpleNamespace()

    def __exit__(self, *_args: object) -> bool:
        return False


def _manifest(tmp_path: Path, *, target_modality: str, splits: tuple[str, ...]) -> DatasetManifest:
    records = []
    for index, split in enumerate(splits):
        sample_id = f"{index * 256:05}_00000"
        record = make_manifest_record(
            sample_id,
            split,
            input_paths={
                "LF": Path(f"splits/{split}/{sample_id}_input__LF.tif"),
                "AF": Path(f"splits/{split}/{sample_id}_input__AF.tif"),
            },
            target_path=Path(f"splits/{split}/{sample_id}__target.tif"),
        )
        records.append(record)
        for path in (*record.input_paths.values(), record.target_path):
            full_path = tmp_path / "dataset" / path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.touch()
    return DatasetManifest(
        tuple(records),
        tmp_path / "dataset",
        ManifestMetadata("3.0", ("LF", "AF"), "LF", target_modality),
    )


def _yaml(root: Path, *, target: str = "target", target_modality: str = "target") -> Path:
    return write_yaml(
        root / "run.yaml",
        f"""
dataset_root: {root / "dataset"}
results_path: {root / "results"}
run_name: run
image_size: [16, 16]
model:
  inputs: [LF, AF]
  target: {target}
preprocessing:
  inputs:
    inventory: inputs/slides.csv
    modalities: [LF, AF]
    reference: LF
    target_modality: {target_modality}
  split:
    unit: set
    train: 0.8
    val: 0.1
    test: 0.1
training:
  epochs: 1
  losses:
    generator:
      - name: l1
        weight: 1.0
    discriminator: []
""",
    )


def test_training_config_uses_named_model_contract(tmp_path: Path) -> None:
    config = RunConfig.from_yaml(_yaml(tmp_path))
    assert config.model.inputs == ("LF", "AF")
    assert config.model.target == "target"
    assert config.training is not None
    assert _requires_foreground_masks(config) is False


def test_run_config_rejects_model_target_mismatch(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="target_modality"):
        RunConfig.from_yaml(_yaml(tmp_path, target="other"))


def test_train_rejects_model_target_mismatch_before_manifest_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _yaml(tmp_path, target="other", target_modality="other")
    config = RunConfig.from_yaml(config_path)
    manifest = _manifest(tmp_path, target_modality="target", splits=("train", "val"))
    monkeypatch.setattr(train_app.ExperimentSession, "open", lambda **_kwargs: _InertSession())
    monkeypatch.setattr(train_app, "load_manifest_or_raise", lambda _project: manifest)

    with pytest.raises(ValueError, match="^model.target must equal manifest target modality$"):
        train_app.train(config, config_path)


def test_train_requires_validation_split_before_dataset_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _yaml(tmp_path)
    config = RunConfig.from_yaml(config_path)
    manifest = _manifest(tmp_path, target_modality="target", splits=("train",))
    monkeypatch.setattr(train_app.ExperimentSession, "open", lambda **_kwargs: _InertSession())
    monkeypatch.setattr(train_app, "load_manifest_or_raise", lambda _project: manifest)

    with pytest.raises(ValueError, match=r"^Manifest has no records for required split 'val'$"):
        train_app.train(config, config_path)
