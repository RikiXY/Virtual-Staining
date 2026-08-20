from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from virtual_staining.applications.prepare import prepare
from virtual_staining.config.run import RunConfig
from virtual_staining.data.builder import DatasetBuilder
from virtual_staining.data.config import (
    AlignmentConfig,
    InputConfig,
    MaskConfig,
    PreprocessingConfig,
    SplitConfig,
)
from virtual_staining.data.manifest import DatasetManifest
from virtual_staining.data.pairs import SlidePair, load_pair_inventory
from virtual_staining.data.splitting import assign_group_splits, group_id_for_pair


def _image(path: Path, value: int = 100) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((40, 40, 3), value, dtype=np.uint8)).save(path)


def test_pair_inventory_is_strict_and_row_order_independent(tmp_path: Path) -> None:
    for name in ("s1.png", "t1.png", "s2.png", "t2.png", "m2.png"):
        _image(tmp_path / "raw" / name)
    header = "pair_id,source_path,target_path,already_aligned,shared_mask_path,patient_id\n"
    rows = [
        "P002,raw/s2.png,raw/t2.png,true,raw/m2.png,PT2\n",
        "P001,raw/s1.png,raw/t1.png,false,,PT1\n",
    ]
    inventory = tmp_path / "pairs.csv"
    inventory.write_text(header + "".join(rows), encoding="utf-8")
    pairs = load_pair_inventory(inventory, tmp_path)
    assert [pair.pair_id for pair in pairs] == ["P001", "P002"]
    assert pairs[0].shared_mask_path is None
    assert pairs[1].shared_mask_path == Path("raw/m2.png")

    inventory.write_text(header + "".join(reversed(rows)), encoding="utf-8")
    reordered = load_pair_inventory(inventory, tmp_path)
    assert reordered == pairs


def test_pair_inventory_rejects_traversal_and_partial_masks(tmp_path: Path) -> None:
    inventory = tmp_path / "pairs.csv"
    inventory.write_text(
        "pair_id,source_path,target_path\nP001,../source.png,target.png\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="inside dataset_root"):
        load_pair_inventory(inventory, tmp_path)


def test_group_split_is_leakage_safe_and_row_order_independent() -> None:
    pairs = tuple(
        SlidePair(
            f"P{index}",
            Path(f"s{index}.png"),
            Path(f"t{index}.png"),
            patient_id=f"PT{index // 2}",
        )
        for index in range(8)
    )
    first = assign_group_splits(pairs, unit="patient", ratios=(0.5, 0.25, 0.25), seed=42)
    second = assign_group_splits(
        tuple(reversed(pairs)), unit="patient", ratios=(0.5, 0.25, 0.25), seed=42
    )
    assert first == second
    for patient in {pair.patient_id for pair in pairs}:
        assert (
            len(
                {
                    first[pair.pair_id]
                    for pair in pairs
                    if group_id_for_pair(pair, "patient") == patient
                }
            )
            == 1
        )


def test_group_split_rejects_unrepresentable_scientific_splits() -> None:
    pairs = (SlidePair("P1", Path("s"), Path("t")),)
    with pytest.raises(ValueError, match="Cannot place"):
        assign_group_splits(pairs, unit="pair", ratios=(0.8, 0.1, 0.1), seed=0)


def test_multi_pair_builder_writes_v2_without_cross_pair_cleanup(tmp_path: Path) -> None:
    pairs: list[SlidePair] = []
    for index in range(3):
        source = Path(f"raw/source/S{index}.png")
        target = Path(f"raw/target/T{index}.png")
        _image(tmp_path / source, 80 + index)
        _image(tmp_path / target, 120 + index)
        pairs.append(
            SlidePair(f"P{index}", source, target, already_aligned=True, patient_id=f"PT{index}")
        )

    config = PreprocessingConfig(
        dataset_root=tmp_path,
        inputs=InputConfig(Path("inputs/pairs.csv"), "autofluorescence", "H&E"),
        masks=MaskConfig(generation="never"),
        alignment=AlignmentConfig(mode="auto"),
        split=SplitConfig(unit="pair", train=1 / 3, val=1 / 3, test=1 / 3, seed=7),
        patch_size=(16, 16),
        grid_movement=(16, 16),
        margin=4,
        foreground_enabled=False,
    )
    result = DatasetBuilder(config, tuple(pairs)).run_all()
    assert (result.train_count, result.val_count, result.test_count) == (4, 4, 4)

    manifest = DatasetManifest.from_csv(tmp_path / "manifests/manifest.csv", tmp_path)
    assert manifest.schema_version == "2.0"
    assert len({record.sample_id for record in manifest.records}) == 12
    assert {record.x for record in manifest.records} == {4, 20}
    assert {record.y for record in manifest.records} == {4, 20}
    assert all(record.foreground_mask_path is None for record in manifest.records)
    for pair in pairs:
        assert any(
            (tmp_path / "splits" / split / pair.pair_id).is_dir()
            for split in ("train", "val", "test")
        )

    with (tmp_path / "manifests/pairs.csv").open(newline="", encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == 3


def test_prepare_resolves_inventory_once_and_reuses_v2_dataset(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    _image(dataset / "raw/source.png", 80)
    _image(dataset / "raw/target.png", 120)
    (dataset / "inputs").mkdir()
    (dataset / "inputs/pairs.csv").write_text(
        "pair_id,source_path,target_path,already_aligned\nP1,raw/source.png,raw/target.png,true\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "run.yaml"
    config_path.write_text(
        f"""\
dataset_root: {dataset}
results_path: {tmp_path / "results"}
run_name: inventory
image_size: [16, 16]
preprocessing:
  inputs:
    inventory: inputs/pairs.csv
    source_modality: AF
    target_modality: H&E
  patching:
    patch_size: [16, 16]
    grid_movement: [16, 16]
    margin: 4
  masks:
    generation: never
  alignment:
    mode: auto
  filtering:
    foreground:
      enabled: false
  split:
    unit: pair
    train: 1.0
    val: 0.0
    test: 0.0
    seed: 3
""",
        encoding="utf-8",
    )
    config = RunConfig.from_yaml(config_path)
    first = prepare(config, config_path)
    second = prepare(config, config_path)
    assert first.reused is False
    assert second.reused is True
    assert (
        DatasetManifest.from_csv(dataset / "manifests/manifest.csv", dataset).schema_version
        == "2.0"
    )
