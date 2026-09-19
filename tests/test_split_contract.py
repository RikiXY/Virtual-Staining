from __future__ import annotations

from typing import get_args

from virtual_staining.split_contract import (
    DATASET_SPLITS,
    DISCARDED_SPLIT,
    MANIFEST_SPLITS,
    TEST_SPLIT,
    TRAIN_SPLIT,
    VAL_SPLIT,
    DatasetSplit,
    ManifestSplit,
)


def test_dataset_split_vocabulary_is_canonical() -> None:
    assert DATASET_SPLITS == (TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT) == ("train", "val", "test")
    assert set(get_args(DatasetSplit)) == set(DATASET_SPLITS)


def test_manifest_split_vocabulary_adds_only_discarded_state() -> None:
    assert (*DATASET_SPLITS, DISCARDED_SPLIT) == MANIFEST_SPLITS
    assert set(get_args(ManifestSplit)) == set(MANIFEST_SPLITS)
    assert DISCARDED_SPLIT not in DATASET_SPLITS
