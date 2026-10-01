from __future__ import annotations

from typing import Literal

TRAIN_SPLIT: Literal["train"] = "train"
VAL_SPLIT: Literal["val"] = "val"
TEST_SPLIT: Literal["test"] = "test"
DISCARDED_SPLIT: Literal["discarded"] = "discarded"

DatasetSplit = Literal["train", "val", "test"]
ManifestSplit = Literal["train", "val", "test", "discarded"]

DATASET_SPLITS: tuple[DatasetSplit, ...] = (TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT)
MANIFEST_SPLITS: tuple[ManifestSplit, ...] = (*DATASET_SPLITS, DISCARDED_SPLIT)
