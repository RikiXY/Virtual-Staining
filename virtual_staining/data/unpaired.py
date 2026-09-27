from __future__ import annotations

import glob
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from virtual_staining.data.consumption import (
    AssetRow,
    enrich_with_groups,
    load_group_metadata,
    relative_locator,
)
from virtual_staining.split_contract import DatasetSplit
from virtual_staining.utils.image_io import VALID_IMAGE_EXTENSIONS

SPLIT_PLACEHOLDER = "{split}"


def resolve_domain_images(spec: str, split: DatasetSplit, dataset_root: Path) -> tuple[Path, ...]:
    """Resolve one ``data.domains`` entry to its sorted images for ``split``.

    ``spec`` is either a directory root holding ``train/``, ``val/``, and ``test/``
    (searched recursively) or a path/glob containing the literal ``{split}``.
    Relative specs resolve against ``dataset_root``.
    """
    if SPLIT_PLACEHOLDER in spec:
        pattern = spec.replace(SPLIT_PLACEHOLDER, split)
    else:
        split_dir = Path(spec) / split
        if not (dataset_root / split_dir).is_dir():
            raise FileNotFoundError(
                f"Domain directory {spec!r} has no {split!r} split at {dataset_root / split_dir}"
            )
        pattern = str(split_dir / "**" / "*")
    full_pattern = str(dataset_root / pattern)
    paths = sorted(
        path
        for path in map(Path, glob.glob(full_pattern, recursive=True))
        if path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTENSIONS
    )
    if not paths:
        raise ValueError(
            f"Domain {spec!r} matched no supported images for split {split!r} "
            f"(pattern {full_pattern!r}; extensions {sorted(VALID_IMAGE_EXTENSIONS)})"
        )
    return tuple(paths)


def resolve_domain_collections(
    specs: Mapping[str, str],
    dataset_root: Path,
    *,
    splits: Sequence[DatasetSplit],
    roles: Mapping[str, str],
    group_metadata: Path | None = None,
) -> tuple[dict[tuple[DatasetSplit, str], tuple[Path, ...]], list[AssetRow], tuple[AssetRow, ...]]:
    """Resolve each selected domain/split collection exactly once.

    Returns the resolved paths (to hand to the consumer), their snapshot rows enriched with
    any explicit ``group_metadata`` sidecar identities, and the sidecar entries themselves
    (for split-leakage checks). Locators are relative to ``dataset_root``; a domain resolving
    outside it is rejected.
    """
    groups = (
        load_group_metadata(
            group_metadata if group_metadata.is_absolute() else dataset_root / group_metadata
        )
        if group_metadata is not None
        else ()
    )
    paths: dict[tuple[DatasetSplit, str], tuple[Path, ...]] = {}
    rows: list[AssetRow] = []
    for split in splits:
        for domain, role in roles.items():
            resolved = resolve_domain_images(specs[domain], split, dataset_root)
            paths[split, domain] = resolved
            rows.extend(
                AssetRow(
                    root="dataset",
                    locator=relative_locator(dataset_root, path),
                    role=role,
                    domain=domain,
                    split=split,
                )
                for path in resolved
            )
    return paths, enrich_with_groups(rows, groups), groups


class UnpairedImageDataset(Dataset):
    """Draw one image from each of two independent domains per sample.

    Length is ``max(len(A), len(B))``; the shorter domain wraps around. With
    ``pairing_seed`` set, the domain-B index is a pure function of
    ``(pairing_seed, epoch, index)`` so pairings vary per epoch yet are reproducible
    regardless of worker count. Without it, traversal is deterministic (``index`` in
    both domains) for validation and test.
    """

    def __init__(
        self,
        paths_a: Sequence[Path],
        paths_b: Sequence[Path],
        *,
        transform: Callable[[Image.Image], Any],
        pairing_seed: int | None = None,
    ) -> None:
        if not paths_a or not paths_b:
            raise ValueError("UnpairedImageDataset requires non-empty domain A and B image lists")
        self.paths_a = tuple(paths_a)
        self.paths_b = tuple(paths_b)
        self.transform = transform
        self.pairing_seed = pairing_seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return max(len(self.paths_a), len(self.paths_b))

    def domain_b_index(self, idx: int) -> int:
        if self.pairing_seed is None:
            return idx % len(self.paths_b)
        rng = np.random.default_rng([self.pairing_seed, self.epoch, idx])
        return int(rng.integers(len(self.paths_b)))

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        path_a = self.paths_a[idx % len(self.paths_a)]
        path_b = self.paths_b[self.domain_b_index(idx)]
        return {
            "domain_a": self.transform(Image.open(path_a).convert("RGB")),
            "domain_b": self.transform(Image.open(path_b).convert("RGB")),
            "path_a": str(path_a),
            "path_b": str(path_b),
        }
