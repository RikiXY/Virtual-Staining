from __future__ import annotations

import numpy as np

from virtual_staining.data.patching import iter_patch_origins, mask_window_for_patch


def test_iter_patch_origins_preserves_grid_and_margin_policy() -> None:
    origins = tuple(
        iter_patch_origins(
            image_size=(20, 16),
            patch_size=(8, 8),
            grid_movement=(6, 4),
            margin=2,
        )
    )

    assert origins == ((2, 2), (2, 6), (8, 2), (8, 6))


def test_iter_patch_origins_yields_nothing_when_patch_does_not_fit() -> None:
    origins = tuple(
        iter_patch_origins(
            image_size=(8, 8),
            patch_size=(16, 16),
            grid_movement=(4, 4),
            margin=0,
        )
    )

    assert origins == ()


def test_mask_window_for_patch_maps_full_resolution_patch_to_mask_space() -> None:
    mask = np.zeros((5, 10), dtype=np.uint8)
    mask[1:4, 2:7] = 255

    window = mask_window_for_patch(
        mask,
        (20, 40, 3),
        x=8,
        y=4,
        width=12,
        height=8,
    )

    assert window.shape == (2, 3)
    assert np.all(window == 255)
