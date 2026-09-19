from __future__ import annotations

import cv2
import numpy as np


def foreground_ratios(masks: dict[str, np.ndarray]) -> dict[str, float]:
    if not masks:
        raise ValueError("At least one mask is required")

    ratios = {name: float(cv2.countNonZero(mask) / mask.size) for name, mask in masks.items()}
    ratios["all"] = min(ratios.values())

    first = next(iter(masks.values()))
    intersection = first.copy()
    union = first.copy()
    for mask in list(masks.values())[1:]:
        intersection = cv2.bitwise_and(intersection, mask)
        union = cv2.bitwise_or(union, mask)

    ratios["intersection"] = float(cv2.countNonZero(intersection) / intersection.size)
    ratios["union"] = float(cv2.countNonZero(union) / union.size)
    return ratios


def _compute_white_stats(
    img: np.ndarray,
    white_threshold: int = 245,
    *,
    largest_component_threshold: float | None = None,
) -> tuple[float, float]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    white_mask = gray >= white_threshold
    white_ratio = float(np.mean(white_mask))

    if largest_component_threshold is not None and white_ratio <= largest_component_threshold:
        return white_ratio, 0.0

    white_mask_u8 = white_mask.astype(np.uint8) * 255
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
        white_mask_u8,
        connectivity=8,
    )

    if num_labels <= 1:
        return white_ratio, 0.0

    largest_area = int(np.max(stats[1:, cv2.CC_STAT_AREA]))
    return white_ratio, float(largest_area / white_mask_u8.size)


def is_valid_patch_pair(
    source_img: np.ndarray,
    target_img: np.ndarray,
    source_mask: np.ndarray,
    target_mask: np.ndarray,
    min_foreground_ratio: float,
    max_white_ratio: float,
    white_threshold: int,
    max_largest_white_component_ratio: float,
) -> tuple[bool, dict[str, float | list[str]]]:
    source_foreground_ratio = cv2.countNonZero(source_mask) / source_mask.size
    target_foreground_ratio = cv2.countNonZero(target_mask) / target_mask.size

    source_white_ratio, source_largest_white_component_ratio = _compute_white_stats(
        source_img,
        white_threshold,
        largest_component_threshold=max_largest_white_component_ratio,
    )
    target_white_ratio, target_largest_white_component_ratio = _compute_white_stats(
        target_img,
        white_threshold,
        largest_component_threshold=max_largest_white_component_ratio,
    )

    reasons: list[str] = []
    if source_foreground_ratio < min_foreground_ratio:
        reasons.append("low_source_foreground")
    if target_foreground_ratio < min_foreground_ratio:
        reasons.append("low_target_foreground")
    if source_white_ratio > max_white_ratio:
        reasons.append("high_source_white_ratio")
    if target_white_ratio > max_white_ratio:
        reasons.append("high_target_white_ratio")
    if source_largest_white_component_ratio > max_largest_white_component_ratio:
        reasons.append("high_source_largest_white_component_ratio")
    if target_largest_white_component_ratio > max_largest_white_component_ratio:
        reasons.append("high_target_largest_white_component_ratio")

    debug_info = {
        "source_foreground_ratio": source_foreground_ratio,
        "target_foreground_ratio": target_foreground_ratio,
        "source_white_ratio": source_white_ratio,
        "target_white_ratio": target_white_ratio,
        "source_largest_white_component_ratio": source_largest_white_component_ratio,
        "target_largest_white_component_ratio": target_largest_white_component_ratio,
        "reasons": reasons,
    }
    return len(reasons) == 0, debug_info
