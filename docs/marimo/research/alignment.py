import marimo

__generated_with = "0.23.8"
app = marimo.App(width="full")


@app.cell
def _():
    from pathlib import Path

    import cv2
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from PIL import Image

    from virtual_staining.data.preprocessing import (
        ALLOWED_MASK_STRATEGIES,
        LOWE_RATIO_THRESHOLD,
        MASK_PARAMETER_GRID,
        MIN_INLIER_RATIO,
        MIN_INLIERS,
        MIN_STD_DEV,
        RANSAC_CONFIDENCE,
        RANSAC_MAX_ITERS,
        RANSAC_REFINE_ITERS,
        RANSAC_REPROJECTION_THRESHOLD,
        calculate_mask_by_strategy,
    )

    return (
        ALLOWED_MASK_STRATEGIES,
        Image,
        LOWE_RATIO_THRESHOLD,
        MASK_PARAMETER_GRID,
        MIN_INLIERS,
        MIN_INLIER_RATIO,
        MIN_STD_DEV,
        Path,
        RANSAC_CONFIDENCE,
        RANSAC_MAX_ITERS,
        RANSAC_REFINE_ITERS,
        RANSAC_REPROJECTION_THRESHOLD,
        calculate_mask_by_strategy,
        cv2,
        mo,
        np,
        pd,
        plt,
    )


@app.cell
def _(Path):
    REPO_ROOT = Path(__file__).resolve().parents[3]
    DEFAULT_SOURCE_IMAGE = "examples/label_free_0.png"
    DEFAULT_TARGET_IMAGE = "examples/stained_0.png"

    DEMO_MAX_SIDE = 900
    CLAHE_TILE_GRID = (8, 8)
    PACKAGE_CLAHE_CLIP_LIMIT = 18.0
    CLAHE_COMPARISON_CLIP_LIMITS = (2.0, 4.0, 6.0, 8.0, 16.0, PACKAGE_CLAHE_CLIP_LIMIT)
    CONNECTED_COMPONENT_THRESHOLD = 230
    COMPONENT_MIN_SIDE = 100
    COMPONENT_BACKGROUND_STD_DEV = 10.0
    COMPONENT_TOP_COUNT = 10
    SIFT_NFEATURES = 10000
    KEYPOINT_PREVIEW_LIMIT = 120
    MATCH_PREVIEW_LIMIT = 80
    MATCH_PREVIEW_LINE_THICKNESS = 3
    DEFAULT_MATCH_STRATEGY = "connected_components"
    return (
        CLAHE_COMPARISON_CLIP_LIMITS,
        CLAHE_TILE_GRID,
        CONNECTED_COMPONENT_THRESHOLD,
        DEFAULT_MATCH_STRATEGY,
        DEFAULT_SOURCE_IMAGE,
        DEFAULT_TARGET_IMAGE,
        DEMO_MAX_SIDE,
        KEYPOINT_PREVIEW_LIMIT,
        MATCH_PREVIEW_LINE_THICKNESS,
        MATCH_PREVIEW_LIMIT,
        COMPONENT_BACKGROUND_STD_DEV,
        COMPONENT_MIN_SIDE,
        COMPONENT_TOP_COUNT,
        PACKAGE_CLAHE_CLIP_LIMIT,
        REPO_ROOT,
        SIFT_NFEATURES,
    )


@app.cell
def _(mo):
    mo.output.replace(
        mo.md(
            "# Alignment Research Demo\n\n"
            "This app walks through contrast normalization, mask construction, SIFT "
            "keypoints, descriptor matching, and lightweight alignment diagnostics "
            "using committed demo images by default. It does not inspect prepared "
            "datasets, run dataset preparation, write masks, save previews, or create "
            "dataset artifacts. Full-frame warps below are only in-memory diagnostics "
            "used to compute residual previews."
        )
    )
    return


@app.cell
def _(DEFAULT_SOURCE_IMAGE, DEFAULT_TARGET_IMAGE, mo):
    source_image_input = mo.ui.text(
        label="source_demo_image",
        value=DEFAULT_SOURCE_IMAGE,
        full_width=True,
    )
    target_image_input = mo.ui.text(
        label="target_demo_image",
        value=DEFAULT_TARGET_IMAGE,
        full_width=True,
    )

    mo.vstack(
        [
            mo.md("## Inputs"),
            mo.hstack([source_image_input, target_image_input], widths="equal"),
        ]
    )
    return source_image_input, target_image_input


@app.cell
def _(
    CLAHE_TILE_GRID,
    CONNECTED_COMPONENT_THRESHOLD,
    DEFAULT_MATCH_STRATEGY,
    DEMO_MAX_SIDE,
    Image,
    KEYPOINT_PREVIEW_LIMIT,
    LOWE_RATIO_THRESHOLD,
    MATCH_PREVIEW_LINE_THICKNESS,
    MATCH_PREVIEW_LIMIT,
    MIN_INLIERS,
    MIN_INLIER_RATIO,
    COMPONENT_BACKGROUND_STD_DEV,
    COMPONENT_MIN_SIDE,
    COMPONENT_TOP_COUNT,
    PACKAGE_CLAHE_CLIP_LIMIT,
    Path,
    RANSAC_CONFIDENCE,
    RANSAC_MAX_ITERS,
    RANSAC_REFINE_ITERS,
    RANSAC_REPROJECTION_THRESHOLD,
    SIFT_NFEATURES,
    calculate_mask_by_strategy,
    cv2,
    mo,
    np,
    pd,
    plt,
):
    def resolve_repo_path(value: object, repo_root: Path) -> Path:
        path = Path(str(value).strip()).expanduser()
        if path.is_absolute():
            return path
        return (repo_root / path).resolve()

    def _table_value(value: object) -> object:
        if value is None:
            return None
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, str | int | float | bool):
            return value
        return str(value)

    def _table_records(frame: pd.DataFrame) -> list[dict[str, object]]:
        normalized = frame.reset_index(drop=True)
        return [
            {str(column): _table_value(value) for column, value in row.items()}
            for row in normalized.to_dict("records")
        ]

    def display_table(frame: pd.DataFrame, *, max_height: int | None = None):
        if frame.empty:
            return mo.md("_No rows to display._")
        kwargs: dict[str, object] = {
            "selection": None,
            "pagination": False,
            "show_column_summaries": False,
            "show_data_types": False,
            "show_download": False,
        }
        if max_height is not None:
            kwargs["max_height"] = max_height
        return mo.ui.table(_table_records(frame), **kwargs)

    def load_preview_rgb(path, *, max_side: int = DEMO_MAX_SIDE) -> np.ndarray:
        with Image.open(path) as image:
            image = image.convert("RGB")
            image.thumbnail((max_side, max_side))
            return np.asarray(image, dtype=np.uint8)

    def grayscale(rgb: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    def apply_clahe(gray: np.ndarray, *, clip_limit: float = PACKAGE_CLAHE_CLIP_LIMIT):
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=CLAHE_TILE_GRID)
        return clahe.apply(gray)

    def foreground_mask(rgb: np.ndarray, strategy: str) -> np.ndarray:
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return calculate_mask_by_strategy(bgr, strategy=strategy)

    def component_mask_steps(rgb: np.ndarray) -> dict[str, object]:
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        gray = grayscale(rgb)
        _, binary = cv2.threshold(
            gray,
            CONNECTED_COMPONENT_THRESHOLD,
            255,
            cv2.THRESH_BINARY,
        )
        component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary,
            connectivity=8,
        )
        sorted_indices = np.argsort(stats[1:, cv2.CC_STAT_AREA])[::-1] + 1
        background_mask = np.zeros_like(binary).astype(np.uint8)
        reviewed_labels = np.zeros_like(labels, dtype=np.int32)
        rows = []

        for rank, label_id in enumerate(
            sorted_indices[:COMPONENT_TOP_COUNT],
            start=1,
        ):
            x, y, width, height, area = stats[label_id]
            component_mask = (labels == label_id).astype(np.uint8) * 255
            contours, _ = cv2.findContours(
                component_mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            cv2.drawContours(component_mask, contours, -1, 255, thickness=cv2.FILLED)
            reviewed_labels[component_mask == 255] = int(rank)

            skipped_small = width < COMPONENT_MIN_SIDE and height < COMPONENT_MIN_SIDE
            std_dev = None
            used_as_background = False
            if not skipped_small:
                roi = bgr[y : y + height, x : x + width]
                roi_mask = component_mask[y : y + height, x : x + width]
                std_dev = float(cv2.meanStdDev(roi, mask=roi_mask)[1][0, 0])
                if std_dev < COMPONENT_BACKGROUND_STD_DEV:
                    background_mask[component_mask == 255] = 255
                    used_as_background = True

            rows.append(
                {
                    "rank": rank,
                    "label_id": int(label_id),
                    "area": int(area),
                    "width": int(width),
                    "height": int(height),
                    "std_dev_b_channel": None if std_dev is None else round(std_dev, 4),
                    "skipped_small": skipped_small,
                    "used_as_background": used_as_background,
                }
            )

        foreground = cv2.bitwise_not(background_mask)
        return {
            "binary": binary,
            "labels": labels,
            "reviewed_labels": reviewed_labels,
            "background_mask": background_mask,
            "foreground_mask": foreground,
            "component_count": max(0, int(component_count) - 1),
            "reviewed_components": len(rows),
            "background_components": sum(1 for row in rows if row["used_as_background"]),
            "component_rows": rows,
        }

    def component_labels(mask: np.ndarray) -> tuple[np.ndarray, int]:
        foreground = (mask > 0).astype(np.uint8)
        count, labels = cv2.connectedComponents(foreground, connectivity=8)
        return labels, max(0, int(count) - 1)

    def affine_diagnostics(warp_matrix: np.ndarray | None) -> dict[str, float | None]:
        if warp_matrix is None:
            return {
                "scale_x": None,
                "scale_y": None,
                "rotation_deg": None,
                "translation_x": None,
                "translation_y": None,
            }
        a, b, tx = warp_matrix[0]
        c, d, ty = warp_matrix[1]
        return {
            "scale_x": float(np.sqrt(a * a + c * c)),
            "scale_y": float(np.sqrt(b * b + d * d)),
            "rotation_deg": float(np.degrees(np.arctan2(c, a))),
            "translation_x": float(tx),
            "translation_y": float(ty),
        }

    def load_demo_state(source_path, target_path) -> dict[str, object]:
        if not source_path.is_file() or not target_path.is_file():
            return {
                "status": "missing",
                "source": None,
                "target": None,
                "message": (
                    "Source or target demo image is missing. "
                    f"source={source_path}; target={target_path}"
                ),
            }
        try:
            return {
                "status": "present",
                "source": load_preview_rgb(source_path),
                "target": load_preview_rgb(target_path),
                "message": None,
            }
        except Exception as exc:
            return {
                "status": "error",
                "source": None,
                "target": None,
                "message": str(exc),
            }

    def image_overview_figure(source_rgb: np.ndarray, target_rgb: np.ndarray):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), constrained_layout=True)
        panels = [(source_rgb, "Source demo image"), (target_rgb, "Target demo image")]
        for axis, (image, title) in zip(axes, panels, strict=True):
            axis.imshow(image)
            axis.set_title(title)
            axis.axis("off")
        return fig

    def contrast_grid_figure(
        source_rgb: np.ndarray,
        target_rgb: np.ndarray,
        clip_limits: tuple[float, ...],
    ):
        columns = 1 + len(clip_limits)
        fig, axes = plt.subplots(
            2,
            columns,
            figsize=(2.45 * columns, 5.4),
            constrained_layout=True,
        )
        for row, (name, rgb) in enumerate([("Source", source_rgb), ("Target", target_rgb)]):
            gray = grayscale(rgb)
            variants = [("original grayscale", gray)] + [
                (f"CLAHE clip {clip_limit:g}", apply_clahe(gray, clip_limit=clip_limit))
                for clip_limit in clip_limits
            ]
            for col, (title, image) in enumerate(variants):
                axis = axes[row, col]
                axis.imshow(image, cmap="gray", vmin=0, vmax=255)
                axis.set_title(f"{name}\n{title}", fontsize=9)
                axis.axis("off")
        return fig

    def custom_clahe_figure(
        source_rgb: np.ndarray,
        target_rgb: np.ndarray,
        clip_limit: float,
    ):
        fig, axes = plt.subplots(2, 3, figsize=(13.5, 7), constrained_layout=True)
        for row, (name, rgb) in enumerate([("Source", source_rgb), ("Target", target_rgb)]):
            gray = grayscale(rgb)
            normalized = apply_clahe(gray, clip_limit=clip_limit)
            panels = [
                (gray, "original grayscale"),
                (normalized, f"CLAHE clip {clip_limit:g}"),
            ]
            for col, (image, title) in enumerate(panels):
                axis = axes[row, col]
                axis.imshow(image, cmap="gray", vmin=0, vmax=255)
                axis.set_title(f"{name}\n{title}", fontsize=9)
                axis.axis("off")
            histogram_axis = axes[row, 2]
            histogram_axis.hist(
                gray.ravel(),
                bins=64,
                range=(0, 256),
                histtype="stepfilled",
                alpha=0.3,
                color="tab:blue",
                label="original grayscale",
            )
            histogram_axis.hist(
                normalized.ravel(),
                bins=64,
                range=(0, 256),
                histtype="stepfilled",
                alpha=0.3,
                color="tab:orange",
                label=f"CLAHE clip {clip_limit:g}",
            )
            histogram_axis.hist(
                gray.ravel(),
                bins=64,
                range=(0, 256),
                histtype="step",
                linewidth=1.5,
                color="tab:blue",
            )
            histogram_axis.hist(
                normalized.ravel(),
                bins=64,
                range=(0, 256),
                histtype="step",
                linewidth=1.5,
                color="tab:orange",
            )
            histogram_axis.set_title(f"{name}\noverlaid intensity histogram", fontsize=9)
            histogram_axis.set_xlabel("intensity")
            histogram_axis.set_ylabel("pixels")
            histogram_axis.legend(fontsize=8)
        return fig

    def clahe_histogram_grid_figure(
        source_rgb: np.ndarray,
        target_rgb: np.ndarray,
        clip_limits: tuple[float, ...],
    ):
        fig, axes = plt.subplots(2, 1, figsize=(11, 7), constrained_layout=True)
        for axis, (name, rgb) in zip(
            axes,
            [("Source", source_rgb), ("Target", target_rgb)],
            strict=True,
        ):
            gray = grayscale(rgb)
            axis.hist(
                gray.ravel(),
                bins=64,
                range=(0, 256),
                histtype="step",
                linewidth=1.8,
                label="original grayscale",
            )
            for clip_limit in clip_limits:
                normalized = apply_clahe(gray, clip_limit=clip_limit)
                axis.hist(
                    normalized.ravel(),
                    bins=64,
                    range=(0, 256),
                    histtype="step",
                    linewidth=1.2,
                    label=f"CLAHE clip {clip_limit:g}",
                )
            axis.set_title(f"{name}: original and CLAHE intensity histograms")
            axis.set_xlabel("intensity")
            axis.set_ylabel("pixels")
            axis.legend(ncols=2, fontsize=8)
        return fig

    def mask_statistics_frame(
        source_rgb: np.ndarray,
        target_rgb: np.ndarray,
        strategies: tuple[str, ...],
    ) -> pd.DataFrame:
        rows = []
        for strategy in strategies:
            for role, rgb in [("source", source_rgb), ("target", target_rgb)]:
                mask = foreground_mask(rgb, strategy)
                _, components = component_labels(mask)
                foreground_pixels = int(cv2.countNonZero(mask))
                rows.append(
                    {
                        "strategy": strategy,
                        "image": role,
                        "foreground_pixels": foreground_pixels,
                        "foreground_ratio": round(foreground_pixels / mask.size, 4),
                        "component_count": components,
                    }
                )
        return pd.DataFrame(rows)

    def component_mask_statistics_frame(
        source_steps: dict[str, object],
        target_steps: dict[str, object],
    ) -> pd.DataFrame:
        rows = []
        for role, steps in [("source", source_steps), ("target", target_steps)]:
            foreground = steps["foreground_mask"]
            foreground_pixels = int(cv2.countNonZero(foreground))
            rows.append(
                {
                    "image": role,
                    "threshold": CONNECTED_COMPONENT_THRESHOLD,
                    "component_count": steps["component_count"],
                    "reviewed_components": steps["reviewed_components"],
                    "background_components": steps["background_components"],
                    "foreground_pixels": foreground_pixels,
                    "foreground_ratio": round(foreground_pixels / foreground.size, 4),
                }
            )
        return pd.DataFrame(rows)

    def component_review_frame(
        source_steps: dict[str, object],
        target_steps: dict[str, object],
    ) -> pd.DataFrame:
        rows = []
        for role, steps in [("source", source_steps), ("target", target_steps)]:
            for row in steps["component_rows"]:
                rows.append({"image": role, **row})
        return pd.DataFrame(rows)

    def component_mask_construction_figure(
        source_rgb: np.ndarray,
        target_rgb: np.ndarray,
        source_steps: dict[str, object],
        target_steps: dict[str, object],
    ):
        fig, axes = plt.subplots(2, 5, figsize=(15, 6.4), constrained_layout=True)
        rows = [
            ("Source", source_rgb, source_steps),
            ("Target", target_rgb, target_steps),
        ]
        for row_index, (name, rgb, steps) in enumerate(rows):
            panels = [
                (grayscale(rgb), "grayscale", "gray", None),
                (steps["binary"], f"threshold >= {CONNECTED_COMPONENT_THRESHOLD}", "gray", None),
                (steps["labels"], "all connected-component labels", "nipy_spectral", "nearest"),
                (
                    steps["background_mask"],
                    f"low-std background (< {COMPONENT_BACKGROUND_STD_DEV:g})",
                    "gray",
                    None,
                ),
                (steps["foreground_mask"], "foreground after inversion", "gray", None),
            ]
            for col_index, (image, title, cmap, interpolation) in enumerate(panels):
                axis = axes[row_index, col_index]
                axis.imshow(image, cmap=cmap, interpolation=interpolation)
                axis.set_title(f"{name}\n{title}", fontsize=9)
                axis.axis("off")
        return fig

    def mask_strategy_figure(
        source_rgb: np.ndarray,
        target_rgb: np.ndarray,
        strategies: tuple[str, ...],
    ):
        rows_per_strategy = 2
        total_rows = len(strategies) * rows_per_strategy
        fig, axes = plt.subplots(
            total_rows,
            3,
            figsize=(12, 3.1 * total_rows),
            constrained_layout=True,
        )
        if total_rows == 1:
            axes = np.asarray([axes])
        for strategy_index, strategy in enumerate(strategies):
            for role_index, (role, rgb) in enumerate(
                [("source", source_rgb), ("target", target_rgb)]
            ):
                row = strategy_index * rows_per_strategy + role_index
                gray = grayscale(rgb)
                normalized = apply_clahe(gray)
                mask = foreground_mask(rgb, strategy)
                _, components = component_labels(mask)
                panels = [
                    (rgb, f"{role} original RGB", None),
                    (normalized, f"{role} CLAHE grayscale", "gray"),
                    (mask, f"{role} foreground mask ({components} components)", "gray"),
                ]
                for col, (image, title, cmap) in enumerate(panels):
                    axis = axes[row, col]
                    if cmap is None:
                        axis.imshow(image)
                    else:
                        axis.imshow(image, cmap=cmap, vmin=0, vmax=255)
                    axis.set_title(f"{strategy}\n{title}", fontsize=9)
                    axis.axis("off")
        return fig

    def _detect_keypoints(
        source_rgb: np.ndarray,
        target_rgb: np.ndarray,
        strategy: str,
    ) -> dict[str, object]:
        source_gray = apply_clahe(grayscale(source_rgb))
        target_gray = apply_clahe(grayscale(target_rgb))
        source_mask = foreground_mask(source_rgb, strategy)
        target_mask = foreground_mask(target_rgb, strategy)
        sift = cv2.SIFT_create(nfeatures=SIFT_NFEATURES)
        source_keypoints, source_descriptors = sift.detectAndCompute(source_gray, source_mask)
        target_keypoints, target_descriptors = sift.detectAndCompute(target_gray, target_mask)
        return {
            "strategy": strategy,
            "source_rgb": source_rgb,
            "target_rgb": target_rgb,
            "source_gray": source_gray,
            "target_gray": target_gray,
            "source_mask": source_mask,
            "target_mask": target_mask,
            "source_keypoints": source_keypoints,
            "target_keypoints": target_keypoints,
            "source_descriptors": source_descriptors,
            "target_descriptors": target_descriptors,
        }

    def build_keypoint_bundles(
        source_rgb: np.ndarray,
        target_rgb: np.ndarray,
        strategies: tuple[str, ...],
    ) -> dict[str, dict[str, object]]:
        bundles = {}
        for strategy in strategies:
            try:
                bundles[strategy] = _detect_keypoints(source_rgb, target_rgb, strategy)
            except Exception as exc:
                bundles[strategy] = {"strategy": strategy, "error": str(exc)}
        return bundles

    def _strongest_keypoints(keypoints: list[object] | tuple[object, ...], limit: int):
        return sorted(keypoints, key=lambda keypoint: keypoint.response, reverse=True)[:limit]

    def draw_keypoints_preview(rgb: np.ndarray, keypoints: tuple[object, ...]) -> np.ndarray:
        selected = _strongest_keypoints(keypoints, KEYPOINT_PREVIEW_LIMIT)
        preview = cv2.cvtColor(rgb.copy(), cv2.COLOR_RGB2BGR)
        for keypoint in selected:
            x, y = (int(round(value)) for value in keypoint.pt)
            radius = max(5, min(14, int(round(keypoint.size / 2))))
            cv2.circle(preview, (x, y), radius + 2, (0, 0, 0), thickness=3, lineType=cv2.LINE_AA)
            cv2.circle(preview, (x, y), radius, (0, 255, 255), thickness=2, lineType=cv2.LINE_AA)
            cv2.line(
                preview,
                (x - radius, y),
                (x + radius, y),
                (255, 0, 255),
                thickness=2,
                lineType=cv2.LINE_AA,
            )
            cv2.line(
                preview,
                (x, y - radius),
                (x, y + radius),
                (255, 0, 255),
                thickness=2,
                lineType=cv2.LINE_AA,
            )
        return cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)

    def keypoint_count_frame(bundles: dict[str, dict[str, object]]) -> pd.DataFrame:
        rows = []
        for strategy, bundle in bundles.items():
            if bundle.get("error"):
                rows.append(
                    {
                        "strategy": strategy,
                        "source_keypoints": None,
                        "target_keypoints": None,
                        "status": bundle["error"],
                    }
                )
                continue
            rows.append(
                {
                    "strategy": strategy,
                    "source_keypoints": len(bundle["source_keypoints"]),
                    "target_keypoints": len(bundle["target_keypoints"]),
                    "status": "ok",
                }
            )
        return pd.DataFrame(rows)

    def keypoint_figure(bundles: dict[str, dict[str, object]]):
        strategies = list(bundles)
        fig, axes = plt.subplots(
            len(strategies),
            2,
            figsize=(13, 5.2 * len(strategies)),
            constrained_layout=True,
        )
        if len(strategies) == 1:
            axes = np.asarray([axes])
        for row, strategy in enumerate(strategies):
            bundle = bundles[strategy]
            for col, role in enumerate(["source", "target"]):
                axis = axes[row, col]
                if bundle.get("error"):
                    axis.text(0.5, 0.5, str(bundle["error"]), ha="center", va="center")
                    axis.axis("off")
                    continue
                rgb = bundle[f"{role}_rgb"]
                keypoints = bundle[f"{role}_keypoints"]
                axis.imshow(draw_keypoints_preview(rgb, keypoints))
                axis.set_title(
                    f"{strategy}\n{role} SIFT keypoints: top "
                    f"{min(KEYPOINT_PREVIEW_LIMIT, len(keypoints))} of {len(keypoints)}",
                    fontsize=9,
                )
                axis.axis("off")
        return fig

    def _ratio_test_matches(knn_matches: list[list[object]]) -> list[object]:
        filtered = []
        for candidates in knn_matches:
            if len(candidates) < 2:
                continue
            best, second_best = candidates[0], candidates[1]
            if best.distance < LOWE_RATIO_THRESHOLD * second_best.distance:
                filtered.append(best)
        return filtered

    def build_match_diagnostics(bundle: dict[str, object]) -> dict[str, object]:
        if bundle.get("error"):
            return {"strategy": bundle.get("strategy"), "error": bundle["error"]}

        source_keypoints = bundle["source_keypoints"]
        target_keypoints = bundle["target_keypoints"]
        source_descriptors = bundle["source_descriptors"]
        target_descriptors = bundle["target_descriptors"]
        if (
            len(source_keypoints) < 4
            or len(target_keypoints) < 4
            or source_descriptors is None
            or target_descriptors is None
        ):
            return {
                **bundle,
                "raw_matches": [],
                "ratio_matches": [],
                "inlier_matches": [],
                "inlier_mask": None,
                "warp_matrix": None,
                "error": "Not enough SIFT features for descriptor matching.",
            }

        matcher = cv2.BFMatcher(cv2.NORM_L2)
        knn_matches = matcher.knnMatch(source_descriptors, target_descriptors, k=2)
        raw_matches = [candidates[0] for candidates in knn_matches if candidates]
        ratio_matches = _ratio_test_matches(knn_matches)
        if len(ratio_matches) < 4:
            return {
                **bundle,
                "raw_matches": raw_matches,
                "ratio_matches": ratio_matches,
                "inlier_matches": [],
                "inlier_mask": None,
                "warp_matrix": None,
                "error": "Not enough ratio-test matches for RANSAC affine estimation.",
            }

        points_source = np.asarray(
            [source_keypoints[match.queryIdx].pt for match in ratio_matches],
            dtype=np.float32,
        ).reshape(-1, 1, 2)
        points_target = np.asarray(
            [target_keypoints[match.trainIdx].pt for match in ratio_matches],
            dtype=np.float32,
        ).reshape(-1, 1, 2)
        warp_matrix, inlier_mask = cv2.estimateAffinePartial2D(
            points_target,
            points_source,
            method=cv2.RANSAC,
            ransacReprojThreshold=RANSAC_REPROJECTION_THRESHOLD,
            maxIters=RANSAC_MAX_ITERS,
            confidence=RANSAC_CONFIDENCE,
            refineIters=RANSAC_REFINE_ITERS,
        )
        inlier_matches: list[object] = []
        if inlier_mask is not None:
            inlier_flags = inlier_mask.ravel().astype(bool)
            inlier_matches = [
                match
                for match, is_inlier in zip(ratio_matches, inlier_flags, strict=True)
                if is_inlier
            ]
        error = None if warp_matrix is not None else "RANSAC returned no affine transform."
        return {
            **bundle,
            "raw_matches": raw_matches,
            "ratio_matches": ratio_matches,
            "inlier_matches": inlier_matches,
            "inlier_mask": inlier_mask,
            "warp_matrix": warp_matrix,
            "error": error,
        }

    def matching_count_frame(match_bundle: dict[str, object]) -> pd.DataFrame:
        source_keypoints = match_bundle.get("source_keypoints", ())
        target_keypoints = match_bundle.get("target_keypoints", ())
        ratio_matches = match_bundle.get("ratio_matches", ())
        inlier_matches = match_bundle.get("inlier_matches", ())
        inlier_ratio = len(inlier_matches) / len(ratio_matches) if ratio_matches else 0.0
        diagnostics = affine_diagnostics(match_bundle.get("warp_matrix"))
        quality_status = (
            "passes package gates"
            if len(inlier_matches) >= MIN_INLIERS and inlier_ratio >= MIN_INLIER_RATIO
            else "below package gates"
        )
        row = {
            "strategy": match_bundle.get("strategy", DEFAULT_MATCH_STRATEGY),
            "source_keypoints": len(source_keypoints),
            "target_keypoints": len(target_keypoints),
            "raw_matches": len(match_bundle.get("raw_matches", ())),
            "ratio_test_matches": len(ratio_matches),
            "ransac_inliers": len(inlier_matches),
            "inlier_ratio": round(inlier_ratio, 4),
            "quality_status": quality_status,
            "error": match_bundle.get("error"),
            **{
                key: None if value is None else round(value, 4)
                for key, value in diagnostics.items()
            },
        }
        return pd.DataFrame([row])

    def draw_matches_preview(
        source_gray: np.ndarray,
        source_keypoints: tuple[object, ...],
        target_gray: np.ndarray,
        target_keypoints: tuple[object, ...],
        matches: list[object],
    ) -> np.ndarray:
        selected = sorted(matches, key=lambda match: match.distance)[:MATCH_PREVIEW_LIMIT]
        source_vis = cv2.cvtColor(source_gray, cv2.COLOR_GRAY2BGR)
        target_vis = cv2.cvtColor(target_gray, cv2.COLOR_GRAY2BGR)
        drawn = cv2.drawMatches(
            source_vis,
            source_keypoints,
            target_vis,
            target_keypoints,
            selected,
            None,
            matchesThickness=MATCH_PREVIEW_LINE_THICKNESS,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        return cv2.cvtColor(drawn, cv2.COLOR_BGR2RGB)

    def matching_figure(match_bundle: dict[str, object]):
        if match_bundle.get("error") and not match_bundle.get("raw_matches"):
            return None
        panels = [
            ("Raw nearest-neighbor matches", match_bundle.get("raw_matches", [])),
            (
                f"Ratio-test matches, threshold {LOWE_RATIO_THRESHOLD}",
                match_bundle.get("ratio_matches", []),
            ),
            ("RANSAC inlier matches", match_bundle.get("inlier_matches", [])),
        ]
        fig, axes = plt.subplots(3, 1, figsize=(13.5, 11.2), constrained_layout=True)
        for axis, (title, matches) in zip(axes, panels, strict=True):
            if matches:
                axis.imshow(
                    draw_matches_preview(
                        match_bundle["source_gray"],
                        match_bundle["source_keypoints"],
                        match_bundle["target_gray"],
                        match_bundle["target_keypoints"],
                        matches,
                    )
                )
            else:
                axis.text(0.5, 0.5, "No matches to display.", ha="center", va="center")
            axis.set_title(f"{title} (showing up to {MATCH_PREVIEW_LIMIT})", fontsize=10)
            axis.axis("off")
        return fig

    def resize_to_source(
        image: np.ndarray,
        source_shape: tuple[int, ...],
        *,
        interpolation: int,
    ) -> np.ndarray:
        if image.shape[:2] == source_shape[:2]:
            return image
        return cv2.resize(
            image,
            (source_shape[1], source_shape[0]),
            interpolation=interpolation,
        )

    def alignment_diagnostic_arrays(match_bundle: dict[str, object]) -> dict[str, np.ndarray]:
        warp_matrix = match_bundle["warp_matrix"]
        source_rgb = match_bundle["source_rgb"]
        target_rgb = match_bundle["target_rgb"]
        target_rgb_before = resize_to_source(
            target_rgb,
            source_rgb.shape,
            interpolation=cv2.INTER_AREA,
        )
        target_rgb_warped = cv2.warpAffine(
            target_rgb,
            warp_matrix,
            (source_rgb.shape[1], source_rgb.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )

        source_clahe = match_bundle["source_gray"]
        target_clahe = match_bundle["target_gray"]
        target_clahe_before = resize_to_source(
            target_clahe,
            source_clahe.shape,
            interpolation=cv2.INTER_AREA,
        )
        target_clahe_warped = cv2.warpAffine(
            target_clahe,
            warp_matrix,
            (source_clahe.shape[1], source_clahe.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=255,
        )
        residual_before = cv2.absdiff(source_clahe, target_clahe_before)
        residual_after = cv2.absdiff(source_clahe, target_clahe_warped)
        return {
            "source_rgb": source_rgb,
            "target_rgb_before": target_rgb_before,
            "target_rgb_warped": target_rgb_warped,
            "residual_before": residual_before,
            "residual_after": residual_after,
        }

    def residual_figure(match_bundle: dict[str, object]):
        warp_matrix = match_bundle.get("warp_matrix")
        if warp_matrix is None:
            return None
        arrays = alignment_diagnostic_arrays(match_bundle)
        before = arrays["residual_before"]
        after = arrays["residual_after"]

        fig, axes = plt.subplots(2, 3, figsize=(14, 8.6), constrained_layout=True)
        axes[0, 0].imshow(arrays["source_rgb"])
        axes[0, 0].set_title("Source original RGB")
        axes[0, 1].imshow(arrays["target_rgb_before"])
        axes[0, 1].set_title("Target original RGB before transform")
        axes[0, 2].imshow(arrays["target_rgb_warped"])
        axes[0, 2].set_title("Target original RGB after transform")
        for axis in axes[0]:
            axis.axis("off")

        axes[1, 0].imshow(before, cmap="magma", vmin=0, vmax=255)
        axes[1, 0].set_title(f"CLAHE abs diff before, mean={before.mean():.1f}")
        axes[1, 0].axis("off")
        axes[1, 1].imshow(after, cmap="magma", vmin=0, vmax=255)
        axes[1, 1].set_title(f"CLAHE abs diff after, mean={after.mean():.1f}")
        axes[1, 1].axis("off")

        axes[1, 2].hist(
            before.ravel(),
            bins=64,
            range=(0, 256),
            histtype="stepfilled",
            alpha=0.35,
            color="tab:blue",
            label=f"before, mean={before.mean():.1f}",
        )
        axes[1, 2].hist(
            after.ravel(),
            bins=64,
            range=(0, 256),
            histtype="stepfilled",
            alpha=0.35,
            color="tab:green",
            label=f"after, mean={after.mean():.1f}",
        )
        axes[1, 2].hist(
            before.ravel(),
            bins=64,
            range=(0, 256),
            histtype="step",
            linewidth=1.5,
            color="tab:blue",
        )
        axes[1, 2].hist(
            after.ravel(),
            bins=64,
            range=(0, 256),
            histtype="step",
            linewidth=1.5,
            color="tab:green",
        )
        axes[1, 2].set_title("Overlaid CLAHE residual histograms")
        axes[1, 2].set_xlabel("absolute difference")
        axes[1, 2].set_ylabel("pixels")
        axes[1, 2].legend(fontsize=8)
        return fig

    def residual_summary_frame(match_bundle: dict[str, object]) -> pd.DataFrame:
        warp_matrix = match_bundle.get("warp_matrix")
        if warp_matrix is None:
            return pd.DataFrame()
        arrays = alignment_diagnostic_arrays(match_bundle)
        before = arrays["residual_before"]
        after = arrays["residual_after"]
        return pd.DataFrame(
            [
                {
                    "diagnostic": "mean absolute difference",
                    "before": round(float(before.mean()), 4),
                    "after": round(float(after.mean()), 4),
                    "delta": round(float(after.mean() - before.mean()), 4),
                },
                {
                    "diagnostic": "median absolute difference",
                    "before": round(float(np.median(before)), 4),
                    "after": round(float(np.median(after)), 4),
                    "delta": round(float(np.median(after) - np.median(before)), 4),
                },
                {
                    "diagnostic": "90th percentile absolute difference",
                    "before": round(float(np.percentile(before, 90)), 4),
                    "after": round(float(np.percentile(after, 90)), 4),
                    "delta": round(
                        float(np.percentile(after, 90) - np.percentile(before, 90)),
                        4,
                    ),
                },
            ]
        )

    return (
        build_keypoint_bundles,
        build_match_diagnostics,
        clahe_histogram_grid_figure,
        contrast_grid_figure,
        component_mask_construction_figure,
        component_mask_statistics_frame,
        component_mask_steps,
        component_review_frame,
        custom_clahe_figure,
        display_table,
        image_overview_figure,
        keypoint_count_frame,
        keypoint_figure,
        load_demo_state,
        mask_statistics_frame,
        mask_strategy_figure,
        matching_count_frame,
        matching_figure,
        residual_figure,
        residual_summary_frame,
        resolve_repo_path,
    )


@app.cell
def _(
    REPO_ROOT,
    resolve_repo_path,
    source_image_input,
    target_image_input,
):
    source_image_path = resolve_repo_path(source_image_input.value, REPO_ROOT)
    target_image_path = resolve_repo_path(target_image_input.value, REPO_ROOT)

    return source_image_path, target_image_path


@app.cell
def _(load_demo_state, source_image_path, target_image_path):
    demo_state = load_demo_state(source_image_path, target_image_path)
    source_rgb = demo_state["source"]
    target_rgb = demo_state["target"]
    return demo_state, source_rgb, target_rgb


@app.cell
def _(component_mask_steps, demo_state, source_rgb, target_rgb):
    if demo_state["status"] == "present":
        source_component_mask_steps = component_mask_steps(source_rgb)
        target_component_mask_steps = component_mask_steps(target_rgb)
    else:
        source_component_mask_steps = {}
        target_component_mask_steps = {}
    return source_component_mask_steps, target_component_mask_steps


@app.cell
def _(demo_state, image_overview_figure, mo, source_rgb, target_rgb):
    if demo_state["status"] == "present":
        demo_view = mo.vstack(
            [
                mo.md("## Demo Image Pair"),
                mo.md(
                    "The figures below use the selected images in memory only. "
                    "The default paths are committed repository examples."
                ),
                image_overview_figure(source_rgb, target_rgb),
            ]
        )
    else:
        demo_view = mo.md(
            f"""
            ## Demo Image Pair

            Status: `{demo_state["status"]}`

            {demo_state["message"]}
            """
        )
    mo.output.replace(demo_view)
    return


@app.cell
def _(
    ALLOWED_MASK_STRATEGIES,
    build_keypoint_bundles,
    build_match_diagnostics,
    demo_state,
    source_rgb,
    target_rgb,
):
    if demo_state["status"] == "present":
        keypoint_bundles = build_keypoint_bundles(
            source_rgb,
            target_rgb,
            ALLOWED_MASK_STRATEGIES,
        )
        match_bundle = build_match_diagnostics(keypoint_bundles["connected_components"])
    else:
        keypoint_bundles = {}
        match_bundle = {"error": demo_state["message"], "strategy": "connected_components"}
    return keypoint_bundles, match_bundle


@app.cell
def _(PACKAGE_CLAHE_CLIP_LIMIT, mo):
    custom_clahe_clip_limit = mo.ui.number(
        start=0.1,
        stop=64.0,
        step=0.5,
        value=PACKAGE_CLAHE_CLIP_LIMIT,
        label="custom_CLAHE_clipLimit",
        full_width=True,
    )
    return (custom_clahe_clip_limit,)


@app.cell
def _(
    CLAHE_COMPARISON_CLIP_LIMITS,
    PACKAGE_CLAHE_CLIP_LIMIT,
    contrast_grid_figure,
    custom_clahe_clip_limit,
    custom_clahe_figure,
    demo_state,
    mo,
    source_rgb,
    target_rgb,
):
    if demo_state["status"] == "present":
        selected_clip_limit = custom_clahe_clip_limit.value
        if selected_clip_limit is None:
            selected_clip_limit = PACKAGE_CLAHE_CLIP_LIMIT
        contrast_view = mo.vstack(
            [
                mo.md(
                    f"""
                    ## Contrast Normalization and CLAHE

                    Feature detection is run on grayscale CLAHE-normalized images.
                    The maintained code currently uses `clipLimit={PACKAGE_CLAHE_CLIP_LIMIT:g}`
                    and `tileGridSize=(8, 8)` before SIFT. The comparison grid shows
                    common clip limits immediately so the effect is visible without
                    changing a control.
                    """
                ),
                contrast_grid_figure(
                    source_rgb,
                    target_rgb,
                    CLAHE_COMPARISON_CLIP_LIMITS,
                ),
                mo.md("### Custom CLAHE Output"),
                custom_clahe_clip_limit,
                custom_clahe_figure(
                    source_rgb,
                    target_rgb,
                    float(selected_clip_limit),
                ),
            ]
        )
    else:
        contrast_view = mo.md("## Contrast Normalization and CLAHE\n\nDemo images are unavailable.")
    mo.output.replace(contrast_view)
    return


@app.cell
def _(
    CLAHE_COMPARISON_CLIP_LIMITS,
    clahe_histogram_grid_figure,
    demo_state,
    mo,
    source_rgb,
    target_rgb,
):
    if demo_state["status"] == "present":
        histogram_view = mo.vstack(
            [
                mo.md(
                    """
                    ## CLAHE Intensity Histograms

                    These histograms compare the original grayscale intensity distribution
                    with every CLAHE clip limit shown in the visual grid above. They are
                    displayed by default so contrast-normalization effects are visible
                    without changing controls.
                    """
                ),
                clahe_histogram_grid_figure(
                    source_rgb,
                    target_rgb,
                    CLAHE_COMPARISON_CLIP_LIMITS,
                ),
            ]
        )
    else:
        histogram_view = mo.md("## CLAHE Intensity Histograms\n\nDemo images are unavailable.")
    mo.output.replace(histogram_view)
    return


@app.cell
def _(
    ALLOWED_MASK_STRATEGIES,
    CONNECTED_COMPONENT_THRESHOLD,
    MASK_PARAMETER_GRID,
    MIN_STD_DEV,
    COMPONENT_BACKGROUND_STD_DEV,
    COMPONENT_MIN_SIDE,
    COMPONENT_TOP_COUNT,
    component_mask_construction_figure,
    component_mask_statistics_frame,
    component_review_frame,
    demo_state,
    display_table,
    mask_statistics_frame,
    mask_strategy_figure,
    mo,
    source_component_mask_steps,
    source_rgb,
    target_component_mask_steps,
    target_rgb,
):
    if demo_state["status"] == "present":
        mask_view = mo.vstack(
            [
                mo.md(
                    f"""
                    ## Mask Construction

                    The component-mask walkthrough starts with a single full-image
                    grayscale threshold at `{CONNECTED_COMPONENT_THRESHOLD}`, reviews
                    the `{COMPONENT_TOP_COUNT}` largest connected components, skips
                    components smaller than
                    `{COMPONENT_MIN_SIDE}x{COMPONENT_MIN_SIDE}`, marks components
                    with BGR-channel std dev below `{COMPONENT_BACKGROUND_STD_DEV:g}`
                    as background, then inverts that background mask into foreground.
                    """
                ),
                component_mask_construction_figure(
                    source_rgb,
                    target_rgb,
                    source_component_mask_steps,
                    target_component_mask_steps,
                ),
                mo.md("### Component Mask Statistics"),
                display_table(
                    component_mask_statistics_frame(
                        source_component_mask_steps,
                        target_component_mask_steps,
                    )
                ),
                mo.md("### Component Review"),
                display_table(
                    component_review_frame(
                        source_component_mask_steps,
                        target_component_mask_steps,
                    ),
                    max_height=420,
                ),
                mo.md(
                    f"""
                    ### Mask Strategy Comparison

                    The maintained connected-component strategy applies the same
                    foreground/background idea across grid passes `{MASK_PARAMETER_GRID}`
                    and uses `MIN_STD_DEV={MIN_STD_DEV}`. The `hsv` strategy is shown
                    alongside it by default. These masks are used for the SIFT/keypoint
                    previews in the next section. Each strategy shows the original RGB
                    image, the CLAHE grayscale image, and the resulting foreground mask
                    for both source and target.
                    """
                ),
                mask_strategy_figure(source_rgb, target_rgb, ALLOWED_MASK_STRATEGIES),
                mo.md("### Mask Strategy Statistics"),
                display_table(
                    mask_statistics_frame(source_rgb, target_rgb, ALLOWED_MASK_STRATEGIES)
                ),
            ]
        )
    else:
        mask_view = mo.md("## Mask Construction\n\nDemo images are unavailable.")
    mo.output.replace(mask_view)
    return


@app.cell
def _(demo_state, display_table, keypoint_bundles, keypoint_count_frame, keypoint_figure, mo):
    if demo_state["status"] == "present":
        keypoint_view = mo.vstack(
            [
                mo.md(
                    """
                    ## Keypoint Detection

                    Current alignment uses SIFT on the CLAHE-normalized grayscale
                    images and applies the selected foreground masks during detection.
                    The previews draw the strongest keypoints for readability while the
                    table reports the full detected counts for both mask strategies.
                    """
                ),
                keypoint_figure(keypoint_bundles),
                display_table(keypoint_count_frame(keypoint_bundles)),
            ]
        )
    else:
        keypoint_view = mo.md("## Keypoint Detection\n\nDemo images are unavailable.")
    mo.output.replace(keypoint_view)
    return


@app.cell
def _(
    DEFAULT_MATCH_STRATEGY,
    LOWE_RATIO_THRESHOLD,
    RANSAC_CONFIDENCE,
    RANSAC_MAX_ITERS,
    RANSAC_REFINE_ITERS,
    RANSAC_REPROJECTION_THRESHOLD,
    demo_state,
    display_table,
    match_bundle,
    matching_count_frame,
    matching_figure,
    mo,
):
    if demo_state["status"] == "present":
        match_plot = matching_figure(match_bundle)
        match_items = [
            mo.md(
                f"""
                ## Matching and Filtering

                The runtime diagnostic below uses `{DEFAULT_MATCH_STRATEGY}`, the
                default mask strategy in the example config. It mirrors the package
                sequence: raw nearest-neighbor descriptor matches, Lowe ratio-test
                filtering at `{LOWE_RATIO_THRESHOLD}`, then RANSAC affine estimation
                with threshold `{RANSAC_REPROJECTION_THRESHOLD}`, max iterations
                `{RANSAC_MAX_ITERS}`, confidence `{RANSAC_CONFIDENCE}`, and refine
                iterations `{RANSAC_REFINE_ITERS}`. The figure shows raw matches,
                ratio-test matches, and RANSAC inlier matches directly.
                """
            ),
            display_table(matching_count_frame(match_bundle)),
        ]
        if match_bundle.get("error"):
            match_items.append(mo.md(f"Diagnostic note: `{match_bundle['error']}`"))
        if match_plot is not None:
            match_items.append(match_plot)
        match_view = mo.vstack(match_items)
    else:
        match_view = mo.md("## Matching and Filtering\n\nDemo images are unavailable.")
    mo.output.replace(match_view)
    return


@app.cell
def _(
    MIN_INLIERS,
    MIN_INLIER_RATIO,
    demo_state,
    display_table,
    match_bundle,
    mo,
    residual_figure,
    residual_summary_frame,
):
    if demo_state["status"] == "present":
        residual_plot = residual_figure(match_bundle)
        if residual_plot is None:
            diagnostic_view = mo.md(
                f"""
                ## Alignment Diagnostics

                The demo transform could not be estimated, so residual previews are
                not available.

                Diagnostic note: `{match_bundle.get("error", "unknown")}`
                """
            )
        else:
            diagnostic_view = mo.vstack(
                [
                    mo.md(
                        f"""
                        ## Alignment Diagnostics

                        These residual previews are lightweight diagnostics only. They
                        show the in-memory estimated demo transform on the original RGB
                        image previews. The absolute-difference maps and histogram are
                        computed on the CLAHE-normalized grayscale images used for SIFT.
                        The package quality gates require at least `{MIN_INLIERS}`
                        RANSAC inliers and an inlier ratio of at least
                        `{MIN_INLIER_RATIO:.2f}` before accepting an alignment.
                        """
                    ),
                    display_table(residual_summary_frame(match_bundle)),
                    residual_plot,
                    mo.md(
                        "The target-after panel is the result of running the estimated "
                        "demo transform in memory on the original target preview. The "
                        "overlaid histogram makes the CLAHE before/after residual "
                        "distribution comparable at a glance. Because the modalities "
                        "have different appearance, this is a visual sanity check "
                        "rather than an optimization objective."
                    ),
                ]
            )
    else:
        diagnostic_view = mo.md("## Alignment Diagnostics\n\nDemo images are unavailable.")
    mo.output.replace(diagnostic_view)
    return


if __name__ == "__main__":
    app.run()
