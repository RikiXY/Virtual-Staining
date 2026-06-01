import marimo

__generated_with = "0.23.8"
app = marimo.App(width="full")


@app.cell
def _():
    import json
    from pathlib import Path
    from typing import Any

    import marimo as mo
    import numpy as np
    import pandas as pd

    from virtual_staining.config.run import RunConfig
    from virtual_staining.data.manifest import DatasetManifest

    return Any, DatasetManifest, Path, RunConfig, json, mo, np, pd


@app.cell
def _(Path):
    REPO_ROOT = Path(__file__).resolve().parents[3]
    DEFAULT_RUN_YAML = "config/runs/example.yaml"
    DEFAULT_DATASET_ROOT = "local_workspace/datasets/your_sample"
    return DEFAULT_DATASET_ROOT, DEFAULT_RUN_YAML, REPO_ROOT


@app.cell
def _(mo):
    mo.output.replace(
        mo.md(
            "# Alignment Run Inspector\n\n"
            "This app inspects alignment artifacts from an existing prepared dataset. "
            "It is read-only: it does not run `vs-prepare`, recompute masks, estimate "
            "new transforms, write previews, or materialize a full aligned target image.\n\n"
            "The app opens cleanly when no prepared dataset exists and reports missing "
            "artifacts as table rows or readable messages."
        )
    )
    return


@app.cell
def _(DEFAULT_DATASET_ROOT, DEFAULT_RUN_YAML, mo):
    run_yaml_input = mo.ui.text(
        label="run_yaml_path",
        value=DEFAULT_RUN_YAML,
        full_width=True,
    )
    dataset_root_input = mo.ui.text(
        label="dataset_root",
        value=DEFAULT_DATASET_ROOT,
        full_width=True,
    )

    mo.vstack(
        [
            mo.md("## Inputs"),
            mo.hstack([run_yaml_input, dataset_root_input], widths="equal"),
        ]
    )
    return dataset_root_input, run_yaml_input


@app.cell
def _(Any, DatasetManifest, Path, RunConfig, json, mo, np, pd):
    def resolve_repo_path(value: object, repo_root: Path) -> Path:
        path = Path(str(value).strip()).expanduser()
        if path.is_absolute():
            return path
        return (repo_root / path).resolve()

    def read_json_state(path: Path) -> dict[str, Any]:
        if not path.is_file():
            return {
                "status": "missing",
                "path": path,
                "data": None,
                "message": f"Missing file: {path}",
            }
        try:
            return {
                "status": "present",
                "path": path,
                "data": json.loads(path.read_text(encoding="utf-8")),
                "message": None,
            }
        except (json.JSONDecodeError, OSError) as exc:
            return {
                "status": "error",
                "path": path,
                "data": None,
                "message": str(exc),
            }

    def read_csv_state(path: Path) -> dict[str, Any]:
        if not path.is_file():
            return {
                "status": "missing",
                "path": path,
                "frame": pd.DataFrame(),
                "message": f"Missing CSV: {path}",
            }
        try:
            return {
                "status": "present",
                "path": path,
                "frame": pd.read_csv(path),
                "message": None,
            }
        except Exception as exc:
            return {
                "status": "error",
                "path": path,
                "frame": pd.DataFrame(),
                "message": str(exc),
            }

    def read_run_config_state(path: Path) -> dict[str, Any]:
        if not path.is_file():
            return {
                "status": "missing",
                "path": path,
                "config": None,
                "message": f"Missing run YAML: {path}",
            }
        try:
            return {
                "status": "present",
                "path": path,
                "config": RunConfig.from_yaml(path),
                "message": None,
            }
        except Exception as exc:
            return {
                "status": "error",
                "path": path,
                "config": None,
                "message": str(exc),
            }

    def read_manifest_state(path: Path, dataset_root: Path) -> dict[str, Any]:
        if not path.is_file():
            return {
                "status": "missing",
                "path": path,
                "manifest": None,
                "message": f"Missing manifest: {path}",
            }
        try:
            manifest = DatasetManifest.from_csv(path, dataset_root=dataset_root)
            manifest.validate(check_files_exist=False)
            return {
                "status": "present",
                "path": path,
                "manifest": manifest,
                "message": None,
            }
        except Exception as exc:
            return {
                "status": "error",
                "path": path,
                "manifest": None,
                "message": str(exc),
            }

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

    def artifact_message(title: str, state: dict[str, Any], description: str):
        message = state.get("message")
        details = f"\n\nDetails: `{message}`" if message else ""
        return mo.md(
            f"""
            ### {title}

            {description}

            Status: `{state.get("status", "unknown")}`

            Path: `{state.get("path", "")}`{details}
            """
        )

    def artifact_status_frame(paths: dict[str, Path]) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "artifact": name,
                    "status": "present" if path.exists() else "missing",
                    "path": str(path),
                }
                for name, path in paths.items()
            ]
        )

    def preprocessing_frame(config: object) -> pd.DataFrame:
        preprocessing = getattr(config, "preprocessing", None)
        if preprocessing is None:
            return pd.DataFrame(
                [{"field": "preprocessing", "value": "not defined in this run config"}]
            )
        fields = [
            "source_name",
            "target_name",
            "image_size",
            "grid_movement",
            "margin",
            "mask_strategy",
            "source_mask_strategy",
            "target_mask_strategy",
            "mask_scale",
            "lowres_mask_filtering",
            "tiled_io",
            "min_foreground_ratio",
            "max_white_ratio",
            "white_threshold",
            "max_largest_white_component_ratio",
            "save_masks",
            "save_discarded_patches",
            "seed",
        ]
        return pd.DataFrame(
            [{"field": field, "value": getattr(preprocessing, field)} for field in fields]
        )

    def json_scalar_frame(data: object) -> pd.DataFrame:
        if not isinstance(data, dict):
            return pd.DataFrame()
        rows = []
        for key, value in data.items():
            if isinstance(value, dict | list):
                continue
            rows.append({"field": key, "value": value})
        return pd.DataFrame(rows)

    def alignment_frames(data: object) -> tuple[pd.DataFrame, pd.DataFrame]:
        if not isinstance(data, dict):
            return pd.DataFrame(), pd.DataFrame()
        matrix = data.get("warp_matrix")
        scalar_rows = [
            {"field": key, "value": value} for key, value in data.items() if key != "warp_matrix"
        ]
        matrix_frame = (
            pd.DataFrame(matrix, index=["row_0", "row_1"], columns=["a", "b", "tx"])
            if isinstance(matrix, list) and len(matrix) == 2
            else pd.DataFrame()
        )
        return pd.DataFrame(scalar_rows), matrix_frame

    def manifest_frames(manifest: object) -> tuple[pd.DataFrame, pd.DataFrame]:
        if manifest is None:
            return pd.DataFrame(), pd.DataFrame()
        records = getattr(manifest, "records", ())
        split_rows = [
            {"split": split, "count": sum(1 for record in records if record.split == split)}
            for split in ("train", "val", "test")
        ]
        sample_rows = [
            {
                "sample_id": record.sample_id,
                "split": record.split,
                "x": record.x,
                "y": record.y,
                "width": record.width,
                "height": record.height,
                "input_path": record.input_path.as_posix(),
                "target_path": record.target_path.as_posix(),
            }
            for record in records[:12]
        ]
        return pd.DataFrame(split_rows), pd.DataFrame(sample_rows)

    def csv_summary_frame(state: dict[str, Any], *, name: str) -> pd.DataFrame:
        frame = state.get("frame", pd.DataFrame())
        return pd.DataFrame(
            [
                {
                    "artifact": name,
                    "status": state.get("status", "unknown"),
                    "rows": len(frame) if isinstance(frame, pd.DataFrame) else 0,
                    "columns": ", ".join(frame.columns) if isinstance(frame, pd.DataFrame) else "",
                    "path": str(state.get("path", "")),
                    "message": state.get("message"),
                }
            ]
        )

    def mask_artifact_frame(dataset_root: Path) -> pd.DataFrame:
        root_masks = sorted(dataset_root.glob("mask_*")) if dataset_root.exists() else []
        split_rows = []

        def _limited_foreground_mask_count(split_dir: Path, *, limit: int = 1000) -> int | str:
            count = 0
            for _ in split_dir.glob("*_foreground_mask.*"):
                count += 1
                if count >= limit:
                    return f">= {limit}"
            return count

        for split in ("train", "val", "test"):
            split_dir = dataset_root / "splits" / split
            if not split_dir.is_dir():
                split_rows.append(
                    {
                        "location": f"splits/{split}",
                        "status": "missing",
                        "foreground_mask_files": 0,
                    }
                )
                continue
            split_rows.append(
                {
                    "location": f"splits/{split}",
                    "status": "present",
                    "foreground_mask_files": _limited_foreground_mask_count(split_dir),
                }
            )
        return pd.DataFrame(
            [
                {
                    "location": "dataset root",
                    "status": "present" if root_masks else "missing",
                    "foreground_mask_files": len(root_masks),
                },
                *split_rows,
            ]
        )

    return (
        alignment_frames,
        artifact_message,
        artifact_status_frame,
        csv_summary_frame,
        display_table,
        json_scalar_frame,
        manifest_frames,
        mask_artifact_frame,
        preprocessing_frame,
        read_csv_state,
        read_json_state,
        read_manifest_state,
        read_run_config_state,
        resolve_repo_path,
    )


@app.cell
def _(REPO_ROOT, dataset_root_input, resolve_repo_path, run_yaml_input):
    run_yaml_path = resolve_repo_path(run_yaml_input.value, REPO_ROOT)
    dataset_root_path = resolve_repo_path(dataset_root_input.value, REPO_ROOT)

    alignment_metadata_path = dataset_root_path / "alignment_metadata.json"
    dataset_build_path = dataset_root_path / "metadata" / "dataset_build.json"
    dataset_fingerprint_path = dataset_root_path / "metadata" / "dataset_fingerprint.json"
    manifest_path = dataset_root_path / "manifests" / "manifest.csv"
    discarded_manifest_path = dataset_root_path / "manifests" / "discarded_manifest.csv"
    discarded_log_path = dataset_root_path / "discarded_patches" / "discarded_log.csv"
    return (
        alignment_metadata_path,
        dataset_build_path,
        dataset_fingerprint_path,
        dataset_root_path,
        discarded_log_path,
        discarded_manifest_path,
        manifest_path,
        run_yaml_path,
    )


@app.cell
def _(
    alignment_metadata_path,
    artifact_status_frame,
    dataset_build_path,
    dataset_fingerprint_path,
    discarded_log_path,
    discarded_manifest_path,
    display_table,
    manifest_path,
    mo,
    run_yaml_path,
):
    mo.vstack(
        [
            mo.md("## Artifact Availability"),
            display_table(
                artifact_status_frame(
                    {
                        "run YAML": run_yaml_path,
                        "alignment metadata": alignment_metadata_path,
                        "dataset build metadata": dataset_build_path,
                        "dataset fingerprint": dataset_fingerprint_path,
                        "manifest": manifest_path,
                        "discarded manifest": discarded_manifest_path,
                        "discarded log": discarded_log_path,
                    }
                )
            ),
        ]
    )
    return


@app.cell
def _(
    alignment_metadata_path,
    dataset_build_path,
    dataset_fingerprint_path,
    dataset_root_path,
    discarded_log_path,
    discarded_manifest_path,
    manifest_path,
    read_csv_state,
    read_json_state,
    read_manifest_state,
    read_run_config_state,
    run_yaml_path,
):
    run_config_state = read_run_config_state(run_yaml_path)
    alignment_state = read_json_state(alignment_metadata_path)
    dataset_build_state = read_json_state(dataset_build_path)
    dataset_fingerprint_state = read_json_state(dataset_fingerprint_path)
    manifest_state = read_manifest_state(manifest_path, dataset_root_path)
    discarded_manifest_state = read_csv_state(discarded_manifest_path)
    discarded_log_state = read_csv_state(discarded_log_path)
    return (
        alignment_state,
        dataset_build_state,
        dataset_fingerprint_state,
        discarded_log_state,
        discarded_manifest_state,
        manifest_state,
        run_config_state,
    )


@app.cell
def _(artifact_message, display_table, mo, preprocessing_frame, run_config_state):
    if run_config_state["status"] == "present":
        config_view = mo.vstack(
            [
                mo.md("## Run Configuration"),
                mo.md("Selected preprocessing fields from the active run YAML."),
                display_table(preprocessing_frame(run_config_state["config"])),
            ]
        )
    else:
        config_view = artifact_message(
            "Run Configuration",
            run_config_state,
            "Provide the run YAML used for preparation to inspect config-driven settings.",
        )
    mo.output.replace(config_view)
    return


@app.cell
def _(alignment_frames, alignment_state, artifact_message, display_table, mo):
    alignment_metrics, affine_matrix = alignment_frames(alignment_state["data"])
    if alignment_state["status"] == "present":
        alignment_view = mo.vstack(
            [
                mo.md("## Alignment Metadata"),
                mo.md(
                    "`alignment_metadata.json` records keypoint counts, match counts, "
                    "RANSAC inliers, inlier ratio, transform diagnostics, the affine "
                    "matrix, and optional foreground mask IoU."
                ),
                display_table(alignment_metrics),
                mo.md("### Stored Warp Matrix"),
                display_table(affine_matrix),
            ]
        )
    else:
        alignment_view = artifact_message(
            "Alignment Metadata",
            alignment_state,
            "`vs-prepare` writes this file at the dataset root after alignment.",
        )
    mo.output.replace(alignment_view)
    return


@app.cell
def _(
    artifact_message,
    dataset_build_state,
    dataset_fingerprint_state,
    display_table,
    json_scalar_frame,
    mo,
):
    sections = [mo.md("## Dataset Metadata")]
    if dataset_build_state["status"] == "present":
        sections.extend(
            [
                mo.md("### Dataset Build"),
                display_table(json_scalar_frame(dataset_build_state["data"])),
            ]
        )
    else:
        sections.append(
            artifact_message(
                "Dataset Build",
                dataset_build_state,
                "`metadata/dataset_build.json` summarizes accepted and discarded patches.",
            )
        )
    if dataset_fingerprint_state["status"] == "present":
        sections.extend(
            [
                mo.md("### Dataset Fingerprint"),
                display_table(json_scalar_frame(dataset_fingerprint_state["data"])),
            ]
        )
    else:
        sections.append(
            artifact_message(
                "Dataset Fingerprint",
                dataset_fingerprint_state,
                "`metadata/dataset_fingerprint.json` records prepare cache identity.",
            )
        )
    mo.output.replace(mo.vstack(sections))
    return


@app.cell
def _(artifact_message, display_table, manifest_frames, manifest_state, mo):
    split_counts, sample_rows = manifest_frames(manifest_state["manifest"])
    if manifest_state["status"] == "present":
        manifest_view = mo.vstack(
            [
                mo.md("## Accepted Patch Manifest"),
                mo.md(
                    "`manifests/manifest.csv` is the downstream training and evaluation "
                    "contract. Paths are relative to `dataset_root`; the manifest does "
                    "not point to a saved full aligned target frame."
                ),
                display_table(split_counts),
                mo.md("### First Accepted Patches"),
                display_table(sample_rows, max_height=420),
            ]
        )
    else:
        manifest_view = artifact_message(
            "Accepted Patch Manifest",
            manifest_state,
            "`manifests/manifest.csv` is missing or malformed.",
        )
    mo.output.replace(manifest_view)
    return


@app.cell
def _(
    csv_summary_frame,
    discarded_log_state,
    discarded_manifest_state,
    display_table,
    mo,
):
    mo.vstack(
        [
            mo.md("## Discarded Patch Diagnostics"),
            mo.md(
                "These artifacts are optional inspection outputs. The manifest records "
                "discarded patch coordinates; the log records foreground and white-ratio "
                "filter decisions."
            ),
            display_table(
                csv_summary_frame(
                    discarded_manifest_state,
                    name="manifests/discarded_manifest.csv",
                )
            ),
            display_table(
                csv_summary_frame(
                    discarded_log_state,
                    name="discarded_patches/discarded_log.csv",
                )
            ),
        ]
    )
    return


@app.cell
def _(dataset_root_path, display_table, mask_artifact_frame, mo):
    mo.vstack(
        [
            mo.md("## Optional Mask Artifact Presence"),
            mo.md(
                "When `preprocessing.save_masks` is enabled, preparation may save root "
                "`mask_*` files and accepted patch sidecar masks. This app only counts "
                "their presence; it does not load or regenerate masks."
            ),
            display_table(mask_artifact_frame(dataset_root_path)),
        ]
    )
    return


@app.cell
def _(mo):
    mo.output.replace(
        mo.md(
            "## Read-only Boundary\n\n"
            "Prepare datasets outside Marimo:\n\n"
            "```bash\n"
            "vs-prepare --config config/runs/local/my_run.yaml\n"
            "```\n\n"
            "This inspector reads existing files only:\n\n"
            "```text\n"
            "DATASET_ROOT/alignment_metadata.json\n"
            "DATASET_ROOT/metadata/dataset_build.json\n"
            "DATASET_ROOT/metadata/dataset_fingerprint.json\n"
            "DATASET_ROOT/manifests/manifest.csv\n"
            "DATASET_ROOT/manifests/discarded_manifest.csv\n"
            "DATASET_ROOT/discarded_patches/discarded_log.csv\n"
            "```\n\n"
            "Conceptual demo visuals live in `docs/marimo/research/alignment.py`."
        )
    )
    return


if __name__ == "__main__":
    app.run()
