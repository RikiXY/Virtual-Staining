from __future__ import annotations

from pathlib import Path

from virtual_staining.experiment.run_layout import RunLayout


def test_snapshot_paths_are_stage_owned(tmp_path: Path) -> None:
    paths = RunLayout(tmp_path / "run").stage("infer")
    assert paths.input_config == tmp_path / "run" / "config" / "infer" / "input.yaml"
    assert paths.resolved_config == tmp_path / "run" / "config" / "infer" / "resolved.yaml"
    assert paths.environment == tmp_path / "run" / "metadata" / "environments" / "infer.json"
