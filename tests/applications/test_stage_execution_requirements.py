"""All selected configuration requirements fail before application side effects."""

from pathlib import Path

import pytest

from tests.config_helpers import prepare_config_data, write_config_data
from virtual_staining.applications.pipeline import run_stages
from virtual_staining.config.run import RunConfig


@pytest.mark.parametrize(
    "stages", [("prepare", "train"), ("prepare", "infer"), ("prepare", "evaluate"), None]
)
def test_union_requirements_fail_before_any_artifacts(
    tmp_path: Path, stages: tuple[str, ...] | None
) -> None:
    raw = prepare_config_data(tmp_path)
    raw.update(model={"inputs": ["LF"], "outputs": ["HE"]})
    path = write_config_data(tmp_path / "run.yaml", raw)
    with pytest.raises(ValueError, match="results_path"):
        run_stages(path, stages) if stages is not None else run_stages(path)
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("later", ["train", "infer", "evaluate"])
def test_later_section_error_precedes_prepare(tmp_path: Path, later: str) -> None:
    raw = prepare_config_data(tmp_path)
    raw.update(
        results_path=str(tmp_path / "results"),
        run_name="test",
        model={"inputs": ["LF"], "outputs": ["HE"]},
    )
    if later == "evaluate":
        raw["evaluation"] = {"metrics": [{"name": "invalid_metric"}]}
    path = write_config_data(tmp_path / "run.yaml", raw)
    with pytest.raises(
        ValueError,
        match={"train": "training", "infer": "inference", "evaluate": "invalid_metric"}[later],
    ):
        run_stages(path, ("prepare", later))
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("supplied_method", [False, True])
def test_unpaired_prepare_is_explicitly_unsupported_without_dummy_domains(
    tmp_path: Path, supplied_method: bool
) -> None:
    raw = prepare_config_data(tmp_path)
    raw["data"] = {"pairing": "unpaired"}
    # Inspection accepts the declaration; selected preparation owns the support boundary.
    config = RunConfig.from_mapping(raw)
    assert config.data.domains == {}
    if supplied_method:
        raw["method"] = {"name": "cyclegan"}
    path = write_config_data(tmp_path / "run.yaml", raw)
    with pytest.raises(ValueError, match="prepare.*unpaired.*unsupported"):
        run_stages(path, ("prepare",))
    assert list(tmp_path.iterdir()) == [path]


def test_direct_application_checks_its_own_requirements_before_writes(tmp_path: Path) -> None:
    from virtual_staining.applications.infer import infer
    from virtual_staining.applications.prepare import prepare

    raw = prepare_config_data(tmp_path)
    raw["data"] = {"pairing": "unpaired"}
    path = write_config_data(tmp_path / "run.yaml", raw)
    config = RunConfig.from_mapping(raw)
    with pytest.raises(ValueError, match="prepare.*unpaired.*unsupported"):
        prepare(config, path)
    raw.pop("data")
    raw.update(results_path=str(tmp_path / "results"), run_name="test")
    raw["model"] = {"inputs": ["LF"], "outputs": ["HE"]}
    raw["inference"] = {"checkpoint_policy": "latest"}
    config = RunConfig.from_mapping(raw, stages=("evaluate",))
    with pytest.raises(ValueError, match="model.generator.*resolve with stages"):
        infer(config, path)
    assert list(tmp_path.iterdir()) == [path]
