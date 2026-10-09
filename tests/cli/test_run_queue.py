from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from tests.config_helpers import (
    pix2pix_config_data,
    prepare_config_data,
    write_config_data,
    write_queue_config,
    write_run_config,
    write_yaml,
)
from virtual_staining import cli
from virtual_staining.applications.run_queue import _load_local_run_queue, run_queue

MINIMAL_TRAINING_YAML = (
    "training:\n  epochs: 1\n  losses:\n    generator: []\n    discriminator: []\n"
)


def _write_config(tmp_path: Path, section_yaml: str) -> Path:
    path = write_run_config(tmp_path, section_yaml)
    raw = yaml.safe_load(path.read_text())
    raw["preprocessing"] = prepare_config_data(tmp_path)["preprocessing"]
    raw["preprocessing"]["inputs"].update(
        modalities=["label_free"], reference="label_free", target_modalities=["stained"]
    )
    raw["inference"] = {"checkpoint_policy": "latest"}
    return write_config_data(path, raw)


def _write_queue(tmp_path: Path, jobs_yaml: str, *, continue_on_failure: bool = False) -> Path:
    return write_queue_config(
        tmp_path,
        jobs_yaml,
        continue_on_failure=continue_on_failure,
    )


def test_run_queue_main_passes_queue_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = _write_config(
        tmp_path,
        """\
        training:
          epochs: 1
        """,
    )
    queue_path = _write_queue(
        tmp_path,
        f"""\
          - config_path: {config_path}
        """,
    )
    captured: dict[str, object] = {}

    def _fake_run_queue(incoming_path: Path, **kwargs: object) -> object:
        captured["queue_path"] = incoming_path
        return SimpleNamespace(status="completed")

    monkeypatch.setattr(cli, "run_queue", _fake_run_queue)

    cli.main(["queue", "--queue", str(queue_path)])

    assert captured["queue_path"] == queue_path.resolve()


def test_load_local_run_queue_resolves_relative_job_paths(tmp_path: Path) -> None:
    config_path = write_run_config(
        tmp_path / "configs",
        filename="job.yaml",
        dataset_root=Path("/tmp/data"),
        results_path=Path("/tmp/results"),
        run_name="demo",
    )
    queue_path = _write_queue(
        tmp_path,
        """\
          - config_path: ../../configs/job.yaml
            label: baseline
            notes: first run
        """,
    )

    queue = _load_local_run_queue(queue_path)

    assert queue.name == "nightly"
    assert queue.continue_on_failure is False
    assert queue.jobs[0].config_path == config_path.resolve()
    assert queue.jobs[0].label == "baseline"
    assert queue.jobs[0].notes == "first run"


def test_load_local_run_queue_rejects_unknown_top_level_keys(tmp_path: Path) -> None:
    queue_path = write_yaml(
        tmp_path / "config" / "queues" / "nightly.yaml",
        """
        name: nightly
        continue_on_failure: false
        unexpected: true
        jobs:
          - config_path: ../runs/local/run_a.yaml
        """,
    )

    with pytest.raises(ValueError, match=r"Unknown key\(s\) in queue: unexpected"):
        _load_local_run_queue(queue_path)


def test_load_local_run_queue_rejects_unknown_job_keys(tmp_path: Path) -> None:
    queue_path = write_yaml(
        tmp_path / "config" / "queues" / "nightly.yaml",
        """
        name: nightly
        continue_on_failure: false
        jobs:
          - config_path: ../runs/local/run_a.yaml
            unexpected: true
        """,
    )

    with pytest.raises(ValueError, match=r"Unknown key\(s\) in queue\.jobs\[0\]: unexpected"):
        _load_local_run_queue(queue_path)


def test_load_local_run_queue_requires_yaml_boolean_for_continue_on_failure(
    tmp_path: Path,
) -> None:
    queue_path = write_yaml(
        tmp_path / "config" / "queues" / "nightly.yaml",
        """
        name: nightly
        continue_on_failure: "false"
        jobs:
          - config_path: ../runs/local/run_a.yaml
        """,
    )

    with pytest.raises(TypeError, match="continue_on_failure"):
        _load_local_run_queue(queue_path)


def test_run_queue_executes_jobs_in_order_and_persists_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_a = _write_config(tmp_path / "a", MINIMAL_TRAINING_YAML)
    config_b = _write_config(tmp_path / "b", MINIMAL_TRAINING_YAML)
    queue_path = _write_queue(
        tmp_path,
        f"""\
          - config_path: {config_a}
            label: first
          - config_path: {config_b}
            label: second
        """,
        continue_on_failure=False,
    )
    calls: list[tuple[Path, tuple[str, ...]]] = []

    def _fake_run_stages(config_path: Path, stages: Sequence[str], **kwargs: object) -> None:
        calls.append((config_path, tuple(stages)))

    monkeypatch.setattr(
        "virtual_staining.applications.run_queue.run_stages",
        _fake_run_stages,
    )

    state = run_queue(queue_path)
    state_path = tmp_path / "local_workspace" / "queues" / "nightly.state.json"
    state_data = json.loads(state_path.read_text(encoding="utf-8"))

    assert calls == [
        (config_a.resolve(), ("prepare", "train", "infer", "evaluate")),
        (config_b.resolve(), ("prepare", "train", "infer", "evaluate")),
    ]
    assert state.status == "completed"
    assert state_data["status"] == "completed"
    assert [job["status"] for job in state_data["jobs"]] == ["completed", "completed"]
    assert state_path.exists()
    assert not (tmp_path / "local_workspace" / "queues" / "nightly.ablation.summary.json").exists()


def test_run_queue_ablation_validation_passes_and_writes_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_root = tmp_path / "data"
    results_path = tmp_path / "results"
    config_a = write_run_config(
        tmp_path / "configs",
        """
        training:
          epochs: 1
          losses:
            generator:
              - name: adversarial_bce
                weight: 1.0
              - name: l1
                weight: 25.0
            discriminator:
              - name: adversarial_bce
                weight: 1.0
        """,
        filename="baseline.yaml",
        dataset_root=dataset_root,
        results_path=results_path,
        run_name="ablation_baseline",
    )
    config_b = write_run_config(
        tmp_path / "configs",
        """
        training:
          epochs: 1
          losses:
            generator:
              - name: ssim
                weight: 1.0
                params:
                  window_size: 3
            discriminator: []
        """,
        filename="ssim_only.yaml",
        dataset_root=dataset_root,
        results_path=results_path,
        run_name="ablation_ssim_only",
    )
    queue_path = write_yaml(
        tmp_path / "config" / "queues" / "loss_ablation.yaml",
        f"""
        name: loss_ablation
        continue_on_failure: false
        ablation:
          fixed_fields:
            - model.generator.base_channels
            - training.epochs
          variable_fields:
            - run_name
            - training.losses.generator
            - training.losses.discriminator
        jobs:
          - config_path: {config_a}
            stages: [train]
            label: baseline
          - config_path: {config_b}
            stages: [train]
            label: ssim_only
        """,
    )
    calls: list[Path] = []

    def _fake_run_stages(config_path: Path, stages: Sequence[str], **kwargs: object) -> None:
        del stages
        calls.append(config_path)

    monkeypatch.setattr(
        "virtual_staining.applications.run_queue.run_stages",
        _fake_run_stages,
    )

    state = run_queue(queue_path)

    summary_path = tmp_path / "local_workspace" / "queues" / "loss_ablation.ablation.summary.json"
    state_path = tmp_path / "local_workspace" / "queues" / "loss_ablation.state.json"
    state_data = json.loads(state_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert state.status == "completed"
    assert state_data["ablation_summary_path"] == str(summary_path)
    assert calls == [config_a.resolve(), config_b.resolve()]
    assert summary["queue_name"] == "loss_ablation"
    assert summary["fixed_values"]["training.epochs"] == 1
    assert summary["jobs"][0]["run_name"] == "ablation_baseline"
    assert summary["jobs"][1]["run_name"] == "ablation_ssim_only"
    assert summary["jobs"][0]["variable_values"]["training.losses.generator"][0]["name"] == (
        "adversarial_bce"
    )
    assert summary["jobs"][1]["variable_values"]["training.losses.discriminator"] == []
    assert summary["jobs"][0]["config_hash"].startswith("sha256:")


def test_run_queue_ablation_validation_fails_on_undeclared_difference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_root = tmp_path / "data"
    results_path = tmp_path / "results"
    config_a = write_run_config(
        tmp_path / "configs",
        """        training:
          epochs: 1
          lr_g: 0.0002
          losses:
            generator: []
            discriminator: []
        """,
        filename="a.yaml",
        dataset_root=dataset_root,
        results_path=results_path,
        run_name="a",
    )
    config_b = write_run_config(
        tmp_path / "configs",
        """        training:
          epochs: 1
          lr_g: 0.0001
          losses:
            generator: []
            discriminator: []
        """,
        filename="b.yaml",
        dataset_root=dataset_root,
        results_path=results_path,
        run_name="b",
    )
    queue_path = write_yaml(
        tmp_path / "config" / "queues" / "bad_ablation.yaml",
        f"""
        name: bad_ablation
        continue_on_failure: false
        ablation:
          variable_fields:
            - run_name
        jobs:
          - config_path: {config_a}
            stages: [train]
          - config_path: {config_b}
            stages: [train]
        """,
    )
    calls: list[Path] = []
    monkeypatch.setattr(
        "virtual_staining.applications.run_queue.run_stages",
        lambda config_path, stages, **kwargs: calls.append(config_path),
    )

    state = run_queue(queue_path)

    state_data = json.loads(
        (tmp_path / "local_workspace" / "queues" / "bad_ablation.state.json").read_text(
            encoding="utf-8"
        )
    )
    assert calls == []
    assert state.status == "failed"
    assert state_data["jobs"][0]["status"] == "failed"
    assert "training.lr_g" in state_data["jobs"][0]["error"]
    assert "ablation.variable_fields" in state_data["jobs"][0]["error"]


def test_run_queue_ablation_canonicalizes_loss_list_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_root = tmp_path / "data"
    results_path = tmp_path / "results"
    config_a = write_run_config(
        tmp_path / "configs",
        """
        training:
          epochs: 1
          losses:
            generator:
              - name: adversarial_bce
                weight: 1.0
              - name: l1
                weight: 25.0
        """,
        filename="a.yaml",
        dataset_root=dataset_root,
        results_path=results_path,
        run_name="a",
    )
    config_b = write_run_config(
        tmp_path / "configs",
        """
        training:
          epochs: 1
          losses:
            generator:
              - name: l1
                weight: 25.0
              - name: adversarial_bce
                weight: 1.0
        """,
        filename="b.yaml",
        dataset_root=dataset_root,
        results_path=results_path,
        run_name="b",
    )
    queue_path = write_yaml(
        tmp_path / "config" / "queues" / "ordered_losses.yaml",
        f"""
        name: ordered_losses
        continue_on_failure: false
        ablation:
          variable_fields:
            - run_name
        jobs:
          - config_path: {config_a}
            stages: [train]
          - config_path: {config_b}
            stages: [train]
        """,
    )
    monkeypatch.setattr(
        "virtual_staining.applications.run_queue.run_stages",
        lambda config_path, stages, **kwargs: None,
    )

    state = run_queue(queue_path)

    assert state.status == "completed"


def test_run_queue_stops_on_failure_when_continue_on_failure_is_false(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_a = _write_config(tmp_path / "a", MINIMAL_TRAINING_YAML)
    config_b = _write_config(tmp_path / "b", MINIMAL_TRAINING_YAML)
    queue_path = _write_queue(
        tmp_path,
        f"""\
          - config_path: {config_a}
          - config_path: {config_b}
        """,
        continue_on_failure=False,
    )
    calls: list[Path] = []

    def _fake_run_stages(config_path: Path, stages: Sequence[str], **kwargs: object) -> None:
        del stages
        calls.append(config_path)
        raise RuntimeError("boom")

    monkeypatch.setattr(
        "virtual_staining.applications.run_queue.run_stages",
        _fake_run_stages,
    )

    state = run_queue(queue_path)
    state_data = json.loads(
        (tmp_path / "local_workspace" / "queues" / "nightly.state.json").read_text(encoding="utf-8")
    )

    assert calls == [config_a.resolve()]
    assert state.status == "failed"
    assert [job["status"] for job in state_data["jobs"]] == ["failed", "pending"]
    assert state_data["jobs"][0]["error"] == "boom"


def test_run_queue_continues_after_failure_when_configured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_a = _write_config(tmp_path / "a", MINIMAL_TRAINING_YAML)
    config_b = _write_config(tmp_path / "b", MINIMAL_TRAINING_YAML)
    queue_path = _write_queue(
        tmp_path,
        f"""\
          - config_path: {config_a}
          - config_path: {config_b}
        """,
        continue_on_failure=True,
    )
    calls: list[Path] = []

    def _fake_run_stages(config_path: Path, stages: Sequence[str], **kwargs: object) -> None:
        del stages
        calls.append(config_path)
        if config_path == config_a.resolve():
            raise RuntimeError("boom")

    monkeypatch.setattr(
        "virtual_staining.applications.run_queue.run_stages",
        _fake_run_stages,
    )

    state = run_queue(queue_path)
    state_data = json.loads(
        (tmp_path / "local_workspace" / "queues" / "nightly.state.json").read_text(encoding="utf-8")
    )

    assert calls == [config_a.resolve(), config_b.resolve()]
    assert state.status == "failed"
    assert [job["status"] for job in state_data["jobs"]] == ["failed", "completed"]
    assert state_data["continue_on_failure"] is True


def test_run_queue_preflights_configs_before_running_any_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_a = _write_config(tmp_path / "a", MINIMAL_TRAINING_YAML)
    invalid_config = write_yaml(
        tmp_path / "b" / "run.yaml",
        """
        dataset_root: /tmp/data
        results_path: /tmp/results
        run_name: invalid
        training:
          epochs: 1
          losses:
            generator: []
            discriminator: []
        unexpected: true
        """,
    )
    queue_path = _write_queue(
        tmp_path,
        f"""\
          - config_path: {config_a}
          - config_path: {invalid_config}
        """,
        continue_on_failure=False,
    )
    calls: list[Path] = []

    def _fake_run_stages(config_path: Path, stages: Sequence[str], **kwargs: object) -> None:
        del stages
        calls.append(config_path)

    monkeypatch.setattr(
        "virtual_staining.applications.run_queue.run_stages",
        _fake_run_stages,
    )

    state = run_queue(queue_path)
    state_data = json.loads(
        (tmp_path / "local_workspace" / "queues" / "nightly.state.json").read_text(encoding="utf-8")
    )

    assert calls == []
    assert state.status == "failed"
    assert [job["status"] for job in state_data["jobs"]] == ["pending", "failed"]
    assert state_data["jobs"][1]["error"].startswith("Queue preflight failed for job 1")
    assert "Unknown key(s) in top level: unexpected" in state_data["jobs"][1]["error"]


def test_load_local_run_queue_reads_configurable_stages(tmp_path: Path) -> None:
    config = _write_config(tmp_path / "a", MINIMAL_TRAINING_YAML)
    queue_path = _write_queue(
        tmp_path,
        f"""\
          - config_path: {config}
            label: train_only
            stages: [train, infer, evaluate]
        """,
        continue_on_failure=False,
    )

    queue = _load_local_run_queue(queue_path)

    assert queue.jobs[0].stages == ("train", "infer", "evaluate")


def test_load_local_run_queue_rejects_unknown_stage(tmp_path: Path) -> None:
    config = _write_config(tmp_path / "a", MINIMAL_TRAINING_YAML)
    queue_path = _write_queue(
        tmp_path,
        f"""\
          - config_path: {config}
            stages: [train, banana, evaluate]
        """,
        continue_on_failure=False,
    )

    with pytest.raises(ValueError, match="unknown stage"):
        _load_local_run_queue(queue_path)


def test_mixed_operation_queue_resolves_each_job_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from virtual_staining.applications.run_queue import _preflight_run_configs
    from virtual_staining.config.run import RunConfig

    jobs = []
    for index, stages in enumerate(
        [("prepare",), ("train",), ("infer",), ("evaluate",), ("train", "infer")]
    ):
        raw = prepare_config_data(tmp_path)
        if stages != ("prepare",):
            raw = pix2pix_config_data(tmp_path)
            if "train" not in stages:
                raw.pop("training")
                raw["model"].pop("discriminator")
            if "infer" in stages:
                raw["inference"] = {"checkpoint_policy": "latest"}
            if stages == ("evaluate",):
                raw["model"].pop("generator")
        path = write_config_data(tmp_path / f"job{index}.yaml", raw)
        jobs.append({"config_path": path.name, "stages": list(stages)})
    queue_path = write_config_data(tmp_path / "queue.yaml", {"name": "mixed", "jobs": jobs})
    queue = _load_local_run_queue(queue_path)
    configs = _preflight_run_configs(queue)
    calls = []

    def execute(path: Path, stages: Sequence[str], **kwargs: object) -> None:
        config = RunConfig.from_yaml(path, stages=stages)
        assert config.resolved_yaml() == configs[len(calls)].resolved_yaml()
        calls.append(tuple(stages))

    monkeypatch.setattr("virtual_staining.applications.run_queue.run_stages", execute)
    state = run_queue(queue_path)
    assert state.status == "completed"
    assert calls == [job.stages for job in queue.jobs]
    assert not (tmp_path / "dataset").exists() and not (tmp_path / "results").exists()


@pytest.mark.parametrize("failure", ["default_full", "train_infer", "unpaired"])
@pytest.mark.parametrize("continue_on_failure", [False, True])
def test_third_job_configuration_failure_prevents_all_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str, continue_on_failure: bool
) -> None:
    raw = prepare_config_data(tmp_path)
    valid = write_config_data(tmp_path / "prepare.yaml", raw)
    stages = None
    if failure == "train_infer":
        raw = pix2pix_config_data(tmp_path)
        stages = ["train", "infer"]
    elif failure == "unpaired":
        raw["data"] = {"pairing": "unpaired"}
        stages = ["prepare"]
    invalid = write_config_data(tmp_path / "invalid.yaml", raw)
    job: dict[str, Any] = {"config_path": str(invalid)}
    if stages:
        job["stages"] = stages
    queue_path = write_config_data(
        tmp_path / "queue.yaml",
        {
            "name": "blocked",
            "continue_on_failure": continue_on_failure,
            "jobs": [{"config_path": str(valid), "stages": ["prepare"]}] * 2 + [job],
        },
    )
    calls = []
    monkeypatch.setattr(
        "virtual_staining.applications.run_queue.run_stages",
        lambda *args, **kwargs: calls.append(args),
    )
    state = run_queue(queue_path)
    assert calls == []
    assert state.status == "failed"
    assert [job.status for job in state.jobs] == ["pending", "pending", "failed"]
    assert all(job.started_at is None for job in state.jobs)
    error = state.jobs[2].error or ""
    assert "job 2" in error
    assert {
        "default_full": "model requires inputs",
        "train_infer": "inference",
        "unpaired": "unsupported",
    }[failure] in error
    assert not (tmp_path / "dataset").exists() and not (tmp_path / "results").exists()


@pytest.mark.parametrize("kind", ["fixed_fields", "variable_fields"])
@pytest.mark.parametrize("field", ["run_name", "model.generator", "training.epochs"])
def test_ablation_rejects_absent_fields_on_the_responsible_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str, field: str
) -> None:
    full = pix2pix_config_data(tmp_path)
    first = write_config_data(tmp_path / "train.yaml", full)
    second = write_config_data(tmp_path / "prepare.yaml", prepare_config_data(tmp_path))
    ablation = {"variable_fields": ["dataset_root"], kind: [field]}
    queue_path = write_config_data(
        tmp_path / "queue.yaml",
        {
            "name": "absent",
            "ablation": ablation,
            "jobs": [
                {"config_path": str(first), "stages": ["train"]},
                {"config_path": str(second), "stages": ["prepare"]},
            ],
        },
    )
    calls = []
    monkeypatch.setattr(
        "virtual_staining.applications.run_queue.run_stages",
        lambda *args, **kwargs: calls.append(args),
    )
    state = run_queue(queue_path)
    assert calls == [] and state.status == "failed"
    assert [job.status for job in state.jobs] == ["pending", "failed"]
    error = state.jobs[1].error or ""
    assert field in error and "absent or inapplicable" in error and "stages prepare" in error
    assert not _load_local_run_queue(queue_path).ablation_summary_path.exists()


def test_prepare_ablation_does_not_invent_run_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = write_config_data(tmp_path / "prepare.yaml", prepare_config_data(tmp_path))
    queue_path = write_config_data(
        tmp_path / "queue.yaml",
        {
            "name": "prepare",
            "ablation": {"variable_fields": ["preprocessing.split.seed"]},
            "jobs": [{"config_path": str(path), "stages": ["prepare"]}],
        },
    )
    monkeypatch.setattr("virtual_staining.applications.run_queue.run_stages", lambda *a, **k: None)
    assert run_queue(queue_path).status == "completed"
    summary = json.loads(_load_local_run_queue(queue_path).ablation_summary_path.read_text())
    assert "run_name" not in summary["jobs"][0]
