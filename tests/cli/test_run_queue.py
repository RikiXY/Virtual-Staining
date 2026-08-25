from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.config_helpers import write_queue_config, write_run_config, write_yaml
from virtual_staining import cli
from virtual_staining.applications.run_queue import load_local_run_queue, run_queue


def _write_config(tmp_path: Path, section_yaml: str) -> Path:
    return write_run_config(tmp_path, section_yaml)


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

    queue = load_local_run_queue(queue_path)

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
        load_local_run_queue(queue_path)


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
        load_local_run_queue(queue_path)


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
        load_local_run_queue(queue_path)


def test_run_queue_executes_jobs_in_order_and_persists_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_a = _write_config(tmp_path / "a", "training:\n  epochs: 1\n")
    config_b = _write_config(tmp_path / "b", "training:\n  epochs: 1\n")
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
            label: baseline
          - config_path: {config_b}
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
        "training:\n  epochs: 1\n  lr_g: 0.0002\n",
        filename="a.yaml",
        dataset_root=dataset_root,
        results_path=results_path,
        run_name="a",
    )
    config_b = write_run_config(
        tmp_path / "configs",
        "training:\n  epochs: 1\n  lr_g: 0.0001\n",
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
          - config_path: {config_b}
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
          - config_path: {config_b}
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
    config_a = _write_config(tmp_path / "a", "training:\n  epochs: 1\n")
    config_b = _write_config(tmp_path / "b", "training:\n  epochs: 1\n")
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
    config_a = _write_config(tmp_path / "a", "training:\n  epochs: 1\n")
    config_b = _write_config(tmp_path / "b", "training:\n  epochs: 1\n")
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
    config_a = _write_config(tmp_path / "a", "training:\n  epochs: 1\n")
    invalid_config = write_yaml(
        tmp_path / "b" / "run.yaml",
        """
        dataset_root: /tmp/data
        results_path: /tmp/results
        run_name: invalid
        training:
          epochs: 1
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
    config = _write_config(tmp_path / "a", "training:\n  epochs: 1\n")
    queue_path = _write_queue(
        tmp_path,
        f"""\
          - config_path: {config}
            label: train_only
            stages: [train, infer, evaluate]
        """,
        continue_on_failure=False,
    )

    queue = load_local_run_queue(queue_path)

    assert queue.jobs[0].stages == ("train", "infer", "evaluate")


def test_load_local_run_queue_rejects_unknown_stage(tmp_path: Path) -> None:
    config = _write_config(tmp_path / "a", "training:\n  epochs: 1\n")
    queue_path = _write_queue(
        tmp_path,
        f"""\
          - config_path: {config}
            stages: [train, banana, evaluate]
        """,
        continue_on_failure=False,
    )

    with pytest.raises(ValueError, match="unknown stage"):
        load_local_run_queue(queue_path)
