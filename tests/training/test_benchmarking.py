from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from virtual_staining.training.benchmarking import (
    TrainingBenchmarkRecorder,
    build_benchmark_metadata,
)


def test_recorder_excludes_warmup_and_reports_phase_breakdown(tmp_path: Path) -> None:
    recorder = TrainingBenchmarkRecorder(torch.device("cpu"), warmup_batches=1)
    recorder.set_workload(batch_size=2, train_sample_count=4)
    recorder.start_run()
    recorder.start_epoch(0)

    for batch_index in range(2):
        cycle_started = time.perf_counter()
        recorder.batch_received(
            epoch=0,
            batch_index=batch_index,
            samples=2,
            data_wait_seconds=0.001,
            cycle_started_at=cycle_started,
        )
        with recorder.phase("h2d"):
            torch.zeros(1)
        with recorder.phase("discriminator_update"):
            torch.ones(1)
        with recorder.phase("generator_update"):
            _ = torch.ones(1) + 1
        recorder.finish_batch()

    with recorder.phase("validation"):
        torch.zeros(1)
    with recorder.phase("preview_io"):
        torch.zeros(1)
    with recorder.phase("checkpoint"):
        torch.zeros(1)
    recorder.finish_epoch(0)
    recorder.finish_run()

    report = recorder.report(metadata={"test": True})
    summary = report["summary"]
    assert report["representative_cuda_baseline"] is False
    assert summary["steady_state_batches"] == 1
    assert summary["steady_state_samples"] == 2
    assert summary["samples_per_second"] is not None
    assert summary["median_batch_latency_ms"] is not None
    assert summary["phases"]["h2d"]["count"] == 1
    assert summary["phases"]["discriminator_update"]["count"] == 1
    assert summary["phases"]["generator_update"]["count"] == 1
    assert summary["phases"]["validation"]["count"] == 1
    assert summary["phases"]["preview_io"]["count"] == 1
    assert summary["phases"]["checkpoint"]["count"] == 1
    assert summary["bottleneck"]["classification"] == "not-classified-without-cuda"

    output = recorder.write_report(tmp_path / "report.json", metadata={"test": True})
    assert json.loads(output.read_text(encoding="utf-8"))["metadata"] == {"test": True}


def test_benchmark_metadata_records_config_and_environment(tmp_path: Path) -> None:
    config_path = tmp_path / "run.yaml"
    config_path.write_text("run_name: benchmark\n", encoding="utf-8")

    metadata = build_benchmark_metadata(
        config_path,
        run_config={"run_name": "benchmark", "training": {"batch_size": 8}},
    )

    assert metadata["config_path"] == str(config_path.resolve())
    assert len(metadata["config_sha256"]) == 64
    assert metadata["resolved_run_config"]["training"]["batch_size"] == 8
    assert metadata["torch_version"] == torch.__version__
    assert metadata["device"]["type"] in {"cpu", "cuda"}
    assert "cudnn_deterministic" in metadata["determinism"]
