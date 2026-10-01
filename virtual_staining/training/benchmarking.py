from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import statistics
import subprocess
import threading
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass
class _PhaseMeasurement:
    name: str
    wall_seconds: float
    batch_sequence: int | None
    cuda_start: Any = None
    cuda_end: Any = None


@dataclass
class _BatchMeasurement:
    sequence: int
    epoch: int
    batch_index: int
    samples: int
    data_wait_seconds: float
    cycle_started_at: float
    cycle_seconds: float | None = None


class _GpuUtilizationSampler:
    def __init__(self, device_index: int, interval_seconds: float) -> None:
        self.device_index = device_index
        self.interval_seconds = interval_seconds
        self.samples: list[float] = []
        self._process: subprocess.Popen[str] | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        executable = shutil.which("nvidia-smi")
        if executable is None:
            return
        interval_ms = max(100, int(self.interval_seconds * 1000))
        try:
            self._process = subprocess.Popen(
                [
                    executable,
                    "-i",
                    str(self.device_index),
                    "--query-gpu=utilization.gpu",
                    "--format=csv,noheader,nounits",
                    f"--loop-ms={interval_ms}",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
        except OSError:
            self._process = None
            return
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()

    def _read(self) -> None:
        process = self._process
        if process is None or process.stdout is None:
            return
        for line in process.stdout:
            try:
                self.samples.append(float(line.strip()))
            except ValueError:
                continue

    def stop(self) -> None:
        if self._process is not None:
            self._process.terminate()
            try:
                self._process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=2.0)
        if self._thread is not None:
            self._thread.join(timeout=2.0)


class TrainingBenchmarkRecorder:
    def __init__(
        self,
        device: torch.device,
        *,
        warmup_batches: int = 10,
        gpu_sample_interval_seconds: float = 0.5,
        profile_trace_path: Path | None = None,
        profile_batches: int = 5,
    ) -> None:
        if warmup_batches < 0:
            raise ValueError("warmup_batches must be greater than or equal to 0")
        if gpu_sample_interval_seconds <= 0:
            raise ValueError("gpu_sample_interval_seconds must be greater than 0")
        if profile_batches <= 0:
            raise ValueError("profile_batches must be greater than 0")
        self.device = device
        self.warmup_batches = warmup_batches
        self.gpu_sample_interval_seconds = gpu_sample_interval_seconds
        self.profile_trace_path = profile_trace_path
        self.profile_batches = profile_batches
        self._phases: list[_PhaseMeasurement] = []
        self._batches: list[_BatchMeasurement] = []
        self._epoch_starts: dict[int, float] = {}
        self._epoch_seconds: list[float] = []
        self._current_batch_sequence: int | None = None
        self._run_start_wall: float | None = None
        self._run_start_cpu: float | None = None
        self._run_wall_seconds: float | None = None
        self._run_cpu_seconds: float | None = None
        self._gpu_sampler: _GpuUtilizationSampler | None = None
        self._profiler: Any = None
        self._workload: dict[str, Any] = {}

    def set_workload(self, **fields: Any) -> None:
        self._workload.update(fields)

    def start_run(self) -> None:
        if self._run_start_wall is not None:
            raise RuntimeError("benchmark recorder is already running")
        self._run_start_wall = time.perf_counter()
        self._run_start_cpu = time.process_time()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
            device_index = self.device.index
            if device_index is None:
                device_index = torch.cuda.current_device()
            self._gpu_sampler = _GpuUtilizationSampler(
                device_index=device_index,
                interval_seconds=self.gpu_sample_interval_seconds,
            )
            self._gpu_sampler.start()
        if self.profile_trace_path is not None:
            self.profile_trace_path.parent.mkdir(parents=True, exist_ok=True)
            activities = [torch.profiler.ProfilerActivity.CPU]
            if self.device.type == "cuda":
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            self._profiler = torch.profiler.profile(
                activities=activities,
                schedule=torch.profiler.schedule(
                    wait=self.warmup_batches,
                    warmup=1,
                    active=self.profile_batches,
                    repeat=1,
                ),
                on_trace_ready=lambda profiler: profiler.export_chrome_trace(
                    str(self.profile_trace_path)
                ),
                record_shapes=False,
                profile_memory=False,
                with_stack=False,
            )
            self._profiler.__enter__()

    def finish_run(self) -> None:
        if self._run_start_wall is None or self._run_start_cpu is None:
            raise RuntimeError("benchmark recorder has not been started")
        if self._profiler is not None:
            self._profiler.__exit__(None, None, None)
            self._profiler = None
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        if self._gpu_sampler is not None:
            self._gpu_sampler.stop()
        self._run_wall_seconds = time.perf_counter() - self._run_start_wall
        self._run_cpu_seconds = time.process_time() - self._run_start_cpu

    def start_epoch(self, epoch: int) -> None:
        self._epoch_starts[epoch] = time.perf_counter()

    def finish_epoch(self, epoch: int) -> None:
        started_at = self._epoch_starts.pop(epoch, None)
        if started_at is not None:
            self._epoch_seconds.append(time.perf_counter() - started_at)

    def batch_received(
        self,
        *,
        epoch: int,
        batch_index: int,
        samples: int,
        data_wait_seconds: float,
        cycle_started_at: float,
    ) -> None:
        sequence = len(self._batches)
        self._current_batch_sequence = sequence
        self._batches.append(
            _BatchMeasurement(
                sequence=sequence,
                epoch=epoch,
                batch_index=batch_index,
                samples=samples,
                data_wait_seconds=data_wait_seconds,
                cycle_started_at=cycle_started_at,
            )
        )

    def finish_batch(self) -> None:
        sequence = self._current_batch_sequence
        if sequence is None:
            return
        self._batches[sequence].cycle_seconds = (
            time.perf_counter() - self._batches[sequence].cycle_started_at
        )
        self._current_batch_sequence = None
        if self._profiler is not None:
            self._profiler.step()

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        cuda_start: Any = None
        cuda_end: Any = None
        if self.device.type == "cuda":
            cuda_start = torch.cuda.Event(enable_timing=True)
            cuda_end = torch.cuda.Event(enable_timing=True)
            cuda_start.record()
        started_at = time.perf_counter()
        try:
            yield
        finally:
            wall_seconds = time.perf_counter() - started_at
            if cuda_end is not None:
                cuda_end.record()
            self._phases.append(
                _PhaseMeasurement(
                    name=name,
                    wall_seconds=wall_seconds,
                    batch_sequence=self._current_batch_sequence,
                    cuda_start=cuda_start,
                    cuda_end=cuda_end,
                )
            )

    def report(self, *, metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
        if self._run_wall_seconds is None or self._run_cpu_seconds is None:
            raise RuntimeError("finish_run() must be called before report()")
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

        steady_batches = [
            batch
            for batch in self._batches
            if batch.sequence >= self.warmup_batches and batch.cycle_seconds is not None
        ]
        steady_sequences = {batch.sequence for batch in steady_batches}
        steady_cycle_seconds = sum(batch.cycle_seconds or 0.0 for batch in steady_batches)
        steady_samples = sum(batch.samples for batch in steady_batches)
        batch_latencies = [batch.cycle_seconds or 0.0 for batch in steady_batches]

        phases = self._phase_summary(steady_sequences)
        cpu_cores = max(1, os.cpu_count() or 1)
        process_cpu_percent_one_core = 100.0 * self._run_cpu_seconds / self._run_wall_seconds
        gpu_samples = self._gpu_sampler.samples if self._gpu_sampler is not None else []
        summary: dict[str, Any] = {
            "run_wall_seconds": self._run_wall_seconds,
            "epoch_wall_seconds": self._epoch_seconds,
            "steady_state_batches": len(steady_batches),
            "steady_state_samples": steady_samples,
            "samples_per_second": (
                steady_samples / steady_cycle_seconds if steady_cycle_seconds > 0 else None
            ),
            "batches_per_second": (
                len(steady_batches) / steady_cycle_seconds if steady_cycle_seconds > 0 else None
            ),
            "median_batch_latency_ms": (
                statistics.median(batch_latencies) * 1000.0 if batch_latencies else None
            ),
            "mean_data_wait_ms": (
                statistics.fmean(batch.data_wait_seconds for batch in steady_batches) * 1000.0
                if steady_batches
                else None
            ),
            "process_cpu_utilization_percent_one_core": process_cpu_percent_one_core,
            "process_cpu_utilization_percent_host": process_cpu_percent_one_core / cpu_cores,
            "gpu_utilization_percent_mean": statistics.fmean(gpu_samples) if gpu_samples else None,
            "gpu_utilization_sample_count": len(gpu_samples),
            "peak_vram_allocated_mib": (
                torch.cuda.max_memory_allocated(self.device) / (1024**2)
                if self.device.type == "cuda"
                else None
            ),
            "peak_vram_reserved_mib": (
                torch.cuda.max_memory_reserved(self.device) / (1024**2)
                if self.device.type == "cuda"
                else None
            ),
            "phases": phases,
        }
        summary["bottleneck"] = _classify_bottleneck(summary)
        return {
            "schema_version": 1,
            "representative_cuda_baseline": self.device.type == "cuda",
            "warmup_batches": self.warmup_batches,
            "workload": dict(self._workload),
            "metadata": dict(metadata or {}),
            "summary": summary,
        }

    def write_report(
        self,
        path: Path,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.report(metadata=metadata), indent=2), encoding="utf-8")
        return path

    def _phase_summary(self, steady_sequences: set[int]) -> dict[str, dict[str, float | int]]:
        grouped: dict[str, list[_PhaseMeasurement]] = {}
        for measurement in self._phases:
            if (
                measurement.batch_sequence is not None
                and measurement.batch_sequence not in steady_sequences
            ):
                continue
            grouped.setdefault(measurement.name, []).append(measurement)

        result: dict[str, dict[str, float | int]] = {}
        for name, measurements in sorted(grouped.items()):
            wall_values = [measurement.wall_seconds for measurement in measurements]
            data: dict[str, float | int] = {
                "count": len(measurements),
                "wall_seconds_total": sum(wall_values),
                "wall_seconds_median": statistics.median(wall_values),
            }
            cuda_values: list[float] = []
            for measurement in measurements:
                if measurement.cuda_start is None or measurement.cuda_end is None:
                    continue
                cuda_values.append(
                    float(measurement.cuda_start.elapsed_time(measurement.cuda_end)) / 1000.0
                )
            if cuda_values:
                data["cuda_seconds_total"] = sum(cuda_values)
                data["cuda_seconds_median"] = statistics.median(cuda_values)
            result[name] = data
        return result


def build_benchmark_metadata(config_path: Path, *, run_config: Mapping[str, Any]) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]
    commit = _git_value(repo_root, "rev-parse", "HEAD")
    branch = _git_value(repo_root, "branch", "--show-current")
    device: dict[str, Any] = {"type": "cuda" if torch.cuda.is_available() else "cpu"}
    if torch.cuda.is_available():
        index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        device.update(
            {
                "index": index,
                "name": properties.name,
                "total_memory_mib": properties.total_memory / (1024**2),
                "capability": list(torch.cuda.get_device_capability(index)),
            }
        )
    return {
        "config_path": str(config_path.resolve()),
        "config_sha256": _sha256_file(config_path),
        "resolved_run_config": dict(run_config),
        "git_branch": branch,
        "git_commit": commit,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "torch_version": torch.__version__,
        "cuda_runtime_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device": device,
        "determinism": {
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
        },
    }


def _classify_bottleneck(summary: Mapping[str, Any]) -> dict[str, Any]:
    if summary.get("gpu_utilization_percent_mean") is None:
        return {
            "classification": "not-classified-without-cuda",
            "reason": "Representative bottleneck classification requires a CUDA run.",
        }
    epochs = summary.get("epoch_wall_seconds") or []
    epoch_total = float(sum(epochs))
    phases = summary.get("phases") or {}
    validation = float((phases.get("validation") or {}).get("wall_seconds_total", 0.0))
    data_wait_ms = summary.get("mean_data_wait_ms")
    median_batch_ms = summary.get("median_batch_latency_ms")
    h2d_median_seconds = (phases.get("h2d") or {}).get("wall_seconds_median", 0.0)
    gpu_util = float(summary["gpu_utilization_percent_mean"])
    validation_share = validation / epoch_total if epoch_total > 0 else 0.0
    input_ms = (
        float(data_wait_ms) + float(h2d_median_seconds) * 1000.0
        if isinstance(data_wait_ms, int | float) and isinstance(h2d_median_seconds, int | float)
        else 0.0
    )
    input_share = (
        input_ms / float(median_batch_ms)
        if isinstance(median_batch_ms, int | float) and median_batch_ms != 0
        else 0.0
    )
    if validation_share >= 0.25:
        classification = "validation-bound"
    elif input_share >= 0.25 and gpu_util < 80.0:
        classification = "input-bound"
    elif gpu_util >= 85.0:
        classification = "compute-bound"
    else:
        classification = "mixed-or-synchronization-bound"
    return {
        "classification": classification,
        "validation_wall_share": validation_share,
        "data_wait_share_of_median_batch": input_share,
        "mean_gpu_utilization_percent": gpu_util,
        "note": (
            "Use the optional torch.profiler trace to distinguish synchronization/kernel-launch "
            "stalls from genuinely mixed workloads."
        ),
    }


def _git_value(repo_root: Path, *args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), *args],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
