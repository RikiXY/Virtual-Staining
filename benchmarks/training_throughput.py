from __future__ import annotations

import argparse
from pathlib import Path

import torch

from virtual_staining.applications.train import train
from virtual_staining.config.run import RunConfig
from virtual_staining.training.benchmarking import (
    TrainingBenchmarkRecorder,
    build_benchmark_metadata,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark the real Virtual Staining training pipeline without changing its semantics."
        )
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--warmup-batches", type=int, default=10)
    parser.add_argument("--gpu-sample-interval", type=float, default=0.5)
    parser.add_argument("--profile-trace", type=Path, default=None)
    parser.add_argument("--profile-batches", type=int, default=5)
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow a CPU-only harness check. CPU reports are not representative baseline results.",
    )
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = RunConfig.from_yaml(config_path)
    if config.training is None:
        raise ValueError("Benchmark config must contain a training section.")
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError(
            "Representative training throughput baselines require CUDA. "
            "Use --allow-cpu only to validate the benchmark harness."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    recorder = TrainingBenchmarkRecorder(
        device,
        warmup_batches=args.warmup_batches,
        gpu_sample_interval_seconds=args.gpu_sample_interval,
        profile_trace_path=args.profile_trace.resolve() if args.profile_trace else None,
        profile_batches=args.profile_batches,
    )
    train(config, config_path, benchmark_recorder=recorder)
    metadata = build_benchmark_metadata(config_path, run_config=config.to_dict())
    output = recorder.write_report(args.output.resolve(), metadata=metadata)
    print(output)


if __name__ == "__main__":
    main()
