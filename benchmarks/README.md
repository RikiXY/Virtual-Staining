# Training Throughput Benchmark

This benchmark is the shared end-to-end performance gate for training optimizations. It runs the real training pipeline and records steady-state batch throughput, epoch wall time, data wait, host-to-device transfer, discriminator/generator update time, validation, preview I/O, checkpoint I/O, CPU use, GPU utilization, and peak VRAM.

Use a prepared representative dataset and a frozen run config. Run from the repository root inside the Nix shell. A representative baseline requires CUDA; `--allow-cpu` exists only to verify the harness.

```bash
python benchmarks/training_throughput.py \
  --config config/runs/local/baseline.yaml \
  --output local_workspace/results/benchmarks/baseline-r1.json \
  --warmup-batches 10
```

Run at least three independent repetitions after warmup with unique `run_name` values so checkpoints/history from one repetition cannot affect another. Keep dataset, image size, batch size, worker count, seed, validation/checkpoint cadence, filesystem, GPU power/clock policy, and background load fixed. Report median end-to-end epoch time and throughput across repetitions rather than the best run.

If augmentation is publication-relevant, repeat the same matrix with the representative augmentation policy enabled. Treat augmentation-disabled and augmentation-enabled results as separate workloads rather than averaging them together.

For a diagnostic trace, add `--profile-trace <path> --profile-batches 5`. Profiling is opt-in because it perturbs timing. Use the normal unprofiled run for throughput numbers; use the trace only to distinguish synchronization/kernel-launch stalls from input or compute bottlenecks.

The `discriminator_update` phase intentionally includes the current fake-generation work performed before the discriminator backward pass. This preserves the existing baseline semantics and makes the duplicate-generator/autograd optimization measurable rather than hiding it.

The JSON report records the exact resolved run config and its SHA-256, git branch/commit, Python/PyTorch/CUDA/cuDNN versions, deterministic backend flags, device identity, resource metrics, and phase timings. CUDA events measure GPU time without adding per-phase synchronization; a single synchronization occurs when the report is finalized.

Performance changes should be compared with this same procedure. Retain an optimization only when the repeated end-to-end epoch result materially improves, even if an isolated phase or microbenchmark looks faster.
