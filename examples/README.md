# Qallow Examples and Benchmarks

This directory contains runnable samples, benchmarks, and per-phase CUDA demonstrations.

## Layout

- `benchmarks/` – Repeatable throughput and latency benchmarks.
- `phase_demos/` – Minimal CUDA kernels showcasing Phases 1–13 (`phaseX_demo.cu`).
- `qallow_ethics_integration.c` – Legacy integration demo (still supported).

Build everything via CMake:

```bash
cmake --build build --target qallow_examples
```

Outputs land under `build/`. Benchmark logs are written to `data/logs/benchmarks/`.
