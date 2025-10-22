# Qallow Examples and Benchmarks

This directory contains runnable samples, benchmarks, and per-phase CUDA demonstrations.

## Layout

- `benchmarks/` – Repeatable throughput and latency benchmarks.
- `phase_demos/` – Minimal CUDA kernels showcasing Phases 1–13 (`phaseX_demo.cu`).
- `qallow_ethics_integration.c` – Legacy integration demo (still supported).
- `qsvc_synthetic_demo.py` – QSVC classification example powered by Qiskit Aer.
- `vqc_binary_classifier.py` – Variational quantum classifier (VQC) for a binary Iris subset.
- `qgan_torch_gaussian.py` – Torch-integrated qGAN that learns a 2D Gaussian surface.
- `quantum_meta_learning.py` – Hybrid variational learner with parameter-shift optimisation on synthetic data.
- `grover_50q_search.py` – Large-qubit Grover search optimized for Aer MPS simulation.
- `quantum_bandit_policy.py` – VQC-based policy gradient agent for a four-arm quantum bandit.
- `quantum_bandit_runtime.py` – Hardware-ready bandit agent using Qiskit Runtime Sampler.

Build everything via CMake:

```bash
cmake --build build --target qallow_examples
```

Outputs land under `build/`. Benchmark logs are written to `data/logs/benchmarks/`.
