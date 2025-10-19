# Qallow

Qallow is an experimental autonomous intelligence runtime that blends photonic simulation, quantum harmonic propagation, and an ethics-first supervisory layer. The project ships thirteen research phases that can be executed on CPU or CUDA devices, with a closed-loop telemetry pipeline that records performance, ethics outcomes, and operator feedback. This repository now provides a cohesive onboarding experience, modular build system, and standardized documentation for future contributors.

## Project Goals

- **Unified AGI phases** – run all thirteen Qallow phases (adaptive chronometrics, multi-pocket orchestration, ethics monitoring, etc.) from a single entry point.
- **Ethics and safety first** – enforce the sustainability + compassion + harmony (E = S + C + H) principle at every layer using the `ethics_core` engine and telemetry.
- **Hardware-aware execution** – support CPU fallback and CUDA acceleration, with profiling hooks for Nsight, nvprof, or custom timers.
- **Deterministic telemetry** – emit structured CSV/JSON logs under `data/logs/` for reproducible analysis and benchmarking.
- **Extensible research platform** – offer clean module boundaries (`core`, `algorithms`, `backend`, `interface`, `tests`) and documented contribution paths.

## Quickstart

1. **Install prerequisites**
   - CMake ≥ 3.20, Ninja or Make
   - GCC ≥ 11, optional Clang ≥ 15
   - CUDA Toolkit ≥ 12.0 (optional, auto-detected)
   - Python ≥ 3.10 with `pip`
   - Nsight Compute (optional profiling)
2. **Clone & configure**
   ```bash
   git clone https://github.com/xingxerx/Qallow.git
   cd Qallow
   cp .env.example .env   # customize runtime options
   ./scripts/build_all.sh
   ```
3. **Run the VM**
   ```bash
   ./build/qallow --phase=13 --ticks=400 --log=data/logs/phase13.csv
   ```
4. **Execute examples**
   ```bash
   cmake --build build --target qallow_examples
   ./build/phase07_demo --ticks=100
   ```

See `docs/QUICKSTART.md` for extended dependency notes, CUDA driver installation steps, and troubleshooting guidance.

## Repository Layout

```
core/            # Core headers (exported through include/)
backend/         # CPU and CUDA backends
algorithms/      # Ethics, learning, and probabilistic modules
interface/       # Launchers and CLI entry points
src/             # Runtime support (logging, profiling)
include/         # Public headers (`qallow/` namespace)
tests/           # Unit and integration tests (CTest)
examples/        # Benchmarks and per-phase CUDA demos
scripts/         # Tooling, builds, continuous monitors
data/logs/       # Metrics emitted by telemetry
config/          # Version manifest and runtime schemas
```

## Phase Overview

A condensed summary lives in `docs/ARCHITECTURE_SPEC.md`. Each phase has:

| Phase | Purpose | Inputs | Outputs |
| --- | --- | --- | --- |
| 1 | Sandboxed bootstrapping and self-tests | `sandbox.h` primitives | PASS/FAIL diagnostics |
| 2 | Baseline telemetry ingestion | Hardware metrics, CSV feeds | Normalized telemetry stream |
| 3 | Adaptive run-time tuning | `adaptive_state_t` | Updated scheduler params |
| 4 | Chronometric prediction | Historical event timings | Confidence-adjusted forecasts |
| 5 | Poly-Pocket AI (PPAI) routing | Pocket overlay graphs | Multi-pocket state vector |
| 6 | Overlay coherence control | Decoherence metrics | Stabilized overlay matrix |
| 7 | Harmonic governance | Photonic node graph | Harmonic energy distribution |
| 8 | Ethics signal ingestion | Human + hardware feedback | `ethics_metrics_t` sample |
| 9 | Ethics reasoning | Prior models, telemetry | PASS/FAIL + adjustment hints |
| 10 | Ethics learning loop | Historical verdicts | Updated priors & thresholds |
| 11 | Quantum-Coherence pipeline | CUDA kernels | Per-node coherence layers |
| 12 | Elasticity simulation | Tachyon ticks, `eps` | Equilibrium metrics |
| 13 | Closed-loop ethics accelerator | Phase 12/13 outputs, human feedback | Audit log + intervention hooks |

## Building & Testing

- **Build everything:** `./scripts/build_all.sh`
- **Run unit tests:** `ctest --test-dir build`
- **Run phase demos:** `cmake --build build --target qallow_examples && build/phase13_demo`
- **Dockerized run:** `docker compose up --build`

See `CONTRIBUTING.md` for coding standards, branching model, and CI expectations.

## Telemetry & Logging

- Structured logs are emitted to `data/logs/telemetry.csv` and `data/logs/telemetry.jsonl`.
- `include/qallow/logging.h` exposes `qallow_log_*` helpers backed by `spdlog`.
- Profiling macros (`QALLOW_PROFILE_SCOPE`) pipe into Nsight ranges when CUDA is enabled.

## License & Governance

This repo is available under the MIT license (`LICENSE`). Contributions must respect the ethics charter (`docs/ETHICS_CHARTER.md`) and the sustainability + compassion + harmony mandate.

## Getting Help

- Documentation: `docs/`
- Issues & roadmap: GitHub Issues / Projects
- Discussion: open a thread tagged `support` or `design`

If you plan major changes, propose them in an RFC under `docs/rfcs/` before starting implementation.
