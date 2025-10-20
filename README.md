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

## IBM Quantum Integration

To connect Phase 11 coherence routines to IBM Quantum hardware or simulators, follow the step-by-step instructions in `docs/IBM_QUANTUM_PLATFORM_SETUP.md`. The guide covers account creation, API token management, the Bell-state smoke test in `examples/ibm_quantum_bell.py`, and the bridge module exposed at `python/quantum/qallow_ibm_bridge.py`.

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

## Qallow Internal Release v0.1

- **Unified builds** – use `make ACCELERATOR=CPU` or `make ACCELERATOR=CUDA` for deterministic outputs under `build/CPU/` and `build/CUDA/`; the CUDA/CPU chooser script `scripts/build_wrapper.sh [CPU|CUDA|AUTO]` now mirrors the same source layout and feature flags.
- **Smoke validation** – run `tests/smoke/test_modules.sh` to compile the CPU binary and execute ethics, governance, and phase 12/13 runners with explicit success markers.
- **Dependency audit** – execute `scripts/check_dependencies.sh` for version checks on Python ≥ 3.13, CUDA 13.0, Nsight Compute CLI, and the `sentence-transformers/all-MiniLM-L6-v2` model.
- **Accelerator CI** – see `.github/workflows/internal-ci.yml` for the CUDA 13.0 container job that builds, runs the smoke tests, and exercises `qallow run --accelerator --file=/tmp/accelerator_input.json`.
- **Readiness snapshot** – consolidated module status and metrics live in `docs/internal_readiness_v0_1.md`.
- **Dockerized run:** `docker compose up --build`
- **Hybrid quantum bridge** – export `QALLOW_QISKIT=1` (and optionally `QALLOW_QISKIT_BACKEND`) to feed Phase 11 topology samples through `scripts/qiskit_bridge.py`, which in turn invokes Qiskit (IBM Runtime or Aer) before reintegrating the coherence metric into the overlay loop.

See `CONTRIBUTING.md` for coding standards, branching model, and CI expectations.

## Quantum-AI Hyperparameter Optimizer

- Generate a QUBO problem from the discrete search space in `configs/hparam_space.yaml` via `python algos/qaoa_hparam.py --space configs/hparam_space.yaml --out /tmp/qubo.json`.
- Feed the JSON into Phase 11 with `./build/qallow_unified --phase=11 --algo=qaoa --qubo=/tmp/qubo.json --shots=4096 --p=2 --ticks=300 > /tmp/qaoa_out.json`.
- Rank the resulting bitstrings and launch the lightweight trainer using `python scripts/hparam_eval.py --in /tmp/qaoa_out.json --topk 5 --epochs 3`.
- Optional C assist: compile `c_ext/qaoa_eval.c` as `gcc -shared -O2 -fPIC c_ext/qaoa_eval.c -o build/libqaoa_eval.so` to let `scripts/train_small_model.py` call into native code for score calculations.

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
