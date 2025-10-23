# Qallow Capabilities & Glossary

## Runtime Highlights
- Unified quantum-photonic AGI runtime that strings together 13+ research phases behind a single CLI (`qallow`, `qallow_unified`).
- Dual hardware backends: portable CPU implementation plus CUDA acceleration, with profiling hooks for Nsight and custom timers.
- Ethics-first orchestration enforcing the sustainability + compassion + harmony axiom (E = S + C + H) across ingestion, reasoning, and feedback loops.
- Closed-loop telemetry with deterministic CSV/JSON outputs, live dashboards, and audit trails under `data/logs/` and `data/quantum/`.
- Extensible module boundaries (`core/`, `backend/`, `algorithms/`, `interface/`, `python/`) to support research experimentation, plug-in analyzers, and future phases.

## Phase Capabilities
| Phase | Focus Area | Representative Inputs | Primary Outputs/Effects |
|-------|------------|------------------------|--------------------------|
| 1 | Sandboxed bootstrapping & confidence checks | `sandbox.h` primitives, baseline configs | PASS/FAIL diagnostics, safe startup envelope |
| 2 | Telemetry ingestion & normalization | Hardware counters, CSV feeds, environment state | Canonical telemetry stream, health markers |
| 3 | Adaptive runtime tuning | `adaptive_state_t`, phase feedback | Updated scheduler/priority weights |
| 4 | Chronometric prediction | Historical tick timing, latency traces | Confidence-adjusted forecast vectors |
| 5 | Poly-Pocket AI routing (PPAI) | Overlay graphs, pocket manifests | Multi-pocket state routing table |
| 6 | Overlay coherence control | Decoherence metrics, node harmonics | Stabilized overlay matrix, coherence deltas |
| 7 | Harmonic governance | Photonic node topology, harmonic budgets | Harmonic energy distribution, guardrails |
| 8 | Ethics signal ingestion | Human/operator feedback, hardware telemetry | `ethics_metrics_t` sample, ingestion audit |
| 9 | Ethics reasoning core | Prior models, phase telemetry | PASS/FAIL verdict, intervention hints |
| 10 | Ethics learning loop | Historical verdicts, thresholds | Updated priors, recalibrated thresholds |
| 11 | Quantum coherence pipeline | CUDA kernels, QUBO payloads | Per-node coherence layers, bridge payloads |
| 12 | Elasticity simulation | Tachyon ticks, epsilon tolerances | Elasticity equilibrium metrics |
| 13 | Closed-loop ethics accelerator | Phase 12 outputs, live feedback | Ethics audit log, intervention hooks |
| 14 | Deterministic coherence lattice | Target fidelity, gain sources (CLI/JSON/CUDA) | Deterministic alpha schedule, fidelity trace |
| 15 | Convergence & lock-in | Phase 14 metrics, stability bounds | Stable convergence report, non-negative lock-in |
| 16 | Meta introspection (experimental) | Phase 15 state, meta triggers | Introspection metrics (Rust + CUDA bridge) |

## Operational Modes
- **CLI execution**: `./build/qallow` for focused phases, `./build/qallow_unified` for orchestrating multiple phases and continuous runs; `interface/main.c` coordinates parsing and routing.
- **Scripts**: `scripts/build_all.sh` (auto CPU/CUDA builds + ctest), `run_qallow_unified.sh` and `run_phase14_16.sh` for guided demos, `scripts/check_dependencies.sh` for environment validation.
- **Examples & demos**: Phase demos under `build/phase##_demo`, quantum adaptive loop via `examples/quantum_adaptive_demo.py`, and SDL visualizer (`interface/qallow_ui.c`) when SDL2 is available.
- **Python bridge**: `python/quantum/run_phase11_bridge.py` integrates Qiskit (`QALLOW_QISKIT` env) for hybrid quantum runs; additional utilities in `python/` and `alg/` for hyperparameter search.
- **Testing**: CTest-based suites (`unit_ethics_core`, `unit_dl_integration`, `unit_cuda_parallel`), smoke harness `tests/smoke/test_modules.sh`, integration tests under `tests/integration/`.

## Module Responsibilities
- `core/` – shared headers, runtime primitives, and orchestration contracts consumed by CPU/CUDA backends.
- `backend/cpu/` – C11 implementations of all phases; mirrors CUDA kernels for parity.
- `backend/cuda/` – C++17/CUDA kernels (e.g., `phase11.cu`, `koopman_cuda.cu`) for accelerated coherence and routing.
- `algorithms/` – ethics engines (`ethics_core.c`, `ethics_bayes.c`, `ethics_feed.c`), learning loops, and governance logic.
- `include/qallow/` – public API headers (`logging.h`, `profiling.h`, `phases.h`) exposing runtime contracts and macros.
- `interface/` – entry points (`main.c`), CLI driver, optional SDL UI glue, and argument parsing.
- `python/`, `alg/`, `quantum_algorithms/` – Python tooling for QAOA, SPSA, QUBO generation, and adaptive demos.
- `scripts/` – build, deployment, CI helpers, dependency auditors, telemetry collectors.
- `tests/` – unit/integration layouts, smoke harness, CSV baselines in `data/logs/` for deterministic verification.

## Toolchain & Workflows
- **Builds**: CMake (`cmake -S . -B build`) with optional `-DQALLOW_ENABLE_CUDA=ON`; Makefile shim `make ACCELERATOR=CPU|CUDA`; Docker compose workflow for containerized CI.
- **Profiling**: `QALLOW_PROFILE_SCOPE` macros use `runtime/profiling.cpp` to emit scoped metrics; Nsight Compute recommended for CUDA loops.
- **Telemetry**: Logs under `data/logs/` (`log_phase12.csv`, `log_phase13.csv`), phase summaries (e.g., `phase14_metrics.json`), and streaming CSV (`qallow_stream.csv`).
- **Deployment hooks**: `deploy/` and `ops/` provide manifests, while `config/thresholds.json` and `configs/hparam_space.yaml` capture tunable parameters.
- **Validation**: `verify_implementation.sh`, `tests/smoke/test_modules.sh`, and `scripts/run_unified_agi.sh` execute end-to-end checks prior to release snapshots.

## Integration Points
- **Ethics pipeline**: Phases 8–13 enforce ethics metrics, with hooks for operator feedback and audit trails.
- **Quantum acceleration**: Phase 11 drives QAOA through CUDA kernels or the Python/Qiskit bridge; Rust helper `qallow_quantum_rust/` exports phase metrics consumed later.
- **Telemetry dashboards**: CSV metrics feed monitoring under `monitoring/` and documentation in `docs/` for post-run analysis.
- **External dependencies**: FetchContent-managed `spdlog`, optional SDL2/SDL2_ttf, CUDA Toolkit ≥ 12, Python packages `qiskit`, `sentence-transformers` for advanced features.

## Glossary (A–Z)
- **Adaptive Chronometrics** – Phase 4’s time-series forecasting step that stabilizes tick pacing against historical latency.
- **Alpha Schedule** – Deterministic gain function in Phase 14 ensuring fidelity surpasses the target within the requested tick budget.
- **Audit Trail** – Ethics pipeline CSV/JSON entries documenting ingestion, reasoning, and interventions for review.
- **Closed-Loop Ethics Accelerator** – Phase 13 feedback system that monitors ethics metrics in real time and injects interventions when E = S + C + H drifts.
- **Convergence Lock-In** – Phase 15’s guarantee that stability remains non-negative after convergence criteria (`eps`) are satisfied.
- **CUDA Bridge** – Collection of kernels under `backend/cuda/` that mirror CPU logic while exposing accelerated coherence and routing operations.
- **Deterministic Telemetry** – Structured log outputs guaranteeing reproducible metrics across runs (`data/logs/`, `data/quantum/`).
- **Ethics Metrics (`ethics_metrics_t`)** – Structured payload consumed by phases 8–10 representing sustainability, compassion, and harmony signals.
- **Harmonic Governance** – Phase 7’s control system that redistributes energy across photonic nodes to maintain harmonic balance.
- **Hyperparameter Optimizer** – Python tooling in `algos/qaoa_hparam.py` that converts search spaces into QUBO problems for Phase 11 QAOA sweeps.
- **Multi-Pocket Routing (PPAI)** – Phase 5 router that assigns workloads across multiple agent pockets using overlay graphs.
- **Nsight Profiling Hook** – `QALLOW_PROFILE_SCOPE` instrumentation enabling GPU/CPU profiling capture for performance regression analysis.
- **Quantum Bridge** – Phase 11 hardware/software interface that leverages Qiskit (`python/quantum/run_phase11_bridge.py`) to feed coherence data back into the C runtime.
- **QUBO Payload** – Quadratic unconstrained binary optimization instance supplied to the quantum pipeline for QAOA-based decision making.
- **SDL Visualizer** – Optional interface (`interface/qallow_ui.c`) providing runtime visualization when SDL2 dependencies are present.
- **Telemetry Stream** – Continuous CSV feed (`qallow_stream.csv`) reflecting live phase state for dashboards and anomaly detection.
- **Triton Elasticity** – Term used in Phase 12 to describe the simulated elasticity dynamics governing equilibrium metrics.
- **Unified Runner** – `qallow_unified` binary stitching multiple phases into a cohesive execution plan with consistent logging and ethics checks.
- **Validator Harness** – Smoke + unit test suite (`tests/smoke/test_modules.sh`, `ctest`) ensuring core phases and ethics modules remain stable across changes.
