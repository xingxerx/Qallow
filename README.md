# Qallow - Experimental Quantum-Photonic Computing Platform

<div align="center">

**Experimental Quantum-Photonic Computing Platform**

[![Build](https://github.com/xingxerx/Qallow/actions/workflows/internal-ci.yml/badge.svg)](https://github.com/xingxerx/Qallow/actions/workflows/internal-ci.yml)
[![Docs](https://img.shields.io/badge/Docs-Quickstart-blue)](docs/QUICKSTART.md)
[![Phases](https://img.shields.io/badge/Phases-13%20active%20%E2%86%92%2020%20planned-blue)]()
[![Hardware](https://img.shields.io/badge/Hardware-CPU%20%26%20CUDA-green)]()
[![License](https://img.shields.io/badge/License-MIT-blue)]()

</div>

---

## What is Qallow?

**Qallow** is an experimental (v0.1) quantum-photonic computing platform that provides:

- **Quantum Simulation** – photonic propagation and coherence control
- **Quantum Optimization** – QAOA algorithms and quantum decision-making
- **Constraint Satisfaction** – systematic validation and robustness testing
- **Hardware Acceleration** – CPU fallback with CUDA optimization
- **Structured Telemetry** – performance metrics, validation logs, and reproducible analysis

The platform currently ships with **13 active execution phases** (v0.1) orchestrated from a single entry point, with a roadmap to **20 phases** and deterministic output for reproducible results.

---

## Key Features

**Unified Quantum Computing Framework**
- 13 execution phases available today (initialization, optimization, validation) with a roadmap to 20
- Single entry point for complete workflows
- Modular architecture with clean boundaries

**Robust Constraint Validation**
- Systematic constraint checking at every layer
- Resilience testing and robustness metrics
- Comprehensive validation reporting

**Hardware-Optimized Execution**
- CPU fallback for universal compatibility
- CUDA acceleration for high-performance computing
- Profiling hooks for Nsight, nvprof, custom timers

**Structured Output & Metrics**
- Deterministic CSV/JSON logs for reproducible analysis
- Real-time performance metrics
- Comprehensive benchmarking and KPI tracking

**Modular Architecture**
- Clean module boundaries (core, algorithms, backend, interface, tests)
- Documented APIs and integration paths
- Quantum optimization and machine learning capabilities

## 🚀 Quick Start (5 Minutes)

### One-Command Setup

```bash
git clone https://github.com/xingxerx/Qallow.git
cd Qallow
./bootstrap.sh
```

That's it! The bootstrap script automatically:
- Initializes git submodules
- Creates Python virtual environment
- Installs all dependencies
- Downloads optional assets
- Builds C/CUDA binaries
- Runs verification tests

**See [docs/BOOTSTRAP_GUIDE.md](docs/BOOTSTRAP_GUIDE.md) for advanced options.**

### Then Run

```bash
source .venv/bin/activate
./build/qallow run unified
```

---

## Prerequisites

```bash
# Required
cmake ≥ 3.20
gcc ≥ 11 (or clang ≥ 15)
python ≥ 3.10
ninja or make

# Optional (for CUDA acceleration)
cuda toolkit ≥ 12.0
nsight compute (profiling)
```

### Installation & Setup

```bash
# 1. Clone repository
git clone https://github.com/xingxerx/Qallow.git
cd Qallow

# 2. Configure environment
cp .env.example .env   # customize runtime options

# 3. Build everything
./scripts/build_all.sh

# 4. 30-second smoke test (Phase 11 bridge)
./build/qallow run --phase=11 --ticks=32 --states=-1,0,1
```

### Run Your First Simulation

```bash
# Run Phase 13 (Closed-loop ethics accelerator)
./build/qallow --phase=13 --ticks=400 --log=data/logs/phase13.csv

# Run Phase 14 (Deterministic coherence)
./build/qallow phase 14 --ticks=600 --nodes=256 --target_fidelity=0.981

# Run Phase 15 (Convergence & lock-in)
./build/qallow phase 15 --ticks=800 --eps=5e-6
```

### Execute Examples

```bash
# Build all examples
cmake --build build --target qallow_examples

# Run Phase 7 demo
./build/phase07_demo --ticks=100

# Run quantum adaptive demo
python examples/quantum_adaptive_demo.py --episodes 5 --simulate
```

### Examples Index

| Example | Purpose | Command |
|---------|---------|---------|
| Phase 7 harmonic governance | Inspect baseline photonic control loop | `./build/phase07_demo --ticks=100` |
| Phase 11 hardware bridge | Validate coherence bridge with Qiskit | `./build/qallow run --phase=11 --ticks=64 --states=-1,0,1` |
| Throughput benchmark | Profile CPU/CUDA runtime | `cmake --build build --target qallow_throughput_bench && ./build/qallow_throughput_bench` |
| Quantum adaptive demo | Run hybrid adaptive policy search | `python examples/quantum_adaptive_demo.py --episodes 5 --simulate` |
| Unified AGI pipeline | Execute end-to-end integration | `./scripts/run_unified_agi.sh` |

📖 **Need help?** See `docs/QUICKSTART.md` for detailed setup, CUDA installation, and troubleshooting.

## 📈 Deterministic Telemetry Snapshot

Every phase emits deterministic CSV and JSON artifacts so runs can be reproduced end-to-end. A typical Phase 14 export looks like:

```csv
# data/logs/phase14.csv
tick,fidelity,alpha,target
0,0.812340,0.004211,0.981000
50,0.941225,0.004211,0.981000
100,0.977318,0.004211,0.981000
150,0.981043,0.004211,0.981000
```

```json
// data/logs/phase14.jsonl
{"phase":14,"tick":150,"fidelity":0.981043,"alpha_used":0.004211,"status":"OK"}
{"phase":14,"export_path":"data/logs/phase14.json","checksum":"8b77c8e9"}
```

Use `QALLOW_LOG_FORMAT=jsonl` or the default CSV exporters to integrate the telemetry directly into your dashboards.

## 🏗️ System Architecture

### How Qallow Works as One Unit

Qallow is designed as an **integrated quantum-photonic AGI system** with up to 20 research phases (13 active in v0.1) working together:

```
┌─────────────────────────────────────────────────────────────────┐
│                    QALLOW AGI RUNTIME                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  PHASE 1-7: Core Quantum-Photonic Pipeline              │  │
│  │  ├─ Phase 1: Sandboxed bootstrapping                    │  │
│  │  ├─ Phase 2: Baseline telemetry ingestion               │  │
│  │  ├─ Phase 3: Adaptive run-time tuning                   │  │
│  │  ├─ Phase 4: Chronometric prediction                    │  │
│  │  ├─ Phase 5: Poly-Pocket AI (PPAI) routing             │  │
│  │  ├─ Phase 6: Overlay coherence control                  │  │
│  │  └─ Phase 7: Harmonic governance                        │  │
│  └──────────────────────────────────────────────────────────┘  │
│                           ↓                                     │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  PHASE 8-10: Ethics & Learning Loop                     │  │
│  │  ├─ Phase 8: Ethics signal ingestion                    │  │
│  │  ├─ Phase 9: Ethics reasoning                           │  │
│  │  └─ Phase 10: Ethics learning loop                      │  │
│  └──────────────────────────────────────────────────────────┘  │
│                           ↓                                     │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  PHASE 11-13: Quantum Acceleration & Closed-Loop        │  │
│  │  ├─ Phase 11: Quantum-Coherence pipeline                │  │
│  │  ├─ Phase 12: Elasticity simulation                     │  │
│  │  └─ Phase 13: Closed-loop ethics accelerator            │  │
│  └──────────────────────────────────────────────────────────┘  │
│                           ↓                                     │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  PHASE 14-15: Deterministic Coherence & Convergence     │  │
│  │  ├─ Phase 14: Coherence-Lattice Integration             │  │
│  │  │  └─ Deterministic alpha tuning                       │  │
│  │  │  └─ QAOA optimization                                │  │
│  │  └─ Phase 15: Convergence & Lock-in                     │  │
│  │     └─ Stability enforcement                            │  │
│  └──────────────────────────────────────────────────────────┘  │
│                           ↓                                     │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  TELEMETRY & MONITORING                                 │  │
│  │  ├─ Structured CSV/JSON logs                            │  │
│  │  ├─ Real-time performance metrics                       │  │
│  │  ├─ Ethics audit trails                                 │  │
│  │  └─ Operator feedback integration                       │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Module Structure

```
/root/Qallow/
├── core/                    # Core headers & runtime
├── backend/                 # CPU and CUDA backends
│   ├── cpu/                 # CPU implementation
│   └── cuda/                # CUDA acceleration
├── algorithms/              # Ethics, learning, probabilistic
├── interface/               # CLI entry points & launchers
├── src/                     # Runtime support (logging, profiling)
├── include/                 # Public headers (qallow/ namespace)
├── tests/                   # Unit & integration tests (CTest)
├── examples/                # Benchmarks & per-phase demos
├── scripts/                 # Build, CI, monitoring tools
├── data/logs/               # Telemetry & metrics
├── config/                  # Manifests & schemas
├── alg/                     # ALG Quantum Framework
├── quantum_algorithms/      # Unified quantum framework
└── docs/                    # Complete documentation
```

### Data Flow Architecture

```
User Command (CLI)
    ↓
Interface Layer (main.c, launcher.c)
    ├─ Parse arguments
    ├─ Validate configuration
    └─ Route to phase handler
    ↓
Phase Handler (phase_N.c)
    ├─ Load input data
    ├─ Initialize state
    └─ Execute phase logic
    ├─ CPU Path: algorithms/
    └─ CUDA Path: backend/cuda/
    ↓
Telemetry Pipeline
    ├─ Collect metrics
    ├─ Format output (CSV/JSON)
    └─ Write to data/logs/
    ↓
Ethics Layer (algorithms/ethics_*)
    ├─ Evaluate decisions
    ├─ Apply constraints
    └─ Log audit trail
    ↓
Output & Feedback
    ├─ Structured logs
    ├─ Performance metrics
    └─ Operator feedback
```

### Adaptive Quantum Decision Demo

The quantum adaptive loop demonstrates end-to-end integration:

```bash
# Install dependencies
pip install qiskit qiskit-aer

# Simulation-only run
python examples/quantum_adaptive_demo.py --episodes 5 --simulate

# Live run with unified binary
python examples/quantum_adaptive_demo.py --runner ./build/qallow_unified --episodes 3
```

The script:
1. Instantiates `QuantumAdaptiveAgent` (see `python/quantum/adaptive_agent.py`)
2. Feeds telemetry into a two-qubit policy circuit
3. Launches phases 14–16 based on Qiskit measurement outcomes
4. Updates circuit parameters using reward deltas from refreshed telemetry

## 📊 Phase Overview

All 20 research phases with their purposes, inputs, and outputs:

| Phase | Purpose | Inputs | Outputs |
|-------|---------|--------|---------|
| **1** | Sandboxed bootstrapping & self-tests | `sandbox.h` primitives | PASS/FAIL diagnostics |
| **2** | Baseline telemetry ingestion | Hardware metrics, CSV feeds | Normalized telemetry stream |
| **3** | Adaptive run-time tuning | `adaptive_state_t` | Updated scheduler params |
| **4** | Chronometric prediction | Historical event timings | Confidence-adjusted forecasts |
| **5** | Poly-Pocket AI (PPAI) routing | Pocket overlay graphs | Multi-pocket state vector |
| **6** | Overlay coherence control | Decoherence metrics | Stabilized overlay matrix |
| **7** | Harmonic governance | Photonic node graph | Harmonic energy distribution |
| **8** | Ethics signal ingestion | Human + hardware feedback | `ethics_metrics_t` sample |
| **9** | Ethics reasoning | Prior models, telemetry | PASS/FAIL + adjustment hints |
| **10** | Ethics learning loop | Historical verdicts | Updated priors & thresholds |
| **11** | Quantum-Coherence pipeline | CUDA kernels | Per-node coherence layers |
| **12** | Elasticity simulation | Tachyon ticks, `eps` | Equilibrium metrics |
| **13** | Closed-loop ethics accelerator | Phase 12/13 outputs, feedback | Audit log + intervention hooks |

📖 **Detailed specs:** See `docs/ARCHITECTURE_SPEC.md` for complete phase documentation.

## 🔨 Building & Testing

### Build Options

```bash
# Build everything (CPU + CUDA)
./scripts/build_all.sh

# Build CPU only
./scripts/build_wrapper.sh CPU

# Build CUDA only
./scripts/build_wrapper.sh CUDA

# Build with specific generator
cmake -S . -B build -GNinja && cmake --build build
cmake -S . -B build -G"Unix Makefiles" && cmake --build build
```

### Testing

```bash
# Run all unit tests
ctest --test-dir build

# Run smoke tests
tests/smoke/test_modules.sh

# Run phase demos
cmake --build build --target qallow_examples
./build/phase13_demo --ticks=100

# Check dependencies
./scripts/check_dependencies.sh
```

### Validation

```bash
# Validate modules
./scripts/check_dependencies.sh

# Run CI locally
docker compose up --build
```

## ⚛️ Phase 14–15: Deterministic Coherence & Convergence

Phase 14 now guarantees threshold attainment with a closed-form alpha and supports multiple gain sources, all invoked through the unified `qallow` CLI. Phase 15 consumes Phase 14’s output and tightens convergence with non-negative stability.

- Deterministic alpha: α = 1 − ((1 − target) / (1 − f0))^(1/n), applied toward 1.0 so fidelity deterministically crosses the target by tick n.
- Gain sources (highest priority first):
   1. Built-in QAOA tuner: `--tune_qaoa [--qaoa_n N --qaoa_p P]`
   2. External tuner JSON: `--gain_json <file>` containing { "alpha_eff": A }
   3. CUDA J-coupling CSV: `--jcsv <graph.csv>` with `--gain_base` and `--gain_span`
   4. CLI override: `--alpha A`
   5. Closed-form fallback (default)

Examples:

- Minimal deterministic target attainment: `./build/qallow phase 14 --ticks=600 --nodes=256 --target_fidelity=0.981`
- With built-in quantum tuner (keeps everything unified in the CLI): `./build/qallow phase 14 --ticks=600 --target_fidelity=0.981 --tune_qaoa --qaoa_n=16 --qaoa_p=2`
- With CUDA-derived alpha from J-couplings: `./build/qallow phase 14 --ticks=600 --nodes=256 --jcsv=graph.csv --gain_base=0.001 --gain_span=0.009`
- With external tuner JSON: `./build/qallow phase 14 --ticks=600 --gain_json=/path/to/gain.json`
- Export Phase 14 summary: `./build/qallow phase 14 --ticks=600 --target_fidelity=0.981 --export=data/logs/phase14.json`
- Phase 15 convergence and lock-in: `./build/qallow phase 15 --ticks=800 --eps=5e-6`

Notes:
- The Phase 14 loop updates fidelity as f += α(1 − f), and reports [OK] when f ≥ target_fidelity at completion.
- Phase 15 enforces stability ≥ 0 and stops when |score − prev| < eps after a short warm-up.
- Use `qallow help phase` to view all Phase 14/15 flags from the CLI.

## 🚀 Qallow Internal Release v0.1

### Build & Deployment

✅ **Unified builds** – use `make ACCELERATOR=CPU` or `make ACCELERATOR=CUDA` for deterministic outputs under `build/CPU/` and `build/CUDA/`; the CUDA/CPU chooser script `scripts/build_wrapper.sh [CPU|CUDA|AUTO]` now mirrors the same source layout and feature flags.

✅ **Smoke validation** – run `tests/smoke/test_modules.sh` to compile the CPU binary and execute ethics, governance, and phase 12/13 runners with explicit success markers.

✅ **Dependency audit** – execute `scripts/check_dependencies.sh` for version checks on Python ≥ 3.13, CUDA 13.0, Nsight Compute CLI, and the `sentence-transformers/all-MiniLM-L6-v2` model.

✅ **Accelerator CI** – see `.github/workflows/internal-ci.yml` for the CUDA 13.0 container job that builds, runs the smoke tests, and exercises `qallow run --accelerator --file=/tmp/accelerator_input.json`.

✅ **Readiness snapshot** – consolidated module status and metrics live in `docs/internal_readiness_v0_1.md`.

✅ **Dockerized run:** `docker compose up --build`

### Quantum Integration

✅ **Hybrid quantum bridge** – export `QALLOW_QISKIT=1` (and optionally `QALLOW_QISKIT_BACKEND`) to feed Phase 11 topology samples through `scripts/qiskit_bridge.py`, which in turn invokes Qiskit (IBM Runtime or Aer) before reintegrating the coherence metric into the overlay loop.

✅ **Phase 14/15 seeding** – `run_phase14_16.sh` now primes the Rust `qallow_quantum` pipeline; generated metrics land in `data/quantum/phase14_metrics.json` and `data/quantum/phase15_metrics.json`, which are auto-consumed by the C runtime via `QALLOW_PHASE14_METRICS` / `QALLOW_PHASE15_METRICS`.

ℹ️  **Manual refresh** – to reseed without the wrapper script run:

```bash
qallow_quantum pipeline \
    --phase14-ticks=600 --nodes=256 --target-fidelity=0.981 \
    --phase15-ticks=800 --phase15-eps=0.000005 \
    --export-phase14 data/quantum/phase14_metrics.json \
    --export-phase15 data/quantum/phase15_metrics.json
```

The next `qallow_unified run --integrate phase14 phase15` will ingest the refreshed JSON when those environment variables point to the exported files (set automatically by the helper script).

## 🤝 Contributing at a Glance

- **Branching:** start from `main`, push feature work as `feature/<slug>` and rebase before opening a PR.
- **Style:** run `clang-format` for C/C++ (`make format`), `black` + `ruff` for Python helpers, and keep headers ASCII.
- **Tests:** `./scripts/build_all.sh` must complete (build + ctest); add unit coverage for new behaviours.
- **Ethics:** include updates to the ethics fixtures when touching governance-sensitive code.

See `CONTRIBUTING.md` for the full checklist, templates, and review expectations.

## 🧠 Quantum-AI Hyperparameter Optimizer

### Workflow

```bash
# 1. Generate QUBO problem from search space
python algos/qaoa_hparam.py --space configs/hparam_space.yaml --out /tmp/qubo.json

# 2. Feed into Phase 11 with QAOA
./build/qallow_unified --phase=11 --algo=qaoa --qubo=/tmp/qubo.json --shots=4096 --p=2 --ticks=300 > /tmp/qaoa_out.json

# 3. Rank bitstrings and train
python scripts/hparam_eval.py --in /tmp/qaoa_out.json --topk 5 --epochs 3
```

### Optional C Acceleration

```bash
# Compile C extension for native score calculations
gcc -shared -O2 -fPIC c_ext/qaoa_eval.c -o build/libqaoa_eval.so

# Use in training
python scripts/train_small_model.py
```

### Hybrid Execution

```bash
# Run with CUDA + Qiskit
./scripts/build_wrapper.sh CUDA
./scripts/run_auto.sh --cuda --with-qiskit

# One-shot rebuild + run
./scripts/run_latest.sh --cuda --with-qiskit
```

**Bridge Options:**
- `--qiskit-backend` – specify Qiskit backend
- `--qiskit-bridge` – custom bridge configuration

## Quantum ML Integration

Qallow's quantum_ml module provides tools for hybrid quantum-classical machine learning:

- **QuantumNASExplorer** (in `quantum_ml/sampling_nas.py`):
  - Uses Phase 11 quantum states to generate diverse training data for neural architecture search.
  - Example usage:
    ```python
    from quantum_ml import QuantumNASExplorer
    explorer = QuantumNASExplorer()
    architectures = explorer.generate_architectures(10)
    print(architectures)
    ```
- **Hybrid classical-quantum layers:**
  - Implement variational quantum circuits as neural network layers, using Qallow's quantum backend for feature extraction.
- **Quantum attention mechanisms:**
  - Replace transformer attention with quantum amplitude encoding, leveraging Phase 14 coherence control for long-range dependencies.

See `quantum_ml/sampling_nas.py` for code and integration details.

## 📊 Telemetry & Logging

**Structured Output:**
- CSV logs: `data/logs/telemetry.csv`
- JSONL logs: `data/logs/telemetry.jsonl`

**Logging API:**
- `include/qallow/logging.h` exposes `qallow_log_*` helpers backed by `spdlog`
- Profiling macros (`QALLOW_PROFILE_SCOPE`) pipe into Nsight ranges when CUDA is enabled

---

## 🔗 Unified Pipeline Shortcut

To exercise the quantum workloads and the unified runtime in one go:

```bash
# Install dependencies
pip install qiskit-aer qiskit-machine-learning scikit-learn

# Run unified pipeline
./scripts/run_unified_agi.sh
```

See `docs/unified_agi_pipeline.md` for detailed documentation.

---

## 🗺️ Next Milestones

- **v0.2** – extend runtime coverage to Phases 16–18 and stabilize the CUDA bridge.
- **v0.3** – integrate hardware backends for Phase 19 audit trail and Phase 20 synthesis.
- **v1.0** – graduate unified pipeline to production readiness with signed telemetry schemas.

---

## 📚 Documentation

- **Architecture:** `docs/ARCHITECTURE_SPEC.md`
- **Quick Start:** `docs/QUICKSTART.md`
- **Ethics Charter:** `docs/ETHICS_CHARTER.md`
- **Unified Pipeline:** `docs/unified_agi_pipeline.md`
- **Contributing:** `CONTRIBUTING.md`

---

## 📄 License & Governance

This repository is available under the **MIT license** (`LICENSE`).
In practice that means every PR must pass the ethics validation tests and comply with the charter before merge.

**Contributions must respect:**
- The ethics charter (`docs/ETHICS_CHARTER.md`)
- The sustainability + compassion + harmony mandate
- Coding standards in `CONTRIBUTING.md`

---

## 💬 Getting Help

- **Documentation:** `docs/` directory
- **Issues & Roadmap:** GitHub Issues / Projects
- **Discussion:** Open a thread tagged `support` or `design`
- **Contributing:** See `CONTRIBUTING.md`

---

## ✅ Status

| Aspect | Status |
|--------|--------|
| **Version** | v0.1 |
| **Phases** | 13 Active (20 Planned) |
| **Hardware** | CPU & CUDA |
| **Ethics** | Integrated |
| **Telemetry** | Full Coverage |
| **Production Ready** | No (v0.1) |

---

**Made with ❤️ for Autonomous Intelligence**
