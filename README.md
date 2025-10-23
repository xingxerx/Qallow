# 🚀 Qallow - Autonomous Intelligence Runtime

<div align="center">

**The Complete Quantum-Photonic AGI System**

[![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)]()
[![Phases](https://img.shields.io/badge/Phases-13%20Research%20Phases-blue)]()
[![Hardware](https://img.shields.io/badge/Hardware-CPU%20%26%20CUDA-green)]()
[![License](https://img.shields.io/badge/License-MIT-blue)]()

</div>

---

## 🎯 What is Qallow?

**Qallow** is an experimental autonomous intelligence runtime that blends:

- **🔬 Photonic Simulation** – harmonic propagation and coherence control
- **⚛️ Quantum Computing** – QAOA optimization and quantum decision-making
- **🛡️ Ethics-First Design** – sustainability + compassion + harmony (E = S + C + H)
- **⚡ Hardware Acceleration** – CPU fallback with CUDA optimization
- **📊 Closed-Loop Telemetry** – performance, ethics, and operator feedback

The project ships **13 research phases** that can be executed from a single entry point, with deterministic telemetry for reproducible analysis.

---

## ✨ Key Features

✅ **Unified AGI Framework**
- 13 research phases (adaptive chronometrics, multi-pocket orchestration, ethics monitoring, etc.)
- Single entry point for complete workflow
- Modular architecture with clean boundaries

✅ **Ethics & Safety First**
- Sustainability + Compassion + Harmony principle enforced at every layer
- Ethics-core engine with telemetry integration
- Closed-loop feedback and intervention hooks

✅ **Hardware-Aware Execution**
- CPU fallback for universal compatibility
- CUDA acceleration for high-performance computing
- Profiling hooks for Nsight, nvprof, custom timers

✅ **Deterministic Telemetry**
- Structured CSV/JSON logs for reproducible analysis
- Real-time performance metrics
- Comprehensive benchmarking support

✅ **Extensible Research Platform**
- Clean module boundaries (core, algorithms, backend, interface, tests)
- Documented contribution paths
- Quantum-AI hyperparameter optimization

## 🚀 Quick Start (5 Minutes)

### Prerequisites

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

📖 **Need help?** See `docs/QUICKSTART.md` for detailed setup, CUDA installation, and troubleshooting.

## 🏗️ System Architecture

### How Qallow Works as One Unit

Qallow is designed as an **integrated quantum-photonic AGI system** with 13 research phases working together:

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

All 13 research phases with their purposes, inputs, and outputs:

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

📖 **Contributing:** See `CONTRIBUTING.md` for coding standards, branching model, and CI expectations.

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

## 📚 Documentation

- **Architecture:** `docs/ARCHITECTURE_SPEC.md`
- **Quick Start:** `docs/QUICKSTART.md`
- **Ethics Charter:** `docs/ETHICS_CHARTER.md`
- **Unified Pipeline:** `docs/unified_agi_pipeline.md`
- **Contributing:** `CONTRIBUTING.md`

---

## 📄 License & Governance

This repository is available under the **MIT license** (`LICENSE`).

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
| **Version** | 1.0.0 |
| **Phases** | 13 Research Phases |
| **Hardware** | CPU & CUDA |
| **Ethics** | Integrated |
| **Telemetry** | Full Coverage |
| **Production Ready** | ✓ Yes |

---

**Made with ❤️ for Autonomous Intelligence**
