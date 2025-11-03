# Qallow Unified CLI - Complete Reference

## Overview

Qallow has been unified into a **single consolidated command structure** with four main command groups: `run`, `system`, `phase`, and `mind`. All operations can be accessed through these top-level groups, eliminating the need for separate phase commands.

**Design Principle:** `qallow <group> [subcommand] [options]`

---

## Quick Start

```bash
# Build the project
qallow system build

# Run the unified VM
qallow run

# Run with automatic Lightning Agent improvement loop
bash run_agent_lightning_loop.sh

# Run a specific phase
qallow phase 13 --nodes=16 --ticks=500

# Run phases 12-15 in sequence (integrated pipeline)
qallow run unified --integrate-phase13-ticks=400 --integrate-phase13-k=0.002
```

---

## Command Groups

### 1. `qallow run` - Workflow Execution

**Purpose:** Execute Qallow workflows in various profiles (standard, benchmark, live ingestion).

#### Subcommands

| Subcommand | Purpose | Alias |
|-----------|---------|-------|
| `vm` | Execute the unified VM (default) | N/A |
| `bench` | Run with benchmark profiling | `qallow run bench` |
| `live` | Run with live data ingestion streams | `qallow run live` |
| `unified` | Run phases 12-15 sequentially | `qallow run pipeline` |
| `accelerator` | Launch Phase 13 accelerator directly | N/A |
| `entangle` | Generate GHZ/W entanglement via QuTiP | N/A |

#### Common Options

```bash
# Dashboard control
qallow run vm --dashboard=50        # Update dashboard every 50 ticks
qallow run vm --dashboard=off       # Disable dashboard output

# Integrated phase execution
qallow run vm --integrate                    # Phases 12-15 (default)
qallow run vm --integrate phase11 phase12    # Select specific phases
qallow run vm --integrate unified            # Alias for phases 12-15

# Phase-specific overrides
qallow run vm --integrate --integrate-phase13-ticks=200
qallow run vm --integrate --integrate-phase13-k=0.003
qallow run vm --integrate --integrate-phase12-eps=0.00001

# Advanced options
qallow run vm --hardware                    # Route Phase 11 to IBM Quantum
qallow run vm --self-audit                  # Enable meta-introspection logging
qallow run vm --self-audit-path /tmp/audit  # Custom audit directory
qallow run vm --dl-model model.pt          # Load TorchScript model
qallow run vm --dl-device=gpu              # Prefer GPU for ML inference
```

#### Examples

```bash
# Standard execution
qallow run vm

# Benchmark profiling
qallow run bench

# Live ingestion with dashboard every 100 ticks
qallow run live --dashboard=100

# Integrated pipeline (phases 12-15)
qallow run unified

# Custom phase execution with overrides
qallow run vm --integrate phase12 phase13 --integrate-phase13-ticks=300

# Phase 11 (Quantum Bridge) on hardware
qallow run vm --hardware --integrate phase11
```

---

### 2. `qallow system` - Build & Project Management

**Purpose:** Build, clean, and verify the Qallow project.

#### Subcommands

| Subcommand | Purpose | Example |
|-----------|---------|---------|
| `build` | Compile CPU + CUDA backends | `qallow system build` |
| `clear` | Clean build artifacts and logs | `qallow system clear` |
| `verify` | Run system health checks | `qallow system verify` |

#### Examples

```bash
# Build with auto-detection (CPU + CUDA if available)
qallow system build

# Force CUDA build
./scripts/build_all.sh --cuda

# Force CPU-only build
./scripts/build_all.sh --cpu

# Clean everything and rebuild
qallow system clear && qallow system build

# Verify system health
qallow system verify
```

---

### 3. `qallow phase` - Individual Phase Runners

**Purpose:** Execute specific quantum/simulation phases directly.

#### Supported Phases

| Phase | Name | Command |
|-------|------|---------|
| 11 | Coherence Bridge (Quantum) | `qallow phase 11` |
| 12 | Elasticity Simulation | `qallow phase 12` |
| 13 | Harmonic Propagation | `qallow phase 13` |
| 14 | Coherence-Lattice Integration | `qallow phase 14` |
| 15 | Convergence & Lock-in | `qallow phase 15` |
| 16-20 | Advanced Phases | `qallow phase 16` ... `qallow phase 20` |

#### Phase 11 (Coherence Bridge)

```bash
qallow phase 11 --ticks=400 --states=-1,0,1
qallow phase 11 --ticks=400 --states=-1,0,1 --hardware-only  # IBM Quantum
```

**Options:**
- `--ticks=N` - Number of shots (default: 400)
- `--states=S` - Quantum states to prepare (default: -1,0,1)
- `--hardware-only` - Use real quantum hardware (requires QALLOW_QISKIT=1)

#### Phase 12 (Elasticity)

```bash
qallow phase 12 --ticks=100 --eps=0.0001
qallow phase 12 --ticks=200 --audit-tag=demo --log=/tmp/phase12.csv
```

**Options:**
- `--ticks=N` - Number of ticks (default: 1000)
- `--eps=E` - Elasticity epsilon (default: 0.0001)
- `--log=FILE` - Override log path (default: data/logs/phase12.csv)
- `--audit-tag=TAG` - Stamp audit tag into telemetry

#### Phase 13 (Harmonic Propagation)

```bash
qallow phase 13 --nodes=16 --ticks=500 --k=0.002
qallow phase 13 --nodes=32 --ticks=400 --audit-tag=bench
```

**Options:**
- `--nodes=N` - Number of harmonic pockets (default: 8)
- `--ticks=N` - Number of ticks (default: 400)
- `--k=F` - Coupling constant (default: 0.001)
- `--log=FILE` - Override log path
- `--audit-tag=TAG` - Stamp audit tag

#### Phase 14 (Coherence-Lattice Integration)

```bash
qallow phase 14 --ticks=600 --nodes=256 --target_fidelity=0.981
qallow phase 14 --tune_qaoa --qaoa_n=16 --qaoa_p=2
qallow phase 14 --jcsv=graph.csv --gain_base=0.001 --gain_span=0.009
```

**Options:**
- `--ticks=N` - Number of ticks (default: 500)
- `--nodes=N` - Lattice nodes (default: 256)
- `--target_fidelity=F` - Success threshold (default: 0.981)
- `--alpha=A` - Explicit alpha override
- `--tune_qaoa` - Invoke QAOA tuner
- `--qaoa_n=N` - QAOA problem size (default: 16)
- `--qaoa_p=P` - QAOA depth (default: 2)
- `--jcsv=FILE` - CUDA CSR J-couplings for alpha derivation
- `--gain_base=B` - Base gain (default: 0.001)
- `--gain_span=S` - Gain span (default: 0.009)
- `--export=FILE` - Export JSON summary

#### Phase 15 (Convergence & Lock-in)

```bash
qallow phase 15 --ticks=400 --eps=1e-5
qallow phase 15 --ticks=500 --export=/tmp/convergence.json
```

**Options:**
- `--ticks=N` - Max ticks (default: 400)
- `--eps=E` - Convergence tolerance (default: 1e-5)
- `--export=FILE` - Export JSON summary

#### Examples

```bash
# Run Phase 12 with custom parameters
qallow phase 12 --ticks=150 --eps=0.00005

# Run Phase 13 with 20 nodes and coupling constant
qallow phase 13 --nodes=20 --ticks=600 --k=0.0015

# Run Phase 14 with QAOA tuning
qallow phase 14 --ticks=700 --nodes=512 --tune_qaoa --qaoa_n=20 --qaoa_p=3

# Run Phase 15 with convergence tolerance and export
qallow phase 15 --ticks=500 --eps=1e-6 --export=/tmp/phase15_results.json
```

---

### 4. `qallow mind` - Cognitive Pipeline

**Purpose:** Run cognitive and neuromorphic processing pipelines.

#### Subcommands

| Subcommand | Purpose |
|-----------|---------|
| `pipeline` | Run cognitive module pipeline (default) |
| `bench` | Run cognition benchmarks |

#### Examples

```bash
qallow mind pipeline
qallow mind bench
QALLOW_MIND_STEPS=100 qallow mind pipeline
```

---

## Advanced Features

### Integrated Phase Pipeline

Run multiple phases sequentially with coordinated parameters:

```bash
# Run phases 12, 13, 14, 15 with unified ticks
qallow run vm --integrate --integrate-ticks=256

# Run phases 12-15 with custom phase overrides
qallow run vm --integrate \
  --integrate-phase12-ticks=150 \
  --integrate-phase13-ticks=200 \
  --integrate-phase13-k=0.0025

# Run phases 12-15 without splitting between phases
qallow run vm --integrate --no-split

# Run phase 11 (quantum bridge) + phases 12-15
export QALLOW_QISKIT=1
qallow run vm --integrate phase11 --integrate-phase11-ticks=100
```

### Accelerator Mode (Phase 13)

```bash
# Watch a directory for files to process
qallow run accelerator --watch=/data/input --threads=auto

# Process specific files
qallow run accelerator --file=/tmp/input1.bin --file=/tmp/input2.bin

# Export results
qallow run accelerator --watch=/data --export=/tmp/results.json
```

### Meta-Introspection & Auditing

```bash
# Enable phase 16 self-auditing
qallow run vm --self-audit

# Custom audit directory
qallow run vm --self-audit-path /var/lib/qallow/audits

# Export pocket topology after run
qallow run vm --self-audit --export-pocket-map /tmp/pockets.json
```

### Deep Learning Integration

```bash
# Load TorchScript model for inference
qallow run vm --dl-model model.pt

# Prefer GPU for ML workloads
qallow run vm --dl-model model.pt --dl-device=gpu

# Force CPU
qallow run vm --dl-model model.pt --dl-device=cpu
```

---

## Deprecated Commands

The following commands are still supported but show deprecation warnings:

| Old Command | New Command |
|------------|------------|
| `qallow build` | `qallow system build` |
| `qallow clear` | `qallow system clear` |
| `qallow verify` | `qallow system verify` |
| `qallow bench` | `qallow run bench` |
| `qallow live` | `qallow run live` |
| `qallow accelerator` | `qallow run accelerator` |
| `qallow phase11` | `qallow phase 11` |
| `qallow phase12` | `qallow phase 12` |
| `qallow phase13` | `qallow phase 13` |

```bash
# These still work but will show deprecation notices:
qallow build          # -> qallow system build
qallow phase12 --ticks=100  # -> qallow phase 12 --ticks=100
```

---

## Environment Variables

```bash
# Quantum integration
export QALLOW_QISKIT=1          # Enable Phase 11 quantum bridge
export QALLOW_PHASE11_SHOTS=1024  # Override Phase 11 shots
export QALLOW_PHASE11_STATES="-1,0,1"  # Override Phase 11 states

# Logging & telemetry
export QALLOW_LOG_DIR=/var/log/qallow    # Override log directory
export QALLOW_LOG=data/logs/telemetry.csv  # Enable CSV logging
export QALLOW_DASHBOARD_INTERVAL=100    # Dashboard update frequency

# Hardware modes
export QALLOW_MODE=hardware              # Route to quantum hardware

# Mind pipeline
export QALLOW_MIND_STEPS=50             # Pipeline iteration count

# Build
export QALLOW_ROOT=/path/to/qallow      # Override project root
export QALLOW_SKIP_BUILD_ONCE=1         # Skip one rebuild cycle
```

---

## Lightning Agent Loop (Automatic Improvement)

For continuous automatic codebase improvement:

```bash
# Run Lightning Agent auto-improvement loop
bash run_agent_lightning_loop.sh
```

This loop:
1. Analyzes code for issues
2. Auto-fixes detected problems
3. Runs test suite
4. Benchmarks performance (CUDA vs CPU)
5. Iterates continuously (Ctrl+C to stop)

See `run_agent_lightning_loop.sh` and `qallow_lightning_integration.py` for details.

---

## Help System

```bash
# Show main help
qallow help

# Show group-specific help
qallow run help
qallow system help
qallow phase help
qallow mind help

# Show phase-specific help
qallow phase help
```

---

## Usage Patterns

### Development Workflow

```bash
# 1. Build project
qallow system build

# 2. Run verification
qallow system verify

# 3. Execute main pipeline
qallow run unified

# 4. Run specific phase with custom parameters
qallow phase 13 --nodes=32 --ticks=400 --k=0.0015

# 5. Enable auto-improvement
bash run_agent_lightning_loop.sh &
```

### Performance Benchmarking

```bash
# Benchmark with standard parameters
qallow run bench

# Benchmark Phase 13 specifically
qallow phase 13 --nodes=256 --ticks=600 --audit-tag=bench

# Benchmark with full pipeline and 500 ticks
qallow run vm --integrate --integrate-ticks=500 --dashboard=off
```

### Hardware Quantum Integration

```bash
# Enable Qiskit/IBM Quantum
export QALLOW_QISKIT=1

# Run Phase 11 on real hardware
qallow phase 11 --ticks=400 --hardware-only

# Or via integrated pipeline
qallow run vm --integrate phase11 --integrate-phase11-hardware
```

### Production Deployment

```bash
# Clean build
qallow system clear
qallow system build

# Verify system health
qallow system verify

# Enable self-auditing and run with export
qallow run vm --self-audit --export-pocket-map /var/lib/qallow/pockets.json --dashboard=off

# Export telemetry
export QALLOW_LOG=/var/log/qallow/telemetry.csv
qallow run unified
```

---

## Troubleshooting

### Build Failures

```bash
# Clean and rebuild
qallow system clear
qallow system build

# Check build logs
cat build/CPU/build.log

# Manual CMake build with debug info
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --parallel
```

### Phase Execution Issues

```bash
# Enable verbose logging
export QALLOW_LOG_DIR=/tmp/qallow_logs
mkdir -p /tmp/qallow_logs

# Run phase with audit tag
qallow phase 13 --audit-tag=debug --ticks=100

# Check telemetry output
cat data/logs/phase13.csv
```

### Quantum Integration Issues

```bash
# Verify Qiskit installation
python3 -c "import qiskit; print(qiskit.__version__)"

# Check Python binary detection
export QALLOW_PYTHON=/path/to/python3
qallow phase 11 --ticks=100

# Test Phase 11 bridge
qallow phase 11 --hardware-only --ticks=100
```

---

## Summary of Key Changes

**From:** Multiple separate commands for each phase
```bash
qallow phase11 --ticks=100
qallow phase12 --ticks=100
qallow phase13 --ticks=100
```

**To:** Unified command structure with subgroups
```bash
qallow phase 11 --ticks=100
qallow phase 12 --ticks=100
qallow phase 13 --ticks=100

# Or integrated pipeline
qallow run unified --integrate-phase13-ticks=100
```

**Benefits:**
- ✅ Single entry point (`qallow`)
- ✅ Consistent command structure across all operations
- ✅ No separate phase runners
- ✅ Grouped functionality (run, system, phase, mind)
- ✅ Backward compatible with deprecation warnings
- ✅ Cleaner help system
- ✅ Easier discoverability
- ✅ Reduced mental model complexity

---

## See Also

- [`README.md`](README.md) - Project overview
- [`docs/ARCHITECTURE_SPEC.md`](docs/ARCHITECTURE_SPEC.md) - Technical architecture
- [`run_agent_lightning_loop.sh`](run_agent_lightning_loop.sh) - Auto-improvement loop
- [`qallow_lightning_integration.py`](qallow_lightning_integration.py) - Lightning Agent bridge
