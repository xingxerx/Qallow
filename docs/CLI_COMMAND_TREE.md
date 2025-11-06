# Qallow Unified CLI - Command Tree

```
qallow
├── run                           Execute workflows
│   ├── (default)                → VM execution
│   ├── vm                        Unified VM
│   ├── bench                     Benchmark profile  
│   ├── live                      Live ingestion profile
│   ├── unified|pipeline          Phases 12-15 pipeline
│   ├── accelerator               Phase 13 accelerator
│   ├── entangle                  GHZ/W entanglement generator
│   └── help                      Show run help
│
├── system                        Build & maintenance
│   ├── build                     Compile CPU + CUDA backends
│   ├── clear                     Remove build artifacts
│   ├── verify                    System health checks
│   └── help                      Show system help
│
├── phase                         Individual phase runners
│   ├── 11|phase11               Coherence bridge (quantum)
│   ├── 12|phase12               Elasticity simulation
│   ├── 13|phase13               Harmonic propagation
│   ├── 14|phase14               Coherence-lattice integration
│   ├── 15|phase15               Convergence & lock-in
│   ├── 16-20                    Advanced phases
│   ├── neuro-demo               Neuromorphic demo
│   └── help                      Show phase help
│
├── mind                          Cognitive pipeline
│   ├── pipeline                  Cognitive modules (default)
│   ├── bench                     Benchmarking suite
│   └── help                      Show mind help
│
├── help [topic]                  General help
│   ├── (no args)                → Main help
│   ├── run                      → Run group help
│   ├── system                   → System group help
│   ├── phase                    → Phase group help
│   └── mind                     → Mind group help
│
└── [legacy]                      Deprecated (still work)
    ├── build                    → system build ⚠️
    ├── clear                    → system clear ⚠️
    ├── verify                   → system verify ⚠️
    ├── bench                    → run bench ⚠️
    ├── live                     → run live ⚠️
    ├── accelerator              → run accelerator ⚠️
    ├── phase11                  → phase 11 ⚠️
    ├── phase12                  → phase 12 ⚠️
    └── phase13                  → phase 13 ⚠️
```

---

## RUN Group - Options

```
qallow run [subcommand] [options]

Options for VM execution:
  --integrate [PHASES]          Run integrated phase pipeline
    - phase11, phase12, phase13, phase14, phase15
    - unified, all, full (select all)
    - phase11, 11, bridge (Phase 11)
    - phase12, 12, elasticity (Phase 12)
    - phase13, 13, harmonic (Phase 13)
    - phase14, 14, entanglement, coherence-lattice (Phase 14)
    - phase15, 15, singularity, lock-in (Phase 15)
  
  --integrate-ticks=N           Override all phase ticks
  --integrate-phase11-ticks=N   Phase 11 ticks
  --integrate-phase11-states=S  Phase 11 quantum states
  --integrate-phase11-hardware  Use real quantum hardware
  --integrate-phase12-ticks=N   Phase 12 ticks
  --integrate-phase12-eps=E     Phase 12 epsilon
  --integrate-phase13-ticks=N   Phase 13 ticks
  --integrate-phase13-nodes=N   Phase 13 nodes
  --integrate-phase13-k=F       Phase 13 coupling
  --no-split                    Don't split phases
  --integrate-no-summary        Don't print summary
  
  --bench                       Enable benchmark profile
  --live                        Enable live ingestion profile
  --hardware                    Route Phase 11 to real hardware
  --dashboard=N|off             Dashboard frequency (ticks) or disable
  --self-audit                  Enable phase 16 meta-introspection
  --self-audit-path=DIR         Custom audit directory
  --export-pocket-map=FILE      Export pocket topology
  --dl-model=PATH               Load TorchScript model
  --dl-device=cpu|gpu           ML device preference
  --phase=11|12|13              Dispatch to specific phase
  
For accelerator:
  --threads=N|auto              Worker thread count
  --watch=DIR                   Directory to monitor
  --file=PATH                   Queue file for processing
  --export=FILE                 Export results JSON
```

---

## PHASE Group - Options by Phase

```
qallow phase N [options]

Phase 11 (Coherence Bridge - Quantum):
  --ticks=N                      Number of shots (default: 400)
  --states=S                     Quantum states (default: -1,0,1)
  --hardware-only                Use real quantum hardware

Phase 12 (Elasticity):
  --ticks=N                      Number of ticks (default: 1000)
  --eps=E                        Elasticity epsilon (default: 0.0001)
  --log=FILE                     Override CSV path
  --audit-tag=TAG               Audit tag for telemetry

Phase 13 (Harmonic Propagation):
  --nodes=N                      Harmonic pockets (default: 8)
  --ticks=N                      Number of ticks (default: 400)
  --k=F                          Coupling constant (default: 0.001)
  --log=FILE                     Override CSV path
  --audit-tag=TAG               Audit tag for telemetry

Phase 14 (Coherence-Lattice):
  --ticks=N                      Number of ticks (default: 500)
  --nodes=N                      Lattice nodes (default: 256)
  --target_fidelity=F            Success threshold (default: 0.981)
  --alpha=A                      Explicit alpha override
  --jcsv=FILE                    CUDA CSR J-couplings
  --gain_base=B                  Base gain (default: 0.001)
  --gain_span=S                  Gain span (default: 0.009)
  --gain_json=FILE              JSON alpha override
  --tune_qaoa                    Invoke QAOA tuner
  --qaoa_n=N                     QAOA size (default: 16)
  --qaoa_p=P                     QAOA depth (default: 2)
  --export=FILE                  Export JSON summary

Phase 15 (Convergence & Lock-in):
  --ticks=N                      Max ticks (default: 400)
  --eps=E                        Convergence tolerance (default: 1e-5)
  --export=FILE                  Export JSON summary

Phases 16-20:
  See documentation for specific options
```

---

## SYSTEM Group - Commands

```
qallow system <command>

build                           Compile CPU + CUDA backends
clear                           Remove build artifacts
verify                          Run system health checks
help                            Show system help
```

---

## MIND Group - Commands

```
qallow mind [subcommand] [options]

pipeline                        Cognitive module pipeline (default)
bench                          Benchmarking suite
help                           Show mind help

Environment:
  QALLOW_MIND_STEPS=N          Number of pipeline steps (default: 50)
```

---

## Common Usage Patterns

### Development
```
1. qallow system build         # Build
2. qallow system verify        # Check health
3. qallow run unified          # Execute
4. python3 recursive_improvement_engine.py  # Auto-improve
```

### Testing Individual Phases
```
qallow phase 12 --ticks=50 --eps=0.001
qallow phase 13 --nodes=8 --ticks=100 --k=0.001
qallow phase 14 --ticks=200 --target_fidelity=0.95
```

### Benchmarking
```
qallow run bench               # Benchmark profile
qallow phase 13 --nodes=256 --ticks=600 --audit-tag=bench
```

### Production
```
qallow system build
qallow system verify
qallow run vm --self-audit --export-pocket-map /tmp/pockets.json
```

### Quantum Integration
```
export QALLOW_QISKIT=1
qallow run vm --integrate phase11
qallow phase 11 --hardware-only
```

---

## Environment Variables

```bash
# Quantum backend
QALLOW_QISKIT=1                 Enable Qiskit/IBM Quantum

# Logging
QALLOW_LOG_DIR=/path            Override log directory
QALLOW_LOG=file.csv             Enable CSV logging

# UI
QALLOW_DASHBOARD_INTERVAL=100   Dashboard update frequency (ticks)

# Mind pipeline
QALLOW_MIND_STEPS=50            Pipeline iteration count

# Build
QALLOW_ROOT=/path              Override project root
QALLOW_SKIP_BUILD_ONCE=1        Skip one rebuild cycle

# Hardware
QALLOW_MODE=hardware            Route to quantum hardware
```

---

## Migration Guide

### Command Changes

```
OLD COMMAND                  NEW COMMAND
qallow build                 qallow system build
qallow clear                 qallow system clear
qallow verify                qallow system verify
qallow bench                 qallow run bench
qallow live                  qallow run live
qallow accelerator           qallow run accelerator
qallow phase11               qallow phase 11
qallow phase12               qallow phase 12
qallow phase13               qallow phase 13
```

**Important:** Old commands still work with deprecation warnings. New syntax is preferred.

---

## Help Command Structure

```
qallow help                  Main help screen
qallow run help              Run group help + examples
qallow system help           System group help + examples
qallow phase help            Phase group help + examples
qallow mind help             Mind group help + examples
```

---

## Key Features

✅ **Single Entry Point** - `qallow` is the only command
✅ **Logical Grouping** - Four clear command groups
✅ **Unified Phases** - Consistent phase interface (11-20)
✅ **Flexible Pipelines** - Run single phases or full pipelines
✅ **Help System** - Complete documentation built-in
✅ **Backward Compatible** - Old commands still work
✅ **Easy to Extend** - Clear structure for new commands

---

## Example Execution Flows

### Flow 1: Quick Test
```
$ qallow system verify
$ qallow phase 13 --ticks=100
$ qallow phase 14 --ticks=100
```

### Flow 2: Full Pipeline
```
$ qallow system build
$ qallow system verify
$ qallow run unified
```

### Flow 3: Quantum Integration
```
$ export QALLOW_QISKIT=1
$ qallow run vm --integrate phase11 --integrate-phase11-hardware
$ qallow phase 11 --ticks=400
```

### Flow 4: Benchmarking
```
$ qallow system build
$ qallow run bench --dashboard=50
$ qallow phase 13 --nodes=256 --ticks=600 --audit-tag=benchmark
```

### Flow 5: Auto-Improve
```
$ python3 recursive_improvement_engine.py
$ qallow run unified
$ qallow phase 13 --nodes=32 --ticks=400
```

---

**See `UNIFIED_CLI.md` for complete documentation**
