# Qallow - Final Command Reference

## Overview

Qallow is a unified command-line system for the Photonic & Quantum Hardware Emulation VM with autonomous governance, ethics enforcement, and Phase 6 live data integration.

---

## Commands

### 1. Build

```bash
qallow build
```

**Purpose**: Detect toolchain and compile CPU + CUDA backends

**Output**:
```
[QALLOW] Building Qallow...
[BUILD] Qallow is already built!
[BUILD] To rebuild, run: scripts\build_wrapper.bat CPU
[BUILD] Or for CUDA: scripts\build_wrapper.bat CUDA
```

---

### 2. Run

```bash
qallow run
```

**Purpose**: Execute the VM (auto-selects CPU/CUDA)

**Output**:
```
[SYSTEM] Qallow VM initialized
[SYSTEM] Execution mode: CPU
[KERNEL] Node count: 256 per overlay
[KERNEL] Max ticks: 1000

[MAIN] Starting VM execution loop...

[TICK 0000] Coherence: 0.9992 | Decoherence: 0.000010 | Stability: 0.9984 0.9982 0.9984
[KERNEL] System reached stable equilibrium at tick 0

[MAIN] VM execution completed
[TELEMETRY] Benchmark logged: compile=0.0ms, run=1.00ms, mode=CPU
```

---

### 3. Bench

```bash
qallow bench
```

**Purpose**: Run HITL benchmark with logging

**Output**: Same as `run` with benchmark logging enabled

---

### 4. Govern

```bash
qallow govern
```

**Purpose**: Start autonomous governance and ethics audit loop

**Output**:
```
[GOVERN] Starting autonomous governance loop...
[GOVERN] Audit #1: Ethics Score = 2.3000
[GOVERN] WARNING: Ethics score below threshold
[GOVERN] Autonomous governance completed
```

---

### 5. Verify

```bash
qallow verify
```

**Purpose**: System checkpoint - verify integrity before Phase 6

**Output**:
```
[VERIFY] Running system checkpoint...

Status: PASS
Coherence:     0.999200 (min: 0.995000) [OK]
Decoherence:   0.000010 (max: 0.001000) [OK]
Ethics Score:  2.998400 (min: 2.990000) [OK]

SUBSYSTEMS:
  Sandbox:       ACTIVE
  Telemetry:     ACTIVE
  Ethics:        ENFORCED

[VERIFY] System is healthy
```

---

### 6. Live

```bash
qallow live
```

**Purpose**: Phase 6 - Live interface and external data integration

**Output**:
```
[LIVE] Starting Live Interface and External Data Integration
[LIVE] Ingestion manager initialized with 4 streams
[LIVE] Streams configured and ready for data ingestion
[LIVE] - telemetry_primary: http://localhost:9000/telemetry
[LIVE] - sensor_coherence: http://localhost:9001/coherence
[LIVE] - sensor_decoherence: http://localhost:9002/decoherence
[LIVE] - feedback_hitl: http://localhost:9003/feedback

[LIVE] Running VM with live data integration...
[LIVE] Live interface completed
```

---

### 7. Help

```bash
qallow help
```

**Purpose**: Show help message

**Output**:
```
Usage: qallow [mode]

Modes:
  build    Detect toolchain and compile CPU + CUDA backends
  run      Execute the VM (auto-selects CPU/CUDA)
  bench    Run benchmark with logging
  govern   Start governance and ethics audit loop
  verify   System checkpoint - verify integrity
  live     Live interface and external data integration
  help     Show this help message

Examples:
  qallow build      # Build both CPU and CUDA versions
  qallow run        # Run the VM
  qallow bench      # Run benchmark
  qallow govern     # Run governance audit
  qallow verify     # Verify system health
  qallow live       # Start live interface
```

---

## Quick Start

### 1. Verify System

```bash
qallow verify
```

Wait for `Status: PASS` before proceeding.

### 2. Run VM

```bash
qallow run
```

### 3. Run Benchmark

```bash
qallow bench
```

### 4. Run Governance Audit

```bash
qallow govern
```

### 5. Start Phase 6 Live Interface

```bash
qallow live
```

---

## Architecture

### Single Binary

- **Executable**: `build\qallow.exe`
- **Wrapper**: `qallow.bat` (Windows)
- **Wrapper**: `qallow.ps1` (PowerShell)

### Command Routing

```
qallow [command]
    ↓
qallow.bat
    ↓
build\qallow.exe [command]
    ↓
launcher.c (interface/launcher.c)
    ↓
Mode Handler (build/run/bench/govern/verify/live/help)
```

### Subsystems

- **Kernel**: Core VM state and scheduler
- **Ethics**: E = S + C + H enforcement
- **Sandbox**: Isolation and rollback
- **Telemetry**: Logging and monitoring
- **Adaptive**: Learning and reinforcement
- **Governance**: Autonomous audit loop
- **Verify**: System health checks
- **Ingest**: Phase 6 data ingestion
- **Phase 7**: Proactive AGI (deferred)

---

## Files

### Executables

- `build\qallow.exe` - Main unified binary
- `qallow.bat` - Windows batch wrapper
- `qallow.ps1` - PowerShell wrapper

### Source

- `interface/launcher.c` - Command routing
- `interface/main.c` - VM execution
- `backend/cpu/*.c` - Core modules
- `backend/cuda/*.cu` - GPU acceleration

### Configuration

- `io/sensors.json` - Data stream config
- `adapt_state.json` - Adaptive state

### Output

- `qallow_stream.csv` - Telemetry data
- `qallow_bench.log` - Benchmark results
- `phase7_stream.csv` - Phase 7 telemetry

---

## Status

✅ **All 7 Commands Working**
✅ **Build System Complete**
✅ **Phase 6 Integrated**
✅ **Phase 7 Ready**

---

**Version**: Phase VI Complete
**Last Updated**: 2025-10-18

