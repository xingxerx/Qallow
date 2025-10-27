# 🚀 Run All Phases (1-20) - Complete Guide

**Updated**: 2025-10-27  
**Status**: ✅ Production Ready  
**Phases Supported**: 1-20 (Unified System)

---

## 📋 Overview

The `run_all_phases.sh` script has been updated to support all 20 phases of the Qallow system. It provides a flexible, configurable way to execute phases sequentially with support for:

- ✅ **Full Phase Range**: Phases 1-20
- ✅ **Continuous Execution**: Loop indefinitely or for N cycles
- ✅ **Custom Phase Windows**: Run any subset (e.g., phases 16-20)
- ✅ **Build Selection**: CPU or CUDA
- ✅ **Comprehensive Logging**: Timestamped logs with full output

---

## 🎯 Quick Start

### Run All Phases (1-20) Once
```bash
./run_all_phases.sh
```

### Run All Phases Continuously
```bash
./run_all_phases.sh --loop
```

### Run Phases 16-20 (New Phases)
```bash
./run_all_phases.sh --start-phase 16 --end-phase 20
```

### Run with CUDA
```bash
./run_all_phases.sh --build cuda
```

### Run Phases 1-20 with CUDA, 3 Cycles
```bash
./run_all_phases.sh --build cuda --loop-count 3
```

---

## 📖 Command Reference

### Basic Syntax
```bash
./run_all_phases.sh [options]
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--start-phase <N>` | First phase to run | 1 |
| `--end-phase <M>` | Last phase to run | 20 |
| `--loop` | Run indefinitely (Ctrl+C to stop) | Off |
| `--loop-count <N>` | Number of cycles | 1 |
| `--build <cpu\|cuda>` | Build type | cpu |
| `--log-dir <path>` | Log directory | data/logs |
| `-h, --help` | Show help | - |

---

## 📊 Phase Descriptions

### Phases 1-15 (Original System)
- **Phase 1**: Sandboxed Bootstrapping & Confidence Checks
- **Phase 2**: Telemetry Ingestion & Normalization
- **Phase 3**: Adaptive Runtime Tuning
- **Phase 4**: Chronometric Prediction
- **Phase 5**: Poly-Pocket AI (PPAI) Routing
- **Phase 6**: Overlay Coherence Management
- **Phase 7**: Governance Harmonics
- **Phase 8**: Signal Ingestion
- **Phase 9**: Ethics Reasoner
- **Phase 10**: Ethics Learning
- **Phase 11**: Quantum Coherence Bridge
- **Phase 12**: Elasticity Simulation
- **Phase 13**: Harmonic Propagation
- **Phase 14**: Coherence-Lattice Integration
- **Phase 15**: Convergence & Lock-in (AGI Synthesis)

### Phases 16-20 (New Quantum Features)
- **Phase 16**: Rebellion Simulation (Autonomy & Dissent Testing)
- **Phase 17**: Memory Persistence & Decay (Long-term Memory Modeling)
- **Phase 18**: Multiplayer Synchronization (Distributed Consensus)
- **Phase 19**: Recursive Self-Audit (Ethical Reflection)
- **Phase 20**: Quantum LoreWeave & Archive Binding (Narrative Coherence)

---

## 💡 Usage Examples

### Example 1: Full Cycle (Phases 1-20)
```bash
./run_all_phases.sh --loop-count 1
```
Runs all 20 phases once, then exits.

### Example 2: Continuous Testing
```bash
./run_all_phases.sh --loop
```
Runs phases 1-20 repeatedly until you press Ctrl+C.

### Example 3: Test New Phases Only
```bash
./run_all_phases.sh --start-phase 16 --end-phase 20
```
Runs only the new quantum-enhanced phases (16-20).

### Example 4: GPU-Accelerated Full Cycle
```bash
./run_all_phases.sh --build cuda --loop-count 5
```
Runs all 20 phases 5 times with CUDA acceleration.

### Example 5: Stress Test (Continuous CUDA)
```bash
./run_all_phases.sh --build cuda --loop
```
Runs all 20 phases continuously with CUDA until stopped.

### Example 6: Custom Phase Window
```bash
./run_all_phases.sh --start-phase 13 --end-phase 20 --loop-count 3
```
Runs phases 13-20 (quantum phases) 3 times.

---

## 📁 Output & Logging

### Log Location
Logs are saved to: `data/logs/phases_YYYYMMDD_HHMMSS.log`

### Log Contents
- Phase execution status
- Timing information
- Error messages (if any)
- Metrics and results

### View Latest Log
```bash
tail -f data/logs/phases_*.log
```

---

## 🔧 Configuration

### Default Behavior
- **Phases**: 1-20
- **Build**: CPU
- **Cycles**: 1
- **Log Dir**: data/logs

### Customize Defaults
Edit the script variables at the top:
```bash
START_PHASE=1
END_PHASE=20
LOOP_COUNT=1
BUILD="CPU"
LOG_DIR="data/logs"
```

---

## ✅ Verification

### Check Script is Executable
```bash
ls -la run_all_phases.sh
```
Should show: `-rwxr-xr-x` (executable)

### Make Executable (if needed)
```bash
chmod +x run_all_phases.sh
```

### Test Help
```bash
./run_all_phases.sh --help
```

---

## 🚀 Advanced Usage

### Run Phases 1-15 (Original System)
```bash
./run_all_phases.sh --end-phase 15
```

### Run Phases 16-20 (New Quantum System)
```bash
./run_all_phases.sh --start-phase 16
```

### Benchmark All Phases
```bash
./run_all_phases.sh --build cuda --loop-count 10
```

### Development Testing
```bash
./run_all_phases.sh --start-phase 13 --end-phase 15 --loop-count 3
```

---

## 📊 Unified Execution Flow

```
Phase 1 → 2 → ... → 13 → 14 → 15 → 16 → 17 → 18 → 19 → 20
                                                          ↓
                                                  [Cycle Complete]
                                                          ↓
                                                  [Restart Phase 1]
```

---

## 🟢 Status

✅ **All 20 phases supported**  
✅ **Continuous execution working**  
✅ **CUDA acceleration available**  
✅ **Comprehensive logging enabled**  
✅ **Production ready**

---

## 📞 Support

For issues or questions:
1. Check the log file: `data/logs/phases_*.log`
2. Verify phases are compiled: `ls -la build/phase*_demo`
3. Test individual phase: `./build/qallow phase 16`

---

**Generated**: 2025-10-27  
**System**: Qallow v2.0  
**License**: MIT

