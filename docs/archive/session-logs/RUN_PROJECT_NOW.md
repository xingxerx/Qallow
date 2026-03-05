# Run Qallow Project - Complete Guide

## ✅ Bootstrap Complete!

Your bootstrap finished successfully! Now let's run the actual project code.

---

## 🚀 Quick Start (Right Now!)

### Option 1: Run Full Unified Pipeline
```bash
cd /home/xing/Qallow
source .venv/bin/activate
./build/qallow run unified
```

**What it does**:
- ✅ Runs Phase 12 (elasticity simulation)
- ✅ Runs Phase 13 (harmonic propagation)
- ✅ Runs Phase 14 (governance audit)
- ✅ Generates telemetry and logs
- ✅ Outputs results to `data/logs/`

**Expected output**:
```
[INTEGRATE] Running Phase 12 elasticity...
[PHASE12] Elasticity simulation
[PHASE12] ticks=120 eps=0.000100
[PHASE12] Elastic run complete: ticks=120 eps=0.000100
[PHASE12] Coherence≈0.999884 EntropyΔ≈0.000580 Deco≈0.000008

[INTEGRATE] Running Phase 13 harmonic propagation...
[PHASE13] Harmonic propagation
[PHASE13] nodes=8 ticks=120 k=0.002000
[PHASE13] Harmonic propagation complete: pockets=8 ticks=120 k=0.002000
[PHASE13] avg_coherence: 0.797500 → 0.940956

[TELEMETRY] System initialized
[TELEMETRY] Streaming to: data/logs/telemetry_stream.csv
```

---

## 🎯 All Available Commands

### Run Workflows
```bash
# Full unified pipeline (Phase 12, 13, 14)
./build/qallow run unified

# Virtual machine mode
./build/qallow run vm

# Benchmarking
./build/qallow run bench

# Live monitoring
./build/qallow run live

# GPU acceleration
./build/qallow run accelerator
```

### Run Individual Phases
```bash
# Phase 11 (bridge)
./build/qallow phase 11 --ticks=32

# Phase 12 (elasticity)
./build/qallow phase 12 --ticks=120

# Phase 13 (harmonic)
./build/qallow phase 13 --ticks=120

# Phase 14 (governance)
./build/qallow phase 14

# Phase 15 (integration)
./build/qallow phase 15
```

### System Commands
```bash
# Verify build
./build/qallow system verify

# Build project
./build/qallow system build

# Clean build
./build/qallow system clean

# Show help
./build/qallow --help
./build/qallow help run
./build/qallow help phase
```

### Cognitive Pipeline
```bash
# Run mind pipeline
./build/qallow mind pipeline

# Run benchmarks
./build/qallow mind bench

# Show help
./build/qallow help mind
```

---

## 📊 Output & Logs

### Where Results Are Saved
```
data/logs/
├── phase12.csv              # Phase 12 elasticity data
├── phase13.csv              # Phase 13 harmonic data
├── phase_summary.json       # Summary of all phases
├── telemetry_stream.csv     # System telemetry
└── governance_audit.log     # Governance audit trail
```

### View Results
```bash
# View phase 12 results
cat data/logs/phase12.csv

# View phase 13 results
cat data/logs/phase13.csv

# View summary
cat data/logs/phase_summary.json

# View telemetry
cat data/logs/telemetry_stream.csv

# View governance audit
cat data/logs/governance_audit.log
```

---

## 🔧 Advanced Options

### Run with Custom Parameters
```bash
# Phase 12 with custom ticks
./build/qallow phase 12 --ticks=256

# Phase 13 with custom nodes
./build/qallow phase 13 --nodes=16 --ticks=256

# Phase 14 with custom audit tag
./build/qallow phase 14 --audit-tag=custom_run
```

### Run with Debugging
```bash
# Verbose output
./build/qallow run unified --verbose

# Debug mode
./build/qallow run unified --debug

# Trace execution
./build/qallow run unified --trace
```

### Run with GPU
```bash
# Use CUDA acceleration
./build/qallow run accelerator

# Use GPU for phase 12
./build/qallow phase 12 --gpu
```

---

## 📈 Example Workflows

### Workflow 1: Quick Test (2 minutes)
```bash
cd /home/xing/Qallow
source .venv/bin/activate
./build/qallow phase 12 --ticks=32
```

### Workflow 2: Full Pipeline (5 minutes)
```bash
cd /home/xing/Qallow
source .venv/bin/activate
./build/qallow run unified
```

### Workflow 3: Benchmarking (10 minutes)
```bash
cd /home/xing/Qallow
source .venv/bin/activate
./build/qallow run bench
```

### Workflow 4: Live Monitoring (Continuous)
```bash
cd /home/xing/Qallow
source .venv/bin/activate
./build/qallow run live
```

### Workflow 5: GPU Acceleration (5 minutes)
```bash
cd /home/xing/Qallow
source .venv/bin/activate
./build/qallow run accelerator
```

---

## 🧪 Run Tests

### Run All Tests
```bash
cd /home/xing/Qallow
source .venv/bin/activate

# C/CUDA tests
cd build && ctest

# Python tests
python3 -m pytest tests/ -v

# Smoke tests
bash tests/smoke/test_modules.sh
```

### Run Specific Tests
```bash
# Ethics tests
./build/qallow_unit_ethics

# CUDA tests
./build/qallow_unit_cuda_parallel

# DL integration tests
./build/qallow_unit_dl_integration

# Temporal memory tests
./build/qallow_test_temporal_memory
```

---

## 💡 Pro Tips

### Tip 1: Keep Terminal Open
```bash
source .venv/bin/activate
# Now run multiple commands
./build/qallow phase 12 --ticks=32
./build/qallow phase 13 --ticks=32
./build/qallow phase 14
```

### Tip 2: Monitor Output in Real-Time
```bash
# Run in one terminal
./build/qallow run unified

# In another terminal, monitor logs
tail -f data/logs/telemetry_stream.csv
```

### Tip 3: Save Output to File
```bash
./build/qallow run unified > run_output.log 2>&1
```

### Tip 4: Run in Background
```bash
./build/qallow run unified &
# Do other things
jobs
fg  # bring back to foreground
```

---

## 🎯 What Each Phase Does

### Phase 11: Bridge
- Initializes quantum-classical bridge
- Sets up communication channels
- Validates system state

### Phase 12: Elasticity
- Quantum elasticity simulation
- Coherence measurement
- Entropy calculation
- Decoherence tracking

### Phase 13: Harmonic Propagation
- Harmonic wave propagation
- Node synchronization
- Phase drift measurement
- Coherence evolution

### Phase 14: Governance
- Autonomous governance audit
- Decision validation
- Compliance checking
- Audit trail generation

### Phase 15: Integration
- Full system integration
- Cross-phase validation
- Performance metrics
- Final telemetry

---

## 📊 Understanding Output

### Phase 12 Output Example
```
[PHASE12] Elasticity simulation
[PHASE12] ticks=120 eps=0.000100
[PHASE12] Elastic run complete: ticks=120 eps=0.000100
[PHASE12] Coherence≈0.999884 EntropyΔ≈0.000580 Deco≈0.000008
```

**Metrics**:
- `Coherence`: Quantum coherence level (0-1, higher is better)
- `EntropyΔ`: Entropy change (lower is better)
- `Deco`: Decoherence rate (lower is better)

### Phase 13 Output Example
```
[PHASE13] avg_coherence: 0.797500 → 0.940956
[PHASE13] phase_drift  : 0.100000 → 0.000051
```

**Metrics**:
- `avg_coherence`: Average coherence improvement
- `phase_drift`: Phase drift reduction

---

## 🚨 Troubleshooting

### Issue: "command not found: qallow"
**Solution**: Activate environment first
```bash
source .venv/bin/activate
./build/qallow --help
```

### Issue: "Build not found"
**Solution**: Rebuild
```bash
cmake -S . -B build -DQALLOW_ENABLE_CUDA=ON
cmake --build build
```

### Issue: "Permission denied"
**Solution**: Make executable
```bash
chmod +x ./build/qallow
```

### Issue: "Python module not found"
**Solution**: Install dependencies
```bash
source .venv/bin/activate
pip install -r config/requirements.txt
```

---

## 📚 Related Documentation

- `README.md` - Main documentation
- `RUNNING_QALLOW_GUIDE.md` - Running guide
- `QUICK_START_CARD.md` - Quick reference
- `START_TESTING_NOW.md` - Testing guide

---

## ✅ Next Steps

1. **Close VS Code windows** (if you want)
2. **Run the project**:
   ```bash
   cd /home/xing/Qallow
   source .venv/bin/activate
   ./build/qallow run unified
   ```
3. **Check results**:
   ```bash
   cat data/logs/phase_summary.json
   ```
4. **Explore other commands**:
   ```bash
   ./build/qallow --help
   ```

---

## 🎉 You're Ready!

**Run the project now**:
```bash
source .venv/bin/activate
./build/qallow run unified
```

**That's it!** 🚀

---

*Last Updated: 2025-11-12*

