# ⚡ Qallow Quick Run Guide

**Status**: ✅ Ready to Execute  
**Last Updated**: 2025-10-23

---

## 🚀 One-Command Execution

### Run Complete Pipeline (All Phases)
```bash
cd /root/Qallow

# Phase 13: Harmonic Propagation
./build/qallow phase 13 --ticks=400 --log=data/logs/phase13.csv

# Phase 14: Coherence-Lattice Integration
./build/qallow phase 14 --ticks=600 --nodes=256 --target_fidelity=0.981 --log=data/logs/phase14.csv

# Phase 15: Convergence & Lock-In
./build/qallow phase 15 --ticks=800 --eps=5e-6 --log=data/logs/phase15.csv
```

---

## 🧪 Run Quantum Algorithms

### Activate Python Environment
```bash
cd /root/Qallow
source venv/bin/activate
```

### Run Individual Algorithms
```bash
# Hello Quantum (Basic circuits)
python3 quantum_algorithms/algorithms/hello_quantum.py

# Grover's Algorithm
python3 quantum_algorithms/algorithms/grovers_algorithm.py

# Shor's Algorithm
python3 quantum_algorithms/algorithms/shors_algorithm.py

# VQE Algorithm
python3 quantum_algorithms/algorithms/vqe_algorithm.py
```

### Run ALG Framework (All Algorithms + QAOA)
```bash
python3 alg/main.py run --quick
```

**Output**: `/var/qallow/quantum_report.json` and `/var/qallow/quantum_report.md`

---

## 📊 Run Benchmarks

### Throughput Benchmark
```bash
./build/qallow_throughput_bench
```

### Integration Smoke Test
```bash
./build/qallow_integration_smoke
```

### Full Test Suite
```bash
ctest --test-dir build --output-on-failure
```

---

## 📈 View Results

### Phase Logs
```bash
# View Phase 13 results
cat data/logs/phase13.csv

# View Phase 14 results
cat data/logs/phase14.csv

# View Phase 15 results
cat data/logs/phase15.csv
```

### Quantum Report
```bash
# View JSON report
cat /var/qallow/quantum_report.json

# View Markdown summary
cat /var/qallow/quantum_report.md
```

### Benchmark Logs
```bash
cat data/logs/qallow_bench.log
```

---

## 🔧 Build & Rebuild

### Clean Build (CPU-only)
```bash
cd /root/Qallow
bash scripts/build_all.sh --cpu --clean
```

### Build with CUDA (if available)
```bash
bash scripts/build_all.sh --cuda
```

### Build without Tests
```bash
cmake --build build --parallel
```

---

## 📋 Available Commands

### Main CLI
```bash
./build/qallow --help
./build/qallow help <group>
./build/qallow phase 13
./build/qallow phase 14
./build/qallow phase 15
```

### Phase Demos
```bash
./build/phase01_demo
./build/phase02_demo
# ... through phase13_demo
```

### Utilities
```bash
./build/qallow_ui              # Dashboard
./build/qallow_throughput_bench # Benchmarks
./build/qallow_integration_smoke # Integration tests
```

---

## 📊 Expected Output

### Phase 13 Success
```
[PHASE13] Harmonic propagation complete: pockets=8 ticks=400
[PHASE13] avg_coherence: 0.797500 → 1.000000 ✅
[PHASE13] phase_drift  : 0.100000 → 0.000025 ✅
```

### Phase 14 Success
```
[PHASE14] COMPLETE fidelity=0.981000 [OK] ✅
```

### Phase 15 Success
```
[PHASE15] COMPLETE score=-0.012481 stability=0.000000 ✅
```

### Quantum Algorithms Success
```
✅ hello_quantum: PASSED
✅ bell_state: PASSED
✅ deutsch: PASSED
✅ QAOA Optimizer: CONVERGED (Energy: -4.334)
```

---

## 🐛 Troubleshooting

### Build Fails
```bash
# Clean and rebuild
rm -rf build/
bash scripts/build_all.sh --cpu --clean
```

### Python Environment Issues
```bash
# Recreate venv
rm -rf venv/
python3 -m venv venv
source venv/bin/activate
pip install -r python/requirements-dev.txt
```

### CUDA Not Found
```bash
# Use CPU-only build
bash scripts/build_all.sh --cpu
```

### Tests Fail
```bash
# Run with verbose output
ctest --test-dir build --output-on-failure -VV
```

---

## 📁 Directory Structure

```
/root/Qallow/
├── build/                    # Compiled binaries
│   ├── qallow               # Main executable
│   ├── qallow_unified       # Unified pipeline
│   └── phase*_demo          # Phase demos
├── data/
│   ├── logs/                # Telemetry logs
│   └── quantum/             # Quantum metrics
├── quantum_algorithms/      # Python quantum code
├── alg/                     # ALG framework
└── scripts/                 # Build scripts
```

---

## ⏱️ Typical Execution Times

| Component | Time |
|-----------|------|
| Build (clean) | ~75 seconds |
| Phase 13 | ~1 second |
| Phase 14 | ~2 seconds |
| Phase 15 | ~1 second |
| Quantum Algorithms | ~5 seconds |
| Full Pipeline | ~10 seconds |

---

## 🎯 Common Workflows

### Quick Test (30 seconds)
```bash
./build/qallow phase 13 --ticks=100
./build/qallow phase 14 --ticks=200
./build/qallow phase 15 --ticks=100
```

### Full Analysis (2 minutes)
```bash
./build/qallow phase 13 --ticks=400
./build/qallow phase 14 --ticks=600
./build/qallow phase 15 --ticks=800
source venv/bin/activate
python3 alg/main.py run --quick
```

### Production Run (5 minutes)
```bash
./build/qallow phase 13 --ticks=1000
./build/qallow phase 14 --ticks=1000
./build/qallow phase 15 --ticks=1000
source venv/bin/activate
python3 alg/main.py run
```

---

## 📞 Support

For detailed documentation, see:
- `README.md` - Project overview
- `PIPELINE_EXECUTION_REPORT.md` - Detailed execution report
- `QALLOW_SYSTEM_ARCHITECTURE.md` - System design
- `docs/` - Complete documentation

---

**Ready to run? Start with:**
```bash
cd /root/Qallow
./build/qallow phase 13 --ticks=400
```

✅ **System is operational and ready for deployment!**

