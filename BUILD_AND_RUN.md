# Build & Run Qallow Project - Complete Guide

## 🚀 Quick Start (3 Steps)

### Step 1: Build
```bash
cd /home/xing/Qallow
./scripts/build_all.sh
```
**Time**: ~1-2 minutes (first build), ~5 seconds (incremental)

### Step 2: Run
```bash
# Run unified phases (12-15 with all defaults)
./build/qallow run unified
```

### Step 3: (Optional) Run Tests
```bash
ctest --test-dir build --output-on-failure
```

---

## 📋 Detailed Build Instructions

### Option A: Using Build Script (Recommended)
```bash
cd /home/xing/Qallow

# Build everything with CUDA
./scripts/build_all.sh

# Or specify CPU-only
./scripts/build_all.sh --cpu

# Or specify CUDA
./scripts/build_all.sh --cuda
```

### Option B: Using CMake Directly
```bash
cd /home/xing/Qallow

# Configure
cmake -S . -B build -DQALLOW_ENABLE_CUDA=ON

# Build
cmake --build build --parallel 16

# Optionally install
cmake --install build
```

### Option C: Clean Rebuild
```bash
cd /home/xing/Qallow
rm -rf build
cmake -S . -B build -DQALLOW_ENABLE_CUDA=ON
cmake --build build --parallel 16
```

---

## ▶️ Running the Project

### Main CLI: `./build/qallow`

#### 1. Run All Phases (Unified)
```bash
./build/qallow run unified
```
Default: Runs phases 12-15 with sensible defaults

#### 2. Run Specific Phase
```bash
./build/qallow phase 13 --ticks=400
./build/qallow phase 14 --ticks=600 --nodes=256 --target_fidelity=0.981
./build/qallow phase 15 --ticks=800 --eps=5e-6
```

#### 3. Run with CUDA & Cirq Quantum
```bash
QALLOW_ENABLE_CUDA=ON QALLOW_CIRQ=1 ./build/qallow run unified
```

#### 4. Run with Custom Options
```bash
./build/qallow run unified \
  --integrate-phase13-ticks=400 \
  --integrate-phase13-k=0.003 \
  --integrate-phase14-nodes=512
```

---

## 🎯 Running Examples & Demos

### Phase Demos
```bash
./build/phase01_demo --ticks=100
./build/phase07_demo --ticks=100
./build/phase13_demo --ticks=400
```

### Python Quantum Algorithms
```bash
cd /home/xing/Qallow
python examples/quantum_adaptive_demo.py --episodes 5 --simulate
python examples/quantum_bandit_policy.py --episodes 10
```

### Benchmarks
```bash
./build/qallow_throughput_bench
./build/qallow_integration_smoke
```

---

## 🧪 Running Tests

### Run All Tests
```bash
cd /home/xing/Qallow/build
ctest --output-on-failure
```

### Run Specific Tests
```bash
ctest -R "unit_ethics" --output-on-failure
ctest -R "cuda" --output-on-failure
ctest -R "integration" --output-on-failure
```

### Run with Verbose Output
```bash
ctest -V --output-on-failure
```

### Run Specific Test File
```bash
./unit_ethics_core
./qallow_unit_cuda_parallel
./qallow_integration_smoke
```

---

## 🔧 Development Workflow

### Quick Iterate Loop
```bash
# 1. Edit source
vim src/file.c

# 2. Rebuild incrementally (fast!)
cmake --build build --parallel 16

# 3. Run affected tests
ctest -R "related_test" --output-on-failure

# 4. Run the main program
./build/qallow run unified
```

### Full Clean Build
```bash
cd /home/xing/Qallow
rm -rf build
cmake -S . -B build -DQALLOW_ENABLE_CUDA=ON
cmake --build build --parallel 16
ctest --test-dir build
```

---

## 🤖 Run Continuous Improvement Daemon

The daemon automatically improves code in the background!

### Start Daemon
```bash
QALLOW_CIRQ=1 QALLOW_ENABLE_CUDA=ON python3 lightning_agent_fast.py \
  --fast --use-cuda --daemon --max-iterations=500
```

### Monitor
```bash
# Watch live
tail -f agent_daemon.log

# Check iteration count
grep "Iteration" agent_daemon.log | tail -1

# Check commits
git log --oneline | head -10
```

### Stop
```bash
pkill -f "lightning_agent_fast.py"
```

---

## 🌐 Running Web & GUI

### Web Server
```bash
cd /home/xing/Qallow/server
node api-web.js
# Opens at http://localhost:3000
```

### Native GUI (Rust/FLTK)
```bash
cd /home/xing/Qallow/native_app
cargo run --release
```

---

## 📊 Environment Variables

```bash
# CUDA acceleration
export QALLOW_ENABLE_CUDA=ON

# Quantum bridge
export QALLOW_CIRQ=1

# Profiling
export QALLOW_PROFILE_SCOPE=1

# Logging
export QALLOW_LOG_LEVEL=DEBUG

# Combined example:
QALLOW_ENABLE_CUDA=ON QALLOW_CIRQ=1 QALLOW_LOG_LEVEL=INFO ./build/qallow run unified
```

---

## 🎯 Common Commands

### One-Liner: Build + Run + Test
```bash
./scripts/build_all.sh && \
ctest --test-dir build --output-on-failure && \
./build/qallow run unified
```

### One-Liner: Build + Start Daemon
```bash
./scripts/build_all.sh && \
QALLOW_CIRQ=1 QALLOW_ENABLE_CUDA=ON python3 lightning_agent_fast.py --daemon
```

### Build Only
```bash
./scripts/build_all.sh
```

### Clean Everything
```bash
rm -rf build && ./scripts/build_all.sh
```

### Run Profiling
```bash
ncu --set=detailed ./build/qallow phase 13 --ticks=100
perf record -g ./build/qallow phase 13 --ticks=100
```

---

## 📁 Output & Logs

### Default Locations
```bash
# Build output
./build/

# Logs
./data/logs/phase13.csv
./data/logs/phase14.csv
./data/logs/sequential_benchmark.csv

# Agent daemon log
./agent_daemon.log

# Git commits
git log --oneline
```

---

## ❓ Troubleshooting

### Build Fails
```bash
# Clean and retry
rm -rf build
cmake -S . -B build -DQALLOW_ENABLE_CUDA=ON
cmake --build build --parallel 16
```

### CUDA Not Found
```bash
# Check CUDA
nvcc --version

# Set paths if needed
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

### Tests Failing
```bash
# Run with verbose output
ctest -V --output-on-failure

# Check test logs
cat build/Testing/Temporary/LastTest.log
```

### Runtime Errors
```bash
# Run with debug logging
QALLOW_LOG_LEVEL=DEBUG ./build/qallow run unified
```

---

## ✅ Project Status

| Component | Status |
|-----------|--------|
| Build | ✅ Working |
| Tests | ✅ 8+ test suites |
| CUDA | ✅ Enabled |
| Cirq | ✅ Quantum bridge ready |
| Daemon | ✅ Self-improvement running |
| Web | ✅ Server ready |
| Native | ✅ GUI available |

---

## 🎓 Next Steps

1. **Build**: `./scripts/build_all.sh`
2. **Run**: `./build/qallow run unified`
3. **Explore**: Check `docs/` for detailed documentation
4. **Improve**: Start daemon with `python3 lightning_agent_fast.py --daemon`
5. **Deploy**: Follow `docs/DEPLOYMENT.md` for production setup

---

**Ready to run! Start with:** `./scripts/build_all.sh && ./build/qallow run unified` 🚀
