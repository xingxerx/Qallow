# How to Run Qallow Project

## Quick Start (Fastest Path)

### 1. **Build the Project**
```bash
cd /home/xing/Qallow
./scripts/build_all.sh
```
Or manually with CMake:
```bash
cmake -S . -B build -DQALLOW_ENABLE_CUDA=ON
cmake --build build --parallel 16
```

**Build time**: ~1-2 minutes (first build), ~5 seconds (incremental)

### 2. **Run the Main CLI**

The easiest way to run everything:
```bash
# Run unified phases (12-15 with default ticks)
./build/qallow run unified

# Run a specific phase
./build/qallow phase 13 --ticks=400
./build/qallow phase 14 --ticks=600 --nodes=256
./build/qallow phase 15 --ticks=800
```

---

## Complete Command Reference

### Main Binary: `./build/qallow`

**Unified Run (All Phases)**
```bash
./build/qallow run unified
./build/qallow run unified --integrate-phase13-ticks=400
./build/qallow run unified --integrate-phase13-k=0.003
```

**Individual Phases**
```bash
./build/qallow phase 13 --ticks=400 --log=data/logs/phase13.csv
./build/qallow phase 14 --ticks=600 --nodes=256 --target_fidelity=0.981
./build/qallow phase 15 --ticks=800 --eps=5e-6
```

**With CUDA & Quantum Acceleration**
```bash
QALLOW_ENABLE_CUDA=ON QALLOW_CIRQ=1 ./build/qallow run unified
```

---

## Run Examples & Demos

### Run Phase Demos (Pre-built)
```bash
./build/phase01_demo --ticks=100
./build/phase07_demo --ticks=100
./build/phase13_demo --ticks=400
```

### Run Quantum Algorithms
```bash
cd /home/xing/Qallow
python examples/quantum_adaptive_demo.py --episodes 5 --simulate
python examples/quantum_bandit_policy.py --episodes 10
```

---

## Run Tests

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

### Run Tests with Verbose Output
```bash
ctest -V --output-on-failure
```

---

## Run Continuous Improvement Daemon (Lightning Agent)

The daemon automatically improves code quality in the background.

### Start Daemon
```bash
QALLOW_CIRQ=1 QALLOW_ENABLE_CUDA=ON python3 lightning_agent_fast.py \
  --fast --use-cuda --daemon --max-iterations=500
```

### Monitor Daemon
```bash
# Watch log in real-time
tail -f agent_daemon.log

# Check iterations completed
grep "Iteration" agent_daemon.log | tail -1

# Check commits made
git log --oneline | head -10
```

### Stop Daemon
```bash
pkill -f "lightning_agent_fast.py"
```

---

## Run Web Application

### Start Web Server
```bash
cd /home/xing/Qallow/server
node api-web.js
# Opens at http://localhost:3000
```

### Start Native GUI App (Rust/FLTK)
```bash
cd /home/xing/Qallow/native_app
cargo run --release
```

---

## Run with Profiling

### Profile with CUDA (Nsight Compute)
```bash
ncu --set=detailed ./build/qallow phase 13 --ticks=100
```

### Profile with Linux Perf
```bash
perf record -g ./build/qallow phase 13 --ticks=100
perf report
```

### Profile with Custom Timer (Built-in)
```bash
./build/qallow phase 13 --ticks=100 --profile
```

---

## Development Workflow

### Edit → Build → Test → Iterate
```bash
# 1. Edit source code
vim src/file.c

# 2. Build incrementally
cmake --build build --parallel 16

# 3. Run tests
ctest --output-on-failure -R "related_test"

# 4. Run daemon for automatic improvements
python3 lightning_agent_fast.py --fast --daemon
```

### Clean Build
```bash
rm -rf build
cmake -S . -B build -DQALLOW_ENABLE_CUDA=ON
cmake --build build --parallel 16
```

---

## Key Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `QALLOW_ENABLE_CUDA` | OFF | Enable GPU acceleration |
| `QALLOW_CIRQ` | 1 | Use Cirq quantum simulator |
| `QALLOW_PROFILE_SCOPE` | 0 | Enable profiling hooks |
| `QALLOW_LOG_LEVEL` | INFO | Set logging verbosity |

Example:
```bash
QALLOW_ENABLE_CUDA=ON QALLOW_CIRQ=1 QALLOW_LOG_LEVEL=DEBUG ./build/qallow run unified
```

---

## Common Tasks

### Build Only
```bash
./scripts/build_all.sh --cpu  # CPU only
./scripts/build_all.sh --cuda # CPU + CUDA
```

### Run Benchmark
```bash
cd /home/xing/Qallow/build
./qallow_throughput_bench
```

### Run Integration Test
```bash
cd /home/xing/Qallow/build
./qallow_integration_smoke
```

### View Logs
```bash
# Latest run
cat data/logs/phase13.csv
cat data/logs/sequential_benchmark.csv

# Daemon activity
tail -f agent_daemon.log
```

---

## Troubleshooting

### Build Fails
```bash
# Clean and rebuild
rm -rf build
cmake -S . -B build -DQALLOW_ENABLE_CUDA=ON
cmake --build build --parallel 16
```

### CUDA Not Found
```bash
# Check CUDA installation
nvcc --version

# Ensure CUDA path is set
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
```

### Tests Failing
```bash
# Run with verbose output
ctest -V --output-on-failure

# Check specific test details
cat build/Testing/Temporary/LastTest.log
```

### Daemon Not Committing
```bash
# Configure git (one-time)
git config user.email "agent@qallow.local"
git config user.name "Lightning Agent"

# Restart daemon
pkill -f lightning_agent
QALLOW_CIRQ=1 QALLOW_ENABLE_CUDA=ON python3 lightning_agent_fast.py --daemon
```

---

## Project Status

✅ **Build**: Working  
✅ **Tests**: 10 test suites passing  
✅ **CUDA**: Enabled and accelerating  
✅ **Cirq**: Quantum bridge ready  
✅ **Daemon**: Continuous improvement running  
✅ **Web**: Server running on port 3000  

---

## Quick Reference

```bash
# One-liner: Build + Run + Test
./scripts/build_all.sh && ctest --test-dir build && ./build/qallow run unified

# One-liner: Build + Start Daemon
./scripts/build_all.sh && QALLOW_CIRQ=1 QALLOW_ENABLE_CUDA=ON python3 lightning_agent_fast.py --daemon
```
