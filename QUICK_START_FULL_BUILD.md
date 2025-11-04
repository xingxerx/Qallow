# Quick Start: Full Build (CUDA + Cirq-Q + All Phases + Fast Agent)

## One-Line Setup (Complete)

```bash
cd /home/xing/Qallow && ./bootstrap.sh --cuda && source .venv/bin/activate && export QALLOW_CIRQ=1 QALLOW_ENABLE_CUDA=ON && ./build/qallow run unified --integrate-phase11 && python3 agentlightning_runner.py --fast --use-cuda --daemon --max-iterations=500
```

But if you want to understand each step, read below.

---

## Step-by-Step Setup (5-10 Minutes)

### 1. Bootstrap the Project (1-2 minutes)

```bash
cd /home/xing/Qallow
chmod +x bootstrap.sh
./bootstrap.sh --cuda
```

**What this does:**
- Initializes git submodules
- Creates Python virtual environment (.venv)
- Installs all dependencies (PyTorch, Qiskit, Cirq, CUDA support)
- Downloads optional assets (~500MB)
- Builds C/CUDA binaries with CMake
- Runs verification tests

**Output indicators:**
- ✅ `[1/5] Initializing git submodules...` 
- ✅ `[2/5] Setting up Python environment...`
- ✅ `[3/5] Installing dependencies...`
- ✅ `[4/5] Building CMake targets...`
- ✅ `[5/5] Running verification tests...`

---

### 2. Activate Environment (Required)

```bash
source .venv/bin/activate
```

**Verify it worked:**
```bash
python --version  # Should show 3.10+
which python      # Should show /home/xing/Qallow/.venv/bin/python
```

---

### 3. Set Environment Variables (CUDA + Cirq-Q)

```bash
# Enable CUDA acceleration
export QALLOW_ENABLE_CUDA=ON

# Enable Cirq-Q quantum simulator (Phase 11 quantum bridge)
export QALLOW_CIRQ=1

# Optional: Enable profiling hooks for performance analysis
export QALLOW_PROFILE_SCOPE=1

# Optional: Set logging level
export QALLOW_LOG_LEVEL=INFO
```

**Or as one-liner:**
```bash
export QALLOW_ENABLE_CUDA=ON QALLOW_CIRQ=1 QALLOW_PROFILE_SCOPE=1
```

**Verify:**
```bash
echo $QALLOW_ENABLE_CUDA    # Should print: ON
echo $QALLOW_CIRQ           # Should print: 1
```

---

## Run Options

### Option A: Run All Phases Unified (Recommended for First Time)

```bash
# Run phases 12-15 with default ticks (120 each, lattice=64)
./build/qallow run unified

# Run with custom parameters
./build/qallow run unified \
  --integrate-phase11 \
  --integrate-phase12-ticks=150 \
  --integrate-phase13-ticks=400 \
  --integrate-phase14-ticks=600 \
  --integrate-phase15-ticks=800
```

**Expected output:**
```
╔════════════════════════════════════════════════════════════════════════════╗
║              Qallow Unified Pipeline - All Phases                          ║
╚════════════════════════════════════════════════════════════════════════════╝

Phase 11 (Coherence Bridge - Quantum) ....... RUNNING (Cirq-Q)
Phase 12 (Elasticity Simulation) ........... RUNNING (CUDA enabled)
Phase 13 (Harmonic Propagation) ........... RUNNING (CUDA enabled)
Phase 14 (Coherence-Lattice Integration) . RUNNING (CUDA enabled)
Phase 15 (Convergence & Lock-in) ......... RUNNING (CUDA enabled)

Results saved to: data/logs/
```

---

### Option B: Run Individual Phases (For Debugging)

```bash
# Phase 11: Quantum Bridge (requires Cirq)
./build/qallow phase 11 --ticks=64 --states=-1,0,1

# Phase 12: Elasticity 
./build/qallow phase 12 --ticks=150 --eps=0.0001

# Phase 13: Harmonic Propagation (CPU-intensive)
./build/qallow phase 13 --nodes=256 --ticks=400 --k=0.002

# Phase 14: Coherence-Lattice
./build/qallow phase 14 --ticks=600 --nodes=256 --target_fidelity=0.981

# Phase 15: Convergence
./build/qallow phase 15 --ticks=800 --eps=5e-6
```

---

### Option C: Run Benchmarking

```bash
# Profile CPU vs CUDA performance
./build/qallow run bench

# With custom dashboard update frequency (50ms)
./build/qallow run bench --dashboard=50
```

---

### Option D: Run Live Data Ingestion

```bash
# Stream data through pipeline
./build/qallow run live
```

---

## Run Fast Agent (Continuous Improvement - Background)

The lightning agent automatically improves code quality by detecting and fixing issues.

### Start Fast Agent (Non-Blocking)

```bash
# Start as background daemon (runs forever until stopped)
QALLOW_CIRQ=1 QALLOW_ENABLE_CUDA=ON python3 agentlightning_runner.py \
  --fast \
  --use-cuda \
  --daemon \
  --max-iterations=500 &

# Monitor daemon in another terminal
tail -f agent_daemon.log
```

**Output:**
```
[2025-01-15 10:23:45] INFO: Lightning Agent starting (fast mode)...
[2025-01-15 10:23:46] INFO: Build successful (4.2s)
[2025-01-15 10:23:50] INFO: Found 3 issues to fix
[2025-01-15 10:23:52] INFO: Iteration 1/500 complete
[2025-01-15 10:23:55] INFO: Iteration 2/500 complete
...
```

### Monitor Agent Progress

```bash
# Watch real-time log
tail -f agent_daemon.log

# Check iterations completed
grep "Iteration" agent_daemon.log | wc -l

# Check commits made by agent
git log --oneline --author="Lightning Agent" | head -5

# Check for improvements
grep "IMPROVED" agent_daemon.log | wc -l
```

### Stop Agent When Done

```bash
pkill -f "agentlightning_runner.py"
```

---

## Run Full Stack (Everything at Once)

For a complete end-to-end run with all components:

```bash
#!/bin/bash
# Save as run_full_stack.sh

set -e

cd /home/xing/Qallow
source .venv/bin/activate

export QALLOW_ENABLE_CUDA=ON
export QALLOW_CIRQ=1
export QALLOW_PROFILE_SCOPE=1

echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║          Qallow Full Stack - CUDA + Cirq-Q + All Phases + Agent           ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"

echo ""
echo "📊 [1/3] Running Unified Pipeline (All Phases)..."
./build/qallow run unified --integrate-phase11

echo ""
echo "🤖 [2/3] Starting Fast Agent (Background)..."
QALLOW_CIRQ=1 QALLOW_ENABLE_CUDA=ON python3 agentlightning_runner.py \
  --fast --use-cuda --daemon --max-iterations=500 &
AGENT_PID=$!

echo ""
echo "📈 [3/3] Monitoring System..."
sleep 5
echo "Agent PID: $AGENT_PID"
echo "Monitor with: tail -f agent_daemon.log"
echo ""
echo "✅ Full stack running! Press Ctrl+C to stop, or run:"
echo "   pkill -f 'agentlightning_runner.py'"
```

**Run it:**
```bash
chmod +x run_full_stack.sh
./run_full_stack.sh
```

---

## Verification Checklist

### After Bootstrap
- [ ] No errors in build output
- [ ] `ls -la build/qallow` shows binary
- [ ] `./build/qallow --help` displays help text

### After Running Unified Pipeline
- [ ] All 5 phases completed (11-15)
- [ ] CSV logs created in `data/logs/`
- [ ] No "CUDA error" messages
- [ ] Results show in JSON summary

### After Starting Fast Agent
- [ ] Agent log file exists: `agent_daemon.log`
- [ ] Agent running: `ps aux | grep lightning_agent`
- [ ] Agent makes improvements every few iterations
- [ ] No errors in log file

---

## Troubleshooting

### Issue: "CUDA not found" error

**Symptoms:**
```
ERROR: CUDA toolkit not found
WARNING: Falling back to CPU
```

**Solution:**
```bash
# Check CUDA installation
nvidia-smi                # Should show GPU info
nvcc --version           # Should show CUDA version

# If missing, install CUDA:
# https://developer.nvidia.com/cuda-toolkit

# Rebuild with CUDA:
rm -rf build
cmake -S . -B build -DQALLOW_ENABLE_CUDA=ON
cmake --build build --parallel 16
```

---

### Issue: "Cirq not available" error

**Symptoms:**
```
ERROR: Cirq-Q not found, Phase 11 disabled
```

**Solution:**
```bash
# Activate venv first
source .venv/bin/activate

# Check if Cirq is installed
python -c "import cirq; print(cirq.__version__)"

# If missing, reinstall
pip install cirq qiskit

# Verify environment variable
echo $QALLOW_CIRQ  # Should be 1
```

---

### Issue: "Agent not starting" or crashes

**Symptoms:**
```
python3: can't open file 'agentlightning_runner.py': [Errno 2] No such file or directory
```

**Solution:**
```bash
# Make sure you're in correct directory
cd /home/xing/Qallow
ls agentlightning_runner.py   # Should exist

# Run with full path
python3 $(pwd)/agentlightning_runner.py --fast --use-cuda --daemon
```

---

### Issue: "Out of memory" (OOM)

**Symptoms:**
```
CUDA_ERROR_OUT_OF_MEMORY
or
Killed (SIGKILL)
```

**Solution:**
```bash
# Reduce workload parameters
./build/qallow phase 13 --nodes=128 --ticks=200   # Instead of 256/400

# Monitor GPU memory
watch -n 1 nvidia-smi

# Free up memory
sudo sysctl vm.drop_caches=3

# Use CPU instead temporarily
unset QALLOW_ENABLE_CUDA
./build/qallow run unified
```

---

### Issue: Build errors

**Symptoms:**
```
error: 'qallow.h' file not found
cmake: error while loading shared libraries: libcuda.so.1
```

**Solution:**
```bash
# Clean rebuild
rm -rf build CMakeCache.txt cmake_install.cmake

# Rebuild from scratch
./bootstrap.sh --cuda --skip-tests   # Skip tests, just build

# If still failing, check dependencies
cmake --build build --verbose

# View specific error
cat build/CMakeOutput.log
```

---

## Performance Tips

### Maximize Speed

```bash
# Use fast mode + parallel build
export QALLOW_ENABLE_CUDA=ON
export QALLOW_CIRQ=1

# Rebuild with optimizations
cmake -S . -B build \
  -DQALLOW_ENABLE_CUDA=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON

cmake --build build --parallel $(nproc)

# Run with custom ticks (fewer = faster)
./build/qallow phase 13 --nodes=64 --ticks=100   # Fast test
./build/qallow phase 13 --nodes=256 --ticks=400  # Full run
```

### Monitor Performance

```bash
# Watch GPU usage (continuous)
watch -n 0.5 nvidia-smi

# Profile with Nsight Compute
ncu --set=detailed ./build/qallow phase 13 --ticks=100
ncu --export=/tmp/profile.ncu-rep ./build/qallow phase 13 --ticks=100

# Linux perf profiling
perf record -g ./build/qallow phase 13 --ticks=100
perf report
```

---

## Next Steps

1. **Review Results**: Check `data/logs/*.csv` for phase outputs
2. **Run Benchmarks**: `./build/qallow run bench`
3. **Enable Dashboard**: Add `--dashboard=50` for live metrics
4. **Integrate Custom Data**: See `examples/` for custom workflows
5. **Read Architecture**: See `docs/ARCHITECTURE_SPEC.md`

---

## Quick Command Reference

| Task | Command |
|------|---------|
| **Full setup** | `./bootstrap.sh --cuda && source .venv/bin/activate` |
| **Run everything** | `export QALLOW_CIRQ=1 QALLOW_ENABLE_CUDA=ON && ./build/qallow run unified` |
| **Run one phase** | `./build/qallow phase 13 --ticks=400` |
| **Start agent** | `python3 agentlightning_runner.py --fast --daemon` |
| **Monitor agent** | `tail -f agent_daemon.log` |
| **Stop agent** | `pkill -f agentlightning_runner.py` |
| **Rebuild only** | `cmake --build build --parallel` |
| **Clean build** | `rm -rf build && ./bootstrap.sh --cuda` |
| **Run tests** | `cd build && ctest --output-on-failure` |
| **Profile GPU** | `ncu --set=detailed ./build/qallow phase 13` |

---

## Support

- **Docs**: See `/docs/` directory
- **Issues**: Check existing GitHub issues
- **Build problems**: Run `./bootstrap.sh --cuda` again
- **CUDA issues**: Check `nvidia-smi` output

---

**Happy computing! 🚀**

Last updated: November 2025
