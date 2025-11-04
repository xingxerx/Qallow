# How to Run Qallow - Full Build with CUDA + Cirq-Q + All Phases + Fast Agent

## TL;DR - Three Ways to Run

### 🚀 **Easiest (Recommended for First Time)**
```bash
cd /home/xing/Qallow
chmod +x run_full_build.sh
./run_full_build.sh
```
This does everything automatically:
- ✅ Checks prerequisites (Python, CMake, NVIDIA GPU)
- ✅ Sets up Python environment (first time only)
- ✅ Configures CUDA + Cirq-Q
- ✅ Builds project (if needed)
- ✅ Runs all phases (11-15)
- ✅ Starts fast agent in background

---

### 🎯 **Manual Step-by-Step (For Understanding)**
```bash
cd /home/xing/Qallow

# 1. Bootstrap (if not done)
./bootstrap.sh --cuda

# 2. Activate environment
source .venv/bin/activate

# 3. Set environment variables
export QALLOW_ENABLE_CUDA=ON
export QALLOW_CIRQ=1

# 4. Run all phases
./build/qallow run unified --integrate-phase11

# 5. Start fast agent (background)
python3 agentlightning_runner.py --fast --use-cuda --daemon &
```

---

### 💻 **One-Liner (For Experienced Users)**
```bash
cd /home/xing/Qallow && ./bootstrap.sh --cuda && source .venv/bin/activate && export QALLOW_ENABLE_CUDA=ON QALLOW_CIRQ=1 && ./build/qallow run unified --integrate-phase11 && python3 agentlightning_runner.py --fast --use-cuda --daemon &
```

---

## Detailed Instructions

### What You'll Get

| Component | Status | Details |
|-----------|--------|---------|
| **CUDA GPU** | ✅ Enabled | Automatic GPU detection & setup |
| **Cirq-Q** | ✅ Enabled | Quantum bridge (Phase 11) |
| **All Phases** | ✅ 11-15 | Quantum bridge + elasticity + harmonics + convergence |
| **Fast Agent** | ✅ Background | Continuous code improvement loop |
| **Profiling** | ✅ Enabled | Performance metrics & telemetry |

### Prerequisites

**Required:**
- Linux/macOS/WSL2 (Ubuntu 22.04+ recommended)
- Python 3.10+
- CMake 3.20+
- git

**Optional:**
- NVIDIA GPU + CUDA 11.8+ (will fall back to CPU)
- ~10GB disk space
- ~4GB RAM (8GB+ for GPU)

**Check you have them:**
```bash
python3 --version
cmake --version
git --version
nvidia-smi              # If you have GPU
```

---

## Execution Methods

### Method 1: Automated Script (Best for First Time)

```bash
./run_full_build.sh
```

**Supports options:**
```bash
./run_full_build.sh --phases-only      # Just run phases, no agent
./run_full_build.sh --agent-only       # Just run agent, no phases
./run_full_build.sh --quick            # Skip Phase 11 (faster)
./run_full_build.sh --no-agent         # Run phases, no background agent
./run_full_build.sh --help             # Show help
```

**Output:**
```
╔════════════════════════════════════════════════════════════════════════════════╗
║    Qallow Full Build - CUDA + Cirq-Q + All Phases + Fast Agent                ║
╚════════════════════════════════════════════════════════════════════════════════╝

[Setup] Checking prerequisites...
✓ Python 3 found
✓ CMake found
✓ NVIDIA GPU detected

[Setup] Activating virtual environment...
✓ Virtual environment activated

[Setup] Configuring environment variables...
✓ CUDA enabled
✓ Cirq-Q enabled
✓ Profiling enabled

[Phases] Running unified pipeline...

Running: ./build/qallow run unified --integrate-phase11

Phase 11 (Coherence Bridge - Quantum) ....... RUNNING (Cirq-Q)
Phase 12 (Elasticity Simulation) ........... RUNNING (CUDA enabled)
Phase 13 (Harmonic Propagation) ........... RUNNING (CUDA enabled)
Phase 14 (Coherence-Lattice Integration) . RUNNING (CUDA enabled)
Phase 15 (Convergence & Lock-in) ......... RUNNING (CUDA enabled)

✓ Unified phases complete
✓ Results saved to: data/logs/

[Agent] Starting fast agent in background...
✓ Fast agent started (PID: 12345)

✓ Full build complete!
```

---

### Method 2: Manual Setup (For Understanding)

If you want to understand what's happening step-by-step:

#### Step 1: Bootstrap (First Time Only)
```bash
cd /home/xing/Qallow
chmod +x bootstrap.sh
./bootstrap.sh --cuda
```

This automatically:
- Initializes git submodules
- Creates Python venv in `.venv/`
- Installs all dependencies
- Builds C/CUDA binaries
- Runs verification tests

**Time:** ~10-20 minutes (first run), ~1-2 minutes (cached)

#### Step 2: Activate Environment
```bash
source .venv/bin/activate
```

Verify it worked:
```bash
which python     # Should show: /home/xing/Qallow/.venv/bin/python
python --version # Should show: Python 3.10+
```

#### Step 3: Configure Environment Variables
```bash
# Enable CUDA acceleration
export QALLOW_ENABLE_CUDA=ON

# Enable Cirq-Q quantum simulator (needed for Phase 11)
export QALLOW_CIRQ=1

# Enable profiling hooks
export QALLOW_PROFILE_SCOPE=1

# Set logging level
export QALLOW_LOG_LEVEL=INFO
```

Verify:
```bash
echo $QALLOW_ENABLE_CUDA    # Should print: ON
echo $QALLOW_CIRQ           # Should print: 1
```

#### Step 4: Run Unified Pipeline (All Phases)
```bash
# Simple: run phases 12-15 with default parameters
./build/qallow run unified

# Or with Phase 11 (quantum bridge) included:
./build/qallow run unified --integrate-phase11

# Or with custom parameters:
./build/qallow run unified \
  --integrate-phase11 \
  --integrate-phase12-ticks=150 \
  --integrate-phase13-ticks=400 \
  --integrate-phase14-ticks=600 \
  --integrate-phase15-ticks=800
```

**Expected output:**
```
╔════════════════════════════════════════════════════════════════════════════════╗
║              Qallow Unified Pipeline - All Phases                              ║
╚════════════════════════════════════════════════════════════════════════════════╝

[11/15] Phase 11 - Coherence Bridge (Quantum) ............. OK ✓
[12/15] Phase 12 - Elasticity Simulation ............. OK ✓
[13/15] Phase 13 - Harmonic Propagation .................... OK ✓
[14/15] Phase 14 - Coherence-Lattice Integration .......... OK ✓
[15/15] Phase 15 - Convergence & Lock-in .................. OK ✓

Results:
  CSV Log:  data/logs/unified_run.csv
  JSON:     data/logs/unified_run.json
  Summary:  data/logs/summary.txt

✓ Pipeline complete
```

**Time:** ~5-15 minutes depending on tick parameters

#### Step 5: Start Fast Agent (Background)
```bash
# Start agent as background daemon
QALLOW_CIRQ=1 QALLOW_ENABLE_CUDA=ON python3 agentlightning_runner.py \
  --fast \
  --use-cuda \
  --daemon \
  --max-iterations=500 &

# Or simpler (variables already set):
python3 agentlightning_runner.py --fast --use-cuda --daemon &
```

Agent runs continuously, automatically improving code. You can:
- **Monitor:** `tail -f agent_daemon.log`
- **Check progress:** `grep "Iteration" agent_daemon.log | wc -l`
- **See commits:** `git log --oneline --author="AgentLightning Runner" | head -5`
- **Stop:** `pkill -f "agentlightning_runner.py"`

---

### Method 3: Run Individual Phases

If you want to run phases separately:

```bash
# Phase 11: Quantum Bridge (Cirq-Q)
./build/qallow phase 11 --ticks=64 --states=-1,0,1

# Phase 12: Elasticity
./build/qallow phase 12 --ticks=150 --eps=0.0001 --audit-tag=phase12_demo

# Phase 13: Harmonic Propagation (CPU-intensive)
./build/qallow phase 13 --nodes=256 --ticks=400 --k=0.002 --audit-tag=phase13_demo

# Phase 14: Coherence-Lattice Integration
./build/qallow phase 14 --ticks=600 --nodes=256 --target_fidelity=0.981

# Phase 15: Convergence & Lock-in
./build/qallow phase 15 --ticks=800 --eps=5e-6 --export=/tmp/final_results.json
```

---

## Monitoring & Debugging

### Monitor Unified Pipeline (Real-time)
```bash
# Watch logs as they're written
tail -f data/logs/unified_run.csv

# Or in separate terminal, watch all log files
watch -n 1 'ls -lh data/logs/'
```

### Monitor Fast Agent
```bash
# Real-time log monitoring
tail -f agent_daemon.log

# Count improvements made
grep "IMPROVED" agent_daemon.log | wc -l

# See iteration progress
grep "Iteration" agent_daemon.log | tail -5

# Check code quality improvements
git log --oneline --grep="improvement" | head -10
```

### Monitor GPU/CPU
```bash
# If using GPU
watch -n 0.5 nvidia-smi

# Monitor CPU usage
htop -p $(pgrep qallow)

# Monitor both
while true; do clear; nvidia-smi; echo "---"; top -b -n 1 -p $(pgrep qallow); sleep 1; done
```

### Check Results
```bash
# List output files
ls -lh data/logs/

# View CSV results
head -20 data/logs/unified_run.csv

# View JSON summary
jq . data/logs/unified_run.json | head -50

# Analyze performance
python3 -c "import pandas as pd; df = pd.read_csv('data/logs/unified_run.csv'); print(df.describe())"
```

---

## Troubleshooting

### Issue: "CUDA not found"
```bash
# Check GPU
nvidia-smi

# If not found, install CUDA:
# Ubuntu: sudo apt install nvidia-cuda-toolkit
# See: https://developer.nvidia.com/cuda-toolkit

# Or run on CPU instead:
export QALLOW_ENABLE_CUDA=OFF
./build/qallow run unified
```

### Issue: "Cirq not available"
```bash
# Check Cirq
python -c "import cirq; print(cirq.__version__)"

# If not found, install:
source .venv/bin/activate
pip install cirq qiskit

# Verify env var
echo $QALLOW_CIRQ  # Should be 1
```

### Issue: "Out of memory"
```bash
# Reduce parameters
./build/qallow phase 13 --nodes=128 --ticks=200  # Instead of 256/400

# Monitor GPU memory
watch -n 1 nvidia-smi

# Or use CPU instead
export QALLOW_ENABLE_CUDA=OFF
./build/qallow run unified
```

### Issue: "Build failed"
```bash
# Clean and rebuild
rm -rf build
./bootstrap.sh --cuda

# Or manual rebuild
cmake -S . -B build -DQALLOW_ENABLE_CUDA=ON
cmake --build build --parallel $(nproc)
```

### Issue: Agent crashes
```bash
# Check logs
tail -50 agent_daemon.log

# Check if process exists
ps aux | grep lightning_agent

# Restart
python3 agentlightning_runner.py --fast --daemon --max-iterations=500 &
```

---

## Performance Tips

### Make It Faster
```bash
# Use Release build (not Debug)
cmake -S . -B build \
  -DQALLOW_ENABLE_CUDA=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON

cmake --build build --parallel $(nproc)

# Run with fewer ticks (for testing)
./build/qallow phase 13 --nodes=64 --ticks=100   # Quick test: ~30s
./build/qallow phase 13 --nodes=256 --ticks=400  # Full run: ~2m
```

### Profile Performance
```bash
# Using NVIDIA Nsight Compute
ncu --set=detailed ./build/qallow phase 13 --ticks=100

# Using Linux perf
perf record -g ./build/qallow phase 13 --ticks=100
perf report
```

---

## Complete Example Workflow

Here's a complete real-world workflow:

```bash
#!/bin/bash
set -e

cd /home/xing/Qallow

echo "=== Qallow Full Build Workflow ==="

# 1. Bootstrap (first time only)
if [[ ! -d "build/qallow" ]]; then
    echo "1️⃣  Bootstrapping..."
    ./bootstrap.sh --cuda
fi

# 2. Activate environment
echo "2️⃣  Activating environment..."
source .venv/bin/activate

# 3. Configure
echo "3️⃣  Configuring..."
export QALLOW_ENABLE_CUDA=ON
export QALLOW_CIRQ=1
export QALLOW_PROFILE_SCOPE=1

# 4. Run phases
echo "4️⃣  Running all phases..."
time ./build/qallow run unified --integrate-phase11

# 5. Start agent
echo "5️⃣  Starting fast agent..."
python3 agentlightning_runner.py --fast --use-cuda --daemon --max-iterations=500 &
AGENT_PID=$!

echo "6️⃣  Done!"
echo "   Agent running (PID: $AGENT_PID)"
echo "   Monitor: tail -f agent_daemon.log"
echo "   Stop: pkill -f 'agentlightning_runner.py'"
```

Save as `workflow.sh` and run:
```bash
chmod +x workflow.sh
./workflow.sh
```

---

## Quick Command Reference

| Task | Command |
|------|---------|
| **Full auto setup** | `./run_full_build.sh` |
| **Bootstrap only** | `./bootstrap.sh --cuda` |
| **Activate env** | `source .venv/bin/activate` |
| **Run all phases** | `./build/qallow run unified --integrate-phase11` |
| **Run Phase 13** | `./build/qallow phase 13 --ticks=400` |
| **Run Phase 15** | `./build/qallow phase 15 --ticks=800` |
| **Start agent** | `python3 agentlightning_runner.py --fast --daemon &` |
| **Monitor agent** | `tail -f agent_daemon.log` |
| **Stop agent** | `pkill -f "agentlightning_runner.py"` |
| **Run tests** | `cd build && ctest --output-on-failure` |
| **Profile GPU** | `ncu --set=detailed ./build/qallow phase 13` |
| **Clean build** | `rm -rf build && ./bootstrap.sh --cuda` |

---

## Support

- **Docs:** See `/docs/` directory
- **Examples:** See `/examples/` directory
- **Config:** See `.github/instructions/` for guidelines
- **Issues:** Check GitHub issues or open new one

---

**Ready to go! 🚀**

For detailed documentation, see:
- `QUICK_START_FULL_BUILD.md` - Comprehensive guide
- `HOW_TO_RUN.md` - Detailed run instructions
- `bootstrap.sh` - Bootstrap documentation
- `README.md` - Project overview

Last updated: November 2025
