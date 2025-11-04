# Command Cheat Sheet - Full Build Execution

Copy-paste ready commands for running Qallow with CUDA + Cirq-Q + All Phases + Agent.

---

## 🚀 QUICKEST START (Copy & Paste)

### One Command - Everything
```bash
cd /home/xing/Qallow && ./run_full_build.sh
```

**That's it!** Will automatically:
- Setup environment ✓
- Build if needed ✓
- Run all phases 11-15 ✓
- Start fast agent ✓

---

## 🎯 THREE SEPARATE COMMANDS (If Running Manually)

### 1. Bootstrap (First Time Only)
```bash
cd /home/xing/Qallow && chmod +x bootstrap.sh && ./bootstrap.sh --cuda
```

### 2. Run All Phases
```bash
cd /home/xing/Qallow && source .venv/bin/activate && export QALLOW_ENABLE_CUDA=ON QALLOW_CIRQ=1 && ./build/qallow run unified --integrate-phase11
```

### 3. Start Fast Agent
```bash
cd /home/xing/Qallow && source .venv/bin/activate && export QALLOW_ENABLE_CUDA=ON QALLOW_CIRQ=1 && python3 lightning_agent_fast.py --fast --use-cuda --daemon &
```

---

## 📋 INDIVIDUAL PHASE RUNS

### Phase 11 Only (Quantum Bridge)
```bash
cd /home/xing/Qallow && source .venv/bin/activate && export QALLOW_CIRQ=1 && ./build/qallow phase 11 --ticks=64 --states=-1,0,1
```

### Phase 12 Only (Elasticity)
```bash
cd /home/xing/Qallow && source .venv/bin/activate && export QALLOW_ENABLE_CUDA=ON && ./build/qallow phase 12 --ticks=150 --eps=0.0001
```

### Phase 13 Only (Harmonic Propagation)
```bash
cd /home/xing/Qallow && source .venv/bin/activate && export QALLOW_ENABLE_CUDA=ON && ./build/qallow phase 13 --nodes=256 --ticks=400 --k=0.002
```

### Phase 14 Only (Coherence-Lattice)
```bash
cd /home/xing/Qallow && source .venv/bin/activate && export QALLOW_ENABLE_CUDA=ON && ./build/qallow phase 14 --ticks=600 --nodes=256 --target_fidelity=0.981
```

### Phase 15 Only (Convergence)
```bash
cd /home/xing/Qallow && source .venv/bin/activate && export QALLOW_ENABLE_CUDA=ON && ./build/qallow phase 15 --ticks=800 --eps=5e-6
```

### Phases 12-15 (No Quantum Bridge)
```bash
cd /home/xing/Qallow && source .venv/bin/activate && export QALLOW_ENABLE_CUDA=ON && ./build/qallow run unified
```

---

## 🤖 FAST AGENT COMMANDS

### Start Agent (Background)
```bash
cd /home/xing/Qallow && source .venv/bin/activate && export QALLOW_ENABLE_CUDA=ON QALLOW_CIRQ=1 && python3 lightning_agent_fast.py --fast --use-cuda --daemon --max-iterations=500 &
```

### Monitor Agent Real-Time
```bash
tail -f /home/xing/Qallow/agent_daemon.log
```

### Check Agent Progress
```bash
cd /home/xing/Qallow && grep "Iteration" agent_daemon.log | wc -l
```

### View Agent Commits
```bash
cd /home/xing/Qallow && git log --oneline --author="Lightning" | head -10
```

### Stop Agent
```bash
pkill -f "lightning_agent_fast.py"
```

---

## 🔧 BUILD & REBUILD COMMANDS

### Rebuild (If Code Changed)
```bash
cd /home/xing/Qallow && cmake --build build --parallel $(nproc)
```

### Clean Rebuild
```bash
cd /home/xing/Qallow && rm -rf build && ./bootstrap.sh --cuda
```

### Build with Optimization
```bash
cd /home/xing/Qallow && cmake -S . -B build -DQALLOW_ENABLE_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON && cmake --build build --parallel $(nproc)
```

---

## 📊 MONITORING COMMANDS

### Monitor Pipeline Output (Real-Time)
```bash
tail -f /home/xing/Qallow/data/logs/unified_run.csv
```

### Monitor GPU Usage
```bash
watch -n 0.5 nvidia-smi
```

### Monitor CPU Usage
```bash
watch -n 1 'top -b -n 1 -p $(pgrep qallow)'
```

### Check Results Files
```bash
cd /home/xing/Qallow && ls -lh data/logs/
```

### View CSV Results (First 20 lines)
```bash
cd /home/xing/Qallow && head -20 data/logs/unified_run.csv
```

### View JSON Summary
```bash
cd /home/xing/Qallow && jq . data/logs/unified_run.json | head -50
```

---

## 🧪 TEST COMMANDS

### Run All Tests
```bash
cd /home/xing/Qallow/build && ctest --output-on-failure
```

### Run Specific Test
```bash
cd /home/xing/Qallow/build && ctest -R "phase13" --output-on-failure
```

### Run CUDA Tests Only
```bash
cd /home/xing/Qallow/build && ctest -R "cuda" --output-on-failure
```

---

## 🎨 PROFILING COMMANDS

### Profile with Nsight Compute (GPU)
```bash
cd /home/xing/Qallow && ncu --set=detailed ./build/qallow phase 13 --ticks=100
```

### Profile with Linux Perf (CPU)
```bash
cd /home/xing/Qallow && perf record -g ./build/qallow phase 13 --ticks=100 && perf report
```

### Export Nsight Profiling Results
```bash
cd /home/xing/Qallow && ncu --export=/tmp/qallow_profile.ncu-rep ./build/qallow phase 13 --ticks=50
```

---

## 🚨 TROUBLESHOOTING COMMANDS

### Check CUDA Availability
```bash
nvidia-smi && echo "✓ CUDA available" || echo "✗ No CUDA"
```

### Check Cirq Installation
```bash
python3 -c "import cirq; print(f'✓ Cirq {cirq.__version__}')" || echo "✗ Cirq not installed"
```

### Check Environment Variables
```bash
echo "CUDA: $QALLOW_ENABLE_CUDA" && echo "Cirq: $QALLOW_CIRQ" && echo "Profile: $QALLOW_PROFILE_SCOPE"
```

### View Build Errors
```bash
cd /home/xing/Qallow && cmake --build build --verbose 2>&1 | tail -50
```

### Check Agent Errors
```bash
cd /home/xing/Qallow && tail -100 agent_daemon.log | grep -E "ERROR|Exception|Traceback"
```

---

## 🔄 WORKFLOW SCRIPTS

### Save as `start_full_build.sh` and run with `chmod +x && ./start_full_build.sh`

```bash
#!/bin/bash
set -e
cd /home/xing/Qallow

# Setup
source .venv/bin/activate 2>/dev/null || ./bootstrap.sh --cuda
source .venv/bin/activate

# Config
export QALLOW_ENABLE_CUDA=ON
export QALLOW_CIRQ=1

# Run
echo "Running phases..."
time ./build/qallow run unified --integrate-phase11

# Start agent
echo "Starting agent..."
python3 lightning_agent_fast.py --fast --use-cuda --daemon &

echo "✓ Done! Monitor with: tail -f agent_daemon.log"
```

---

## ⚡ FAST VERSION (CPU-Only, Faster for Testing)

### Quick Test - No CUDA, No Phase 11
```bash
cd /home/xing/Qallow && source .venv/bin/activate && ./build/qallow run unified
```

### Very Quick Test - Phase 13 with Small Workload
```bash
cd /home/xing/Qallow && source .venv/bin/activate && export QALLOW_ENABLE_CUDA=ON && ./build/qallow phase 13 --nodes=64 --ticks=100
```

---

## 📚 ENVIRONMENT SETUP VARIATIONS

### For CPU-Only (No GPU)
```bash
export QALLOW_ENABLE_CUDA=OFF
export QALLOW_CIRQ=1
```

### For GPU + Quantum
```bash
export QALLOW_ENABLE_CUDA=ON
export QALLOW_CIRQ=1
```

### For Debugging
```bash
export QALLOW_ENABLE_CUDA=ON
export QALLOW_CIRQ=1
export QALLOW_LOG_LEVEL=DEBUG
export QALLOW_PROFILE_SCOPE=1
```

### For Production
```bash
export QALLOW_ENABLE_CUDA=ON
export QALLOW_CIRQ=1
export QALLOW_LOG_LEVEL=INFO
export QALLOW_PROFILE_SCOPE=0
```

---

## 📝 NOTES

- Commands assume you're in `/home/xing/Qallow` or use full paths
- All commands activate venv if not already active
- GPU/CUDA detection is automatic; CPU fallback if unavailable
- Agent runs as background process (`&` at end)
- Results saved to `data/logs/` 
- Logs saved to `agent_daemon.log`

---

**Last Updated:** November 2025
**Qallow Version:** 0.1+
**Status:** Production Ready
