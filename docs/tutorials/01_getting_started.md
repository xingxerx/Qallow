# 🚀 Getting Started with Qallow - Beginner's Guide

**Duration**: 15 minutes | **Difficulty**: Beginner | **Prerequisites**: None

## What is Qallow?

Qallow is a quantum-photonic AGI system that combines:
- **Quantum Computing**: Quantum algorithms and optimization
- **Ethics-First Design**: Safety, Clarity, and Human values
- **Real-time Monitoring**: Live dashboards and telemetry
- **Multi-Phase Pipeline**: 13+ research phases

## Step 1: Verify Installation

```bash
cd /root/Qallow

# Check if binaries exist
ls -lh build/qallow build/qallow_unified

# Expected output:
# -rwxr-xr-x 1 root root 872K Oct 23 13:31 build/qallow
# -rwxr-xr-x 1 root root 872K Oct 23 13:31 build/qallow_unified
```

## Step 2: Run Your First Phase

### Phase 13: Harmonic Propagation (Easiest)

```bash
# Create output directory
mkdir -p data/logs

# Run Phase 13
./build/qallow phase 13 --ticks=100 --log=data/logs/phase13_test.csv

# Expected output:
# [PHASE13] Harmonic propagation complete: pockets=8 ticks=100
# [PHASE13] avg_coherence: 0.797500 → 0.844235
# [PHASE13] phase_drift  : 0.100000 → 0.014155
```

### What Just Happened?

- **Phase 13** simulates harmonic propagation in quantum pockets
- **Coherence** improved from 0.7975 to 0.844 (good!)
- **Phase Drift** decreased (more stable)
- Results saved to `data/logs/phase13_test.csv`

## Step 3: View the Results

```bash
# View the CSV log
cat data/logs/phase13_test.csv | head -20

# Expected columns:
# tick,coherence,fidelity,phase_drift,energy
```

## Step 4: Run Phase 14 (Coherence-Lattice)

```bash
./build/qallow phase 14 --ticks=200 --target_fidelity=0.981 --log=data/logs/phase14_test.csv

# Expected output:
# [PHASE14] COMPLETE fidelity=0.981000 [OK]
```

## Step 5: Run Phase 15 (Convergence)

```bash
./build/qallow phase 15 --ticks=200 --eps=5e-6 --log=data/logs/phase15_test.csv

# Expected output:
# [PHASE15] COMPLETE score=-0.012481 stability=0.000000
```

## Step 6: Monitor with Web Dashboard

```bash
# Terminal 1: Start dashboard
cd /root/Qallow/ui
pip install -r requirements.txt
python3 dashboard.py

# Terminal 2: Open browser
# Navigate to: http://localhost:5000

# Terminal 3: Run a phase
cd /root/Qallow
./build/qallow phase 13 --ticks=400
```

## Step 7: Run Quantum Algorithms

```bash
# Activate Python environment
source venv/bin/activate

# Run quantum algorithms
python3 quantum_algorithms/algorithms/hello_quantum.py

# Expected output:
# ✅ Created 3 qubits
# ✅ Bell State (Entanglement)
# ✅ Deutsch Algorithm
```

## Step 8: Run Full ALG Framework

```bash
# Run all quantum algorithms + QAOA optimizer
python3 alg/main.py run --quick

# Expected output:
# [ALG] Running Hello Quantum...
# [ALG] Running Bell State...
# [ALG] Running Deutsch Algorithm...
# [QAOA] Optimization complete
# [ALG] ✓ Report written to: /var/qallow/quantum_report.json
```

## Step 9: View Quantum Report

```bash
# View the generated report
cat /var/qallow/quantum_report.md

# Expected output:
# # Quantum Algorithm Report
# - Total Algorithms: 3
# - Successful: 3/3
# - Success Rate: 100.0%
```

## Step 10: Run All Tests

```bash
# Run the test suite
ctest --test-dir build --output-on-failure

# Expected output:
# 100% tests passed, 0 tests failed out of 6
```

## 🎯 Next Steps

### For Researchers
- Read: `docs/ARCHITECTURE_SPEC.md`
- Explore: `docs/QUANTUM_WORKLOAD_GUIDE.md`
- Try: `docs/tutorials/02_running_phases.md`

### For Developers
- Read: `CONTRIBUTING.md`
- Explore: `src/` and `backend/` directories
- Try: Building custom phases

### For DevOps
- Read: `docs/KUBERNETES_DEPLOYMENT_GUIDE.md`
- Explore: `k8s/` and `deploy/` directories
- Try: Docker deployment

## 📊 Common Commands

```bash
# View help
./build/qallow --help

# Run phase with custom parameters
./build/qallow phase 13 --ticks=1000 --nodes=32

# Run all phases
./build/qallow phase 13 && ./build/qallow phase 14 && ./build/qallow phase 15

# View logs
tail -f data/logs/phase13.csv

# Monitor dashboard
python3 ui/dashboard.py
```

## ✅ Verification Checklist

- [ ] Binaries exist and are executable
- [ ] Phase 13 runs successfully
- [ ] Phase 14 reaches target fidelity
- [ ] Phase 15 converges
- [ ] Quantum algorithms pass
- [ ] Dashboard loads in browser
- [ ] Tests pass (6/6)

## 🆘 Troubleshooting

### "Command not found: qallow"
```bash
# Make sure you're in the right directory
cd /root/Qallow
./build/qallow --help
```

### "Permission denied"
```bash
# Make binaries executable
chmod +x build/qallow build/qallow_unified
```

### "No such file or directory: data/logs"
```bash
# Create the directory
mkdir -p data/logs
```

## 📚 Learn More

- **Quick Reference**: `QUICK_RUN_GUIDE.md`
- **System Architecture**: `QALLOW_SYSTEM_ARCHITECTURE.md`
- **Phase Details**: `docs/PHASE13_CLOSED_LOOP.md`
- **Ethics Framework**: `docs/ETHICS_CHARTER.md`

---

**Congratulations!** You've successfully run Qallow. 🎉

Next: Read `02_running_phases.md` for advanced phase configuration.

