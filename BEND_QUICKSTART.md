# Qallow Bend - Quick Reference Card

## 🚀 Quick Start Commands

### Run Simulations
```bash
# Phase 12: Elasticity Simulation
./scripts/run_bend_emulated.sh phase12 100 0.0001

# Phase 13: Harmonic Propagation
./scripts/run_bend_emulated.sh phase13 16 500 0.001
```

### View Results
```bash
# Show Phase 12 output
head -10 log_phase12.csv && tail -5 log_phase12.csv

# Show Phase 13 output
head -10 log_phase13.csv && tail -5 log_phase13.csv
```

## 🧠 AGI Self-Correction Features

### Automatic Retry
- Detects empty/failed results
- Purges error state
- Retries execution automatically
- Logs: `[AGI-ERROR]` and `[RECOVERY]`

### Numerical Auditing
- Validates all values in range [0.0, 1.0]
- Automatically clamps out-of-range values
- Logs: `[AUDIT] ⚠️  Value X.XXX out of range, clamping`

## 📊 Output Formats

### Phase 12 CSV
```csv
tick,coherence,entropy,decoherence
1,0.999860,0.000699,0.000009
...
```

### Phase 13 CSV
```csv
tick,avg_coherence,phase_drift
1,0.000636,1.000000
...
```

## 🔧 Parameter Guide

### Phase 12 (Elasticity)
```bash
./scripts/run_bend_emulated.sh phase12 <ticks> <epsilon>
```
- `ticks`: Number of simulation steps (default: 100)
- `epsilon`: Perturbation factor (default: 0.0001)

### Phase 13 (Harmonic)
```bash
./scripts/run_bend_emulated.sh phase13 <nodes> <ticks> <coupling>
```
- `nodes`: Number of oscillators (default: 8)
- `ticks`: Simulation steps (default: 400)
- `coupling`: Coupling strength (default: 0.001)

## 📂 File Locations

- **Main Entry**: `bend/main.bend`
- **Runner**: `scripts/run_bend_emulated.sh`
- **Docs**: `BEND_INTEGRATION_GUIDE.md`
- **Summary**: `BEND_CONVERSION_SUMMARY.md`
- **Logs**: `log_phase12.csv`, `log_phase13.csv`

## 🔍 Troubleshooting

### No Output?
Check stderr for AGI error messages:
```bash
./scripts/run_bend_emulated.sh phase12 100 0.0001 2>&1 | grep ERROR
```

### Values Out of Range?
Check audit logs:
```bash
./scripts/run_bend_emulated.sh phase13 16 500 0.01 2>&1 | grep AUDIT
```

### Compare with C Backend
```bash
# C version
./build/qallow_unified phase12 --ticks=100 --eps=0.0001 --log=c_out.csv

# Bend version
./scripts/run_bend_emulated.sh phase12 100 0.0001
cp log_phase12.csv bend_out.csv

# Compare
diff c_out.csv bend_out.csv
```

## ✅ Health Check

Run both modes to verify everything works:
```bash
# Phase 12
./scripts/run_bend_emulated.sh phase12 50 0.0001
test -f log_phase12.csv && echo "✓ Phase 12 OK"

# Phase 13
./scripts/run_bend_emulated.sh phase13 8 100 0.001
test -f log_phase13.csv && echo "✓ Phase 13 OK"
```

## 🎯 Integration with Existing Workflow

### With Governance
```bash
# Set ethics parameters
./build/qallow_unified govern --adjust H=1.0

# Run Bend simulation
./scripts/run_bend_emulated.sh phase12 1000 0.0001

# Analyze
cat log_phase12.csv
```

### Batch Processing
```bash
# Run multiple simulations
for eps in 0.0001 0.0005 0.001; do
    ./scripts/run_bend_emulated.sh phase12 100 $eps
    mv log_phase12.csv phase12_eps_${eps}.csv
done
```

## 📚 Documentation

- **Complete Guide**: `BEND_INTEGRATION_GUIDE.md`
- **Implementation Summary**: `BEND_CONVERSION_SUMMARY.md`
- **This Card**: `BEND_QUICKSTART.md`

---

## ⚙️ Advanced Automation

### Scheduled Phase Sweeps
```bash
# Nightly sweep across perturbation values
for eps in 0.00005 0.00010 0.00020; do
    ./scripts/run_bend_emulated.sh phase12 200 "$eps"
    mv log_phase12.csv "results/phase12_eps_${eps}.csv"
done
```
- Use `cron` or your CI runner to launch recurring sweeps.
- Store outputs under `results/` to keep `log_phase*.csv` clean for ad-hoc runs.

### Batch Harmonic Profiling
```bash
# Explore multiple topologies in parallel (requires GNU parallel)
parallel './scripts/run_bend_emulated.sh phase13 {1} 300 0.0008 && mv log_phase13.csv results/phase13_nodes_{1}.csv' ::: 8 16 32
```
- Watch stderr for `[AUDIT]` messages while the jobs run.
- Compare the resulting CSVs with `rg '1.000000' results/phase13_nodes_*.csv` to spot convergence differences.

## 📈 Analysis Toolkit

### Quick Metrics in Python
```bash
python3 - <<'PY'
import pandas as pd
df = pd.read_csv("results/phase12_eps_0.00010.csv")
print("Final coherence:", df["coherence"].iloc[-1])
print("Peak entropy:", df["entropy"].max())
PY
```
- Install dependencies inside your virtualenv (`pip install pandas`) if not already available.
- Embed this snippet into your notebook pipelines to keep Bend and C results comparable.

### Drift Heatmap (Phase 13)
```bash
python3 - <<'PY'
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("results/phase13_nodes_16.csv")
plt.plot(df["tick"], df["phase_drift"])
plt.title("Phase 13 Drift")
plt.xlabel("tick")
plt.ylabel("phase_drift")
plt.savefig("plots/phase13_drift.png", dpi=150)
PY
```
- Ensure `plots/` exists (`mkdir -p plots`) so the exporter succeeds.
- Combine with your CI artifact uploads to keep visual regressions in check.

## 🛡 Operational Hardening

### Fail-Fast Smoke Test
```bash
./scripts/run_bend_emulated.sh phase12 5 0.0001 >/tmp/phase12.log 2>&1 || exit 1
./scripts/run_bend_emulated.sh phase13 5 10 0.001 >/tmp/phase13.log 2>&1 || exit 1
rg '\[AGI-ERROR\]' /tmp/phase12.log /tmp/phase13.log && exit 1
echo "Smoke test clean"
```
- Stops early if AGI retries appear, catching regressions before longer jobs fire.
- Reuse inside Git hooks or CI workflows for pre-merge validation.

### Cross-Backend Drift Guard
```bash
./build/qallow_unified phase12 --ticks=100 --eps=0.0001 --log=c_baseline.csv
./scripts/run_bend_emulated.sh phase12 100 0.0001
python3 scripts/compare_csv.py c_baseline.csv log_phase12.csv --abs-tol=5e-5
```
- Wire into the nightly job to detect numerical drift between Bend and C paths.
- `scripts/compare_csv.py` exits non-zero on divergence, ideal for alerting.

---

**Status**: ✅ Production Ready  
**Last Updated**: October 19, 2025
