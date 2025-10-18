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

**Status**: ✅ Production Ready  
**Last Updated**: October 18, 2025
