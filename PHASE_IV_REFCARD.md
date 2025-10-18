# PHASE IV - QUICK REFERENCE CARD

## 🎯 One-Line Summary

Multi-Pocket Scheduler runs 8-16 parallel worldlines, Chronometric Layer predicts temporal drift.

---

## ⚡ Quick Commands

```powershell
# Build
.\build_demo.bat

# Run (default: 8 pockets, 100 ticks)
.\qallow_phase4.exe

# Run custom (16 pockets, 200 ticks)
.\qallow_phase4.exe 16 200

# Clean
.\build_demo.bat clean
```

---

## 📊 Key Metrics

| Metric | Good | Bad |
|--------|------|-----|
| Consensus | > 0.80 | < 0.75 |
| Coherence | > 0.95 | < 0.90 |
| Ethics | > 2.5 | < 2.5 |
| Drift | < 0.01 | > 0.05 |

---

## 📁 Output Files

- `multi_pocket_summary.txt` - Pocket results
- `chronometric_summary.txt` - Time bank stats
- `pocket_[0-N].csv` - Per-pocket telemetry
- `qallow_multi_pocket.csv` - Master telemetry
- `chronometric_telemetry.csv` - Temporal data

---

## 🔧 Key Functions

### Multi-Pocket
- `multi_pocket_init()` - Initialize scheduler
- `multi_pocket_execute_all()` - Run simulation
- `multi_pocket_merge()` - Combine results
- `multi_pocket_calculate_consensus()` - Agreement metric

### Chronometric
- `chrono_bank_init()` - Initialize time bank
- `chrono_bank_add_observation()` - Record delta-t
- `chronometric_generate_forecast()` - Predict 50 ticks
- `chronometric_detect_anomaly()` - Flag outliers

---

## 🎓 Concepts

**Multi-Pocket**: Parallel universe simulation - run N scenarios, merge best outcome

**Time Bank**: Learn from past timing errors to predict future drift

**Consensus**: How much parallel worldlines agree (0.0-1.0)

**Delta-t**: Observed time - Predicted time (in seconds)

---

## 🚀 Status

✅ Core: 80% Complete  
🔵 CUDA: Phase IV.1  
🔵 Dashboard: Phase IV.1  
🔵 MPI: Phase IV.1

---

## 📚 Docs

- `PHASE_IV_QUICKSTART.md` - Getting started
- `PHASE_IV_COMPLETE.md` - Full documentation
- `PHASE_IV_ARCHITECTURE.md` - Architecture diagram
- `PHASE_IV_ACTIVATION_COMPLETE.md` - Summary
