# PHASE IV QUICK START GUIDE

## Build and Run in 3 Steps

### 1. Build the Demo

```powershell
cd d:\Qallow
.\build_demo.bat
```

### 2. Run Phase IV Demo

```powershell
.\qallow_phase4.exe
```

**or with custom parameters:**

```powershell
.\qallow_phase4.exe 16 200
```

(16 pockets, 200 ticks each)

### 3. Check Output Files

- `multi_pocket_summary.txt` - Pocket simulation results
- `chronometric_summary.txt` - Temporal prediction analysis
- `pocket_0.csv` through `pocket_N.csv` - Per-pocket telemetry
- `qallow_multi_pocket.csv` - Master telemetry
- `chronometric_telemetry.csv` - Time bank observations

---

## What Does Phase IV Do?

### Multi-Pocket Scheduler

Runs **N parallel probabilistic worldlines** (like parallel universes):

- Each pocket has different parameters (learning rate, noise, stability)
- Executes independently (CPU or CUDA streams)
- Results are merged using confidence-weighted averaging
- Outliers are detected and filtered

**Think of it as**: Running 8-16 different "what if" scenarios simultaneously

### Chronometric Prediction Layer

Predicts **future system behavior** based on temporal patterns:

- Tracks delta-t (observed vs predicted time)
- Learns from 100 historical observations
- Forecasts 50 ticks into the future
- Detects temporal drift and anomalies

**Think of it as**: A crystal ball that learns from past timing patterns

---

## Key Metrics

| Metric | Good | Needs Tuning |
|--------|------|--------------|
| **Consensus** | > 0.80 | < 0.75 |
| **Coherence** | > 0.95 | < 0.90 |
| **Ethics Score** | > 2.5 / 3.0 | < 2.5 |
| **Prediction Confidence** | > 0.70 | < 0.50 |
| **Temporal Drift** | < 0.01 sec | > 0.05 sec |

---

## Example Output

```
╔══════════════════════════════════════════════════════════════╗
║                     EXECUTION SUMMARY                        ║
╠══════════════════════════════════════════════════════════════╣
║  Pockets Simulated:    8                                     ║
║  Ticks per Pocket:     100                                   ║
║  Total Simulation Time: 2.34 sec                             ║
║  Merged Coherence:     0.9512                                ║
║  Pocket Consensus:     0.8345                                ║
║  Ethics Score:         2.9876 / 3.0                          ║
║  Temporal Drift:       0.000234 sec                          ║
║  Prediction Conf:      0.7821                                ║
╠══════════════════════════════════════════════════════════════╣
║  Phase IV Status:      ✓ SUCCESS                             ║
╚══════════════════════════════════════════════════════════════╝
```

---

## Troubleshooting

### Build Fails

**Problem**: `NVCC not found`
**Solution**: Check CUDA path in `build_demo.bat` line 7

**Problem**: `vcvars64.bat not found`
**Solution**: Update Visual Studio path in `build_demo.bat` line 5

### Runtime Issues

**Problem**: Low consensus (< 0.75)
**Solution**: Increase number of pockets or adjust parameter ranges

**Problem**: High temporal drift
**Solution**: Check system clock, may indicate thread scheduling issues

**Problem**: Ethics score < 2.5
**Solution**: Review pocket parameter ranges (may be too aggressive)

---

## Advanced Usage

### Plot Telemetry Data

Using Python matplotlib:

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('qallow_multi_pocket.csv')
plt.plot(df['tick'], df['coherence'])
plt.xlabel('Tick')
plt.ylabel('Coherence')
plt.title('Multi-Pocket Coherence Over Time')
plt.show()
```

### Analyze Chronometric Patterns

```python
df = pd.read_csv('chronometric_telemetry.csv')
plt.scatter(df['tick'], df['delta_t'], c=df['confidence'], cmap='viridis')
plt.colorbar(label='Confidence')
plt.xlabel('Tick')
plt.ylabel('Delta-t (sec)')
plt.title('Temporal Drift Analysis')
plt.show()
```

---

## Next Steps

After successful Phase IV testing:

1. **Implement CUDA kernels** (`backend/cuda/multi_pocket.cu`) for 3-5x speedup
2. **Add telemetry dashboard** (ImGui or matplotlib-cpp) for real-time monitoring
3. **MPI integration** for distributed multi-node execution

---

## Files Reference

| File | Purpose |
|------|---------|
| `core/include/multi_pocket.h` | Multi-pocket scheduler API |
| `core/include/chronometric.h` | Chronometric prediction API |
| `backend/cpu/multi_pocket.c` | CPU implementation (400+ lines) |
| `backend/cpu/chronometric.c` | Chronometric implementation (450+ lines) |
| `phase4_demo.c` | Complete demonstration program |
| `build_demo.bat` | Build script for demo |
| `PHASE_IV_COMPLETE.md` | Full documentation |
| `PHASE_IV_ARCHITECTURE.md` | Architecture diagram |

---

**Status**: ✅ Phase IV Core Complete  
**Version**: 1.0  
**Ready**: YES
