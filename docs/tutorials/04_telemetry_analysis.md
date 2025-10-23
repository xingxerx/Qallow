# 📊 Telemetry Analysis & Interpretation Guide

**Duration**: 30 minutes | **Difficulty**: Intermediate | **Prerequisites**: 02_running_phases.md

## Overview

Qallow generates comprehensive telemetry data in multiple formats:
- **CSV Files**: Phase metrics (tick-by-tick)
- **JSON Files**: Aggregated metrics and results
- **Log Files**: Audit trails and system events
- **Dashboard**: Real-time visualization

## CSV Telemetry Format

### Phase 13 CSV

```csv
tick,coherence,fidelity,phase_drift,energy
0,0.797500,0.000000,0.100000,0.500000
1,0.798234,0.001000,0.099500,0.500100
...
400,1.000000,0.981000,0.001968,0.500000
```

### Column Definitions

| Column | Range | Meaning | Good Value |
|--------|-------|---------|------------|
| tick | 0-N | Simulation step | N/A |
| coherence | 0.0-1.0 | Quantum coherence | >0.99 |
| fidelity | 0.0-1.0 | State fidelity | >0.98 |
| phase_drift | 0.0-∞ | Phase error | <0.01 |
| energy | 0.0-1.0 | System energy | Stable |

## Reading CSV Files

### Command-line Tools

```bash
# View first 10 rows
head -10 data/logs/phase13.csv

# View last 10 rows
tail -10 data/logs/phase13.csv

# Count total rows
wc -l data/logs/phase13.csv

# View specific column
cut -d',' -f2 data/logs/phase13.csv | head -20

# Get statistics
awk -F',' 'NR>1 {print $2}' data/logs/phase13.csv | \
  awk '{sum+=$1; sumsq+=$1*$1; n++} END {
    print "Mean:", sum/n
    print "Min:", min
    print "Max:", max
    print "StdDev:", sqrt(sumsq/n - (sum/n)^2)
  }'
```

### Python Analysis

```python
import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('data/logs/phase13.csv')

# Basic statistics
print(df.describe())

# Coherence analysis
print(f"Initial coherence: {df['coherence'].iloc[0]:.6f}")
print(f"Final coherence: {df['coherence'].iloc[-1]:.6f}")
print(f"Improvement: {(df['coherence'].iloc[-1] - df['coherence'].iloc[0]):.6f}")

# Phase drift analysis
print(f"Initial phase drift: {df['phase_drift'].iloc[0]:.6f}")
print(f"Final phase drift: {df['phase_drift'].iloc[-1]:.6f}")
print(f"Reduction: {(df['phase_drift'].iloc[0] - df['phase_drift'].iloc[-1]):.6f}")

# Stability check
coherence_std = df['coherence'].std()
print(f"Coherence stability (std): {coherence_std:.6f}")
```

## JSON Metrics Format

### Phase 14 Metrics

```json
{
  "phase": "phase14",
  "ticks": 600,
  "nodes": 256,
  "target_fidelity": 0.981,
  "final_fidelity": 0.981,
  "convergence_tick": 579,
  "alpha_eff": 0.00161134,
  "status": "OK"
}
```

### Quantum Report

```json
{
  "timestamp": "2025-10-23T13:32:03.172465",
  "algorithms": {
    "hello_quantum": "PASSED",
    "bell_state": "PASSED",
    "deutsch": "PASSED"
  },
  "qaoa": {
    "energy": -4.334,
    "alpha_eff": 0.001390,
    "iterations": 50
  },
  "success_rate": 1.0
}
```

### Reading JSON

```python
import json

# Load metrics
with open('data/logs/phase14.json') as f:
    metrics = json.load(f)

print(f"Phase: {metrics['phase']}")
print(f"Fidelity: {metrics['final_fidelity']}")
print(f"Convergence: Tick {metrics['convergence_tick']}/{metrics['ticks']}")
print(f"Status: {metrics['status']}")
```

## Audit Logs

### Format

```
[2025-10-23 13:31:45] [ETHICS] Safety=0.99 Clarity=1.0 Human=1.0 Total=2.99
[2025-10-23 13:31:46] [PHASE13] Coherence improved: 0.7975 → 0.8442
[2025-10-23 13:31:47] [AUDIT] Phase 13 complete, all checks passed
```

### Reading Audit Logs

```bash
# View all audit entries
cat data/ethics_audit.log

# Filter by type
grep ETHICS data/ethics_audit.log

# Count entries
wc -l data/ethics_audit.log

# Recent entries
tail -20 data/ethics_audit.log

# Search for issues
grep -i "error\|warning\|fail" data/ethics_audit.log
```

## Visualization

### Plot with Matplotlib

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('data/logs/phase13.csv')

# Create figure
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Plot 1: Coherence
axes[0, 0].plot(df['tick'], df['coherence'], 'b-')
axes[0, 0].set_xlabel('Tick')
axes[0, 0].set_ylabel('Coherence')
axes[0, 0].set_title('Phase 13: Coherence Progression')
axes[0, 0].grid(True)

# Plot 2: Phase Drift
axes[0, 1].plot(df['tick'], df['phase_drift'], 'r-')
axes[0, 1].set_xlabel('Tick')
axes[0, 1].set_ylabel('Phase Drift')
axes[0, 1].set_title('Phase 13: Phase Drift')
axes[0, 1].grid(True)

# Plot 3: Fidelity
axes[1, 0].plot(df['tick'], df['fidelity'], 'g-')
axes[1, 0].set_xlabel('Tick')
axes[1, 0].set_ylabel('Fidelity')
axes[1, 0].set_title('Phase 13: Fidelity')
axes[1, 0].grid(True)

# Plot 4: Energy
axes[1, 1].plot(df['tick'], df['energy'], 'm-')
axes[1, 1].set_xlabel('Tick')
axes[1, 1].set_ylabel('Energy')
axes[1, 1].set_title('Phase 13: Energy')
axes[1, 1].grid(True)

plt.tight_layout()
plt.savefig('phase13_analysis.png', dpi=150)
plt.show()
```

### Dashboard Visualization

```bash
# Start dashboard
cd /root/Qallow/ui
python3 dashboard.py

# Open browser to http://localhost:5000
# Charts update every 500ms
```

## Comparing Runs

### Multi-Run Analysis

```python
import pandas as pd
import matplotlib.pyplot as plt
import glob

# Load all phase13 runs
files = glob.glob('data/logs/phase13_*.csv')
runs = {}

for f in files:
    name = f.split('/')[-1].replace('.csv', '')
    runs[name] = pd.read_csv(f)

# Compare coherence
fig, ax = plt.subplots()
for name, df in runs.items():
    ax.plot(df['tick'], df['coherence'], label=name)

ax.set_xlabel('Tick')
ax.set_ylabel('Coherence')
ax.set_title('Coherence Comparison Across Runs')
ax.legend()
ax.grid(True)
plt.savefig('coherence_comparison.png')
plt.show()
```

## Performance Metrics

### Calculate Key Metrics

```python
import pandas as pd

df = pd.read_csv('data/logs/phase13.csv')

# Convergence speed
convergence_tick = df[df['coherence'] > 0.99].iloc[0]['tick']
print(f"Convergence at tick: {convergence_tick}")

# Stability
final_coherence = df['coherence'].iloc[-100:].mean()
coherence_variance = df['coherence'].iloc[-100:].var()
print(f"Final coherence (avg): {final_coherence:.6f}")
print(f"Stability (variance): {coherence_variance:.6f}")

# Efficiency
total_ticks = len(df)
efficiency = convergence_tick / total_ticks
print(f"Efficiency: {efficiency:.2%}")

# Quality
final_fidelity = df['fidelity'].iloc[-1]
print(f"Final fidelity: {final_fidelity:.6f}")
```

## Troubleshooting with Telemetry

### Low Coherence

```python
# Check if coherence is improving
df = pd.read_csv('data/logs/phase13.csv')
improvement = df['coherence'].iloc[-1] - df['coherence'].iloc[0]
print(f"Coherence improvement: {improvement:.6f}")

if improvement < 0.1:
    print("WARNING: Low coherence improvement")
    print("Try: Increase ticks or nodes")
```

### High Phase Drift

```python
# Check phase drift trend
df = pd.read_csv('data/logs/phase13.csv')
drift_trend = df['phase_drift'].iloc[-1] - df['phase_drift'].iloc[0]
print(f"Phase drift change: {drift_trend:.6f}")

if drift_trend > 0:
    print("WARNING: Phase drift increasing")
    print("Try: Reduce coupling constant (--k)")
```

### Unstable Energy

```python
# Check energy stability
df = pd.read_csv('data/logs/phase13.csv')
energy_std = df['energy'].std()
print(f"Energy stability (std): {energy_std:.6f}")

if energy_std > 0.01:
    print("WARNING: Unstable energy")
    print("Try: Increase damping or reduce ticks")
```

## Exporting Results

### Export to CSV

```python
import pandas as pd

df = pd.read_csv('data/logs/phase13.csv')

# Export summary
summary = pd.DataFrame({
    'Metric': ['Initial Coherence', 'Final Coherence', 'Phase Drift Reduction'],
    'Value': [
        df['coherence'].iloc[0],
        df['coherence'].iloc[-1],
        df['phase_drift'].iloc[0] - df['phase_drift'].iloc[-1]
    ]
})

summary.to_csv('phase13_summary.csv', index=False)
```

### Export to JSON

```python
import json
import pandas as pd

df = pd.read_csv('data/logs/phase13.csv')

export = {
    'phase': 'phase13',
    'total_ticks': len(df),
    'initial_coherence': float(df['coherence'].iloc[0]),
    'final_coherence': float(df['coherence'].iloc[-1]),
    'phase_drift_reduction': float(df['phase_drift'].iloc[0] - df['phase_drift'].iloc[-1]),
    'final_fidelity': float(df['fidelity'].iloc[-1])
}

with open('phase13_export.json', 'w') as f:
    json.dump(export, f, indent=2)
```

## 📚 Next Steps

- **Advanced Analysis**: `docs/ARCHITECTURE_SPEC.md`
- **Ethics Monitoring**: `docs/ETHICS_CHARTER.md`
- **Performance Tuning**: `docs/SCALING_IMPLEMENTATION_SUMMARY.md`

---

**Pro Tip**: Automate telemetry analysis with cron jobs for continuous monitoring!

