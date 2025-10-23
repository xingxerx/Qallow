# 📊 Running Qallow Phases - Comprehensive Guide

**Duration**: 30 minutes | **Difficulty**: Intermediate | **Prerequisites**: 01_getting_started.md

## Overview of Phases

Qallow has 13+ research phases:

| Phase | Name | Purpose | Time |
|-------|------|---------|------|
| 13 | Harmonic Propagation | Coherence control | 1-5s |
| 14 | Coherence-Lattice | Deterministic coherence | 2-10s |
| 15 | Convergence & Lock-In | Stability constraints | 1-5s |

## Phase 13: Harmonic Propagation

### Purpose
Simulates harmonic propagation in quantum pockets with closed-loop ethics monitoring.

### Basic Usage

```bash
./build/qallow phase 13 --ticks=400 --log=data/logs/phase13.csv
```

### Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `--ticks` | 400 | 1-10000 | Number of simulation ticks |
| `--nodes` | 8 | 1-256 | Number of quantum pockets |
| `--log` | - | path | Output CSV file |
| `--k` | 0.001 | 0.0001-0.1 | Coupling constant |

### Examples

```bash
# Quick test (100 ticks)
./build/qallow phase 13 --ticks=100

# Full simulation (1000 ticks, 32 nodes)
./build/qallow phase 13 --ticks=1000 --nodes=32 --log=data/logs/phase13_full.csv

# High precision (10000 ticks)
./build/qallow phase 13 --ticks=10000 --log=data/logs/phase13_hires.csv
```

### Output Interpretation

```
[PHASE13] Harmonic propagation complete: pockets=8 ticks=400
[PHASE13] avg_coherence: 0.797500 → 1.000000
[PHASE13] phase_drift  : 0.100000 → 0.001968
```

- **avg_coherence**: Quantum coherence (0.0-1.0, higher is better)
- **phase_drift**: Phase error accumulation (lower is better)

### CSV Output Format

```csv
tick,coherence,fidelity,phase_drift,energy
0,0.797500,0.000000,0.100000,0.500000
1,0.798234,0.001000,0.099500,0.500100
...
400,1.000000,0.981000,0.001968,0.500000
```

## Phase 14: Coherence-Lattice Integration

### Purpose
Deterministic coherence control using closed-form alpha calculation.

### Basic Usage

```bash
./build/qallow phase 14 --ticks=600 --target_fidelity=0.981 --log=data/logs/phase14.csv
```

### Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `--ticks` | 600 | 1-10000 | Number of simulation ticks |
| `--nodes` | 256 | 1-1024 | Number of lattice nodes |
| `--target_fidelity` | 0.981 | 0.5-0.999 | Target fidelity threshold |
| `--log` | - | path | Output CSV file |

### Examples

```bash
# Standard run
./build/qallow phase 14 --ticks=600 --target_fidelity=0.981

# High fidelity target
./build/qallow phase 14 --ticks=1000 --target_fidelity=0.99

# Large lattice
./build/qallow phase 14 --ticks=800 --nodes=512 --target_fidelity=0.981
```

### Output Interpretation

```
[PHASE14] COMPLETE fidelity=0.981000 [OK]
```

- **[OK]**: Target fidelity reached
- **[WARN]**: Fidelity close to target
- **[FAIL]**: Target not reached

## Phase 15: Convergence & Lock-In

### Purpose
Convergence monitoring with stability constraints.

### Basic Usage

```bash
./build/qallow phase 15 --ticks=800 --eps=5e-6 --log=data/logs/phase15.csv
```

### Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `--ticks` | 800 | 1-10000 | Maximum simulation ticks |
| `--eps` | 5e-6 | 1e-8-1e-3 | Convergence epsilon |
| `--log` | - | path | Output CSV file |

### Examples

```bash
# Standard convergence
./build/qallow phase 15 --ticks=800 --eps=5e-6

# Tight convergence
./build/qallow phase 15 --ticks=1000 --eps=1e-8

# Loose convergence (faster)
./build/qallow phase 15 --ticks=500 --eps=1e-3
```

### Output Interpretation

```
[PHASE15] COMPLETE score=-0.012481 stability=0.000000
```

- **score**: Convergence metric (lower is better)
- **stability**: Stability measure (0.0 = stable)

## Running Multiple Phases

### Sequential Execution

```bash
#!/bin/bash
# Run all three phases in sequence

echo "Starting Phase 13..."
./build/qallow phase 13 --ticks=400 --log=data/logs/phase13.csv

echo "Starting Phase 14..."
./build/qallow phase 14 --ticks=600 --target_fidelity=0.981 --log=data/logs/phase14.csv

echo "Starting Phase 15..."
./build/qallow phase 15 --ticks=800 --eps=5e-6 --log=data/logs/phase15.csv

echo "All phases complete!"
```

### Parallel Execution (Advanced)

```bash
#!/bin/bash
# Run phases in parallel (requires separate processes)

./build/qallow phase 13 --ticks=400 --log=data/logs/phase13.csv &
PID1=$!

./build/qallow phase 14 --ticks=600 --log=data/logs/phase14.csv &
PID2=$!

./build/qallow phase 15 --ticks=800 --log=data/logs/phase15.csv &
PID3=$!

# Wait for all to complete
wait $PID1 $PID2 $PID3

echo "All phases complete!"
```

## Monitoring Phase Execution

### Real-time Dashboard

```bash
# Terminal 1: Start dashboard
cd /root/Qallow/ui
python3 dashboard.py

# Terminal 2: Run phase
cd /root/Qallow
./build/qallow phase 13 --ticks=400

# Terminal 3: Open browser
# http://localhost:5000
```

### Command-line Monitoring

```bash
# Watch CSV output in real-time
watch -n 1 'tail -5 data/logs/phase13.csv'

# Count lines (progress indicator)
watch -n 1 'wc -l data/logs/phase13.csv'
```

## Analyzing Results

### View CSV Data

```bash
# First 10 rows
head -10 data/logs/phase13.csv

# Last 10 rows
tail -10 data/logs/phase13.csv

# Statistics
awk -F',' 'NR>1 {print $2}' data/logs/phase13.csv | \
  awk '{sum+=$1; sumsq+=$1*$1; n++} END {
    print "Mean:", sum/n
    print "StdDev:", sqrt(sumsq/n - (sum/n)^2)
  }'
```

### Plot Results (Python)

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('data/logs/phase13.csv')

# Plot coherence
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.plot(df['tick'], df['coherence'])
plt.xlabel('Tick')
plt.ylabel('Coherence')
plt.title('Phase 13: Coherence Progression')

plt.subplot(1, 3, 2)
plt.plot(df['tick'], df['phase_drift'])
plt.xlabel('Tick')
plt.ylabel('Phase Drift')
plt.title('Phase 13: Phase Drift')

plt.subplot(1, 3, 3)
plt.plot(df['tick'], df['energy'])
plt.xlabel('Tick')
plt.ylabel('Energy')
plt.title('Phase 13: Energy')

plt.tight_layout()
plt.savefig('phase13_analysis.png')
plt.show()
```

## Troubleshooting

### Phase doesn't complete
```bash
# Check if process is running
ps aux | grep qallow

# Increase timeout
timeout 60 ./build/qallow phase 13 --ticks=10000
```

### Low coherence
```bash
# Try with more nodes
./build/qallow phase 13 --ticks=400 --nodes=32

# Try with different coupling
./build/qallow phase 13 --ticks=400 --k=0.01
```

### Target fidelity not reached
```bash
# Increase ticks
./build/qallow phase 14 --ticks=1000 --target_fidelity=0.981

# Lower target
./build/qallow phase 14 --ticks=600 --target_fidelity=0.95
```

## 📚 Next Steps

- **Quantum Algorithms**: `03_quantum_algorithms.md`
- **Interpreting Results**: `04_telemetry_analysis.md`
- **Advanced Configuration**: `docs/ARCHITECTURE_SPEC.md`

---

**Pro Tip**: Save your best runs with descriptive names:
```bash
./build/qallow phase 13 --ticks=1000 --log=data/logs/phase13_$(date +%Y%m%d_%H%M%S).csv
```

