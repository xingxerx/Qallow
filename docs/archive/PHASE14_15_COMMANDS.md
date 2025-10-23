# Phase 14 & 15: Command Reference

## Build

```bash
cd /root/Qallow
cmake -S . -B build && cmake --build build --parallel
```

## Help

```bash
# Main help
./build/qallow help

# Phase group help
./build/qallow help phase

# Specific phase help
./build/qallow phase help
```

## Phase-14: Coherence-Lattice Integration

### Simplest (Deterministic, No Tuning)
```bash
./build/qallow phase 14 --ticks=600 --nodes=256 --target_fidelity=0.981
```

### With Export
```bash
./build/qallow phase 14 --ticks=600 --nodes=256 --target_fidelity=0.981 \
  --export=/tmp/phase14.json
```

### With QAOA Tuner
```bash
./build/qallow phase 14 --ticks=600 --nodes=256 --target_fidelity=0.981 \
  --tune_qaoa --qaoa_n=16 --qaoa_p=2 --export=/tmp/phase14.json
```

### With CUDA J-Coupling
```bash
./build/qallow phase 14 --ticks=600 --nodes=256 --target_fidelity=0.981 \
  --jcsv=graph.csv --gain_base=0.001 --gain_span=0.009
```

### With Explicit Alpha
```bash
./build/qallow phase 14 --ticks=600 --nodes=256 --target_fidelity=0.981 \
  --alpha=0.00161134
```

### With JSON Gain File
```bash
./build/qallow phase 14 --ticks=600 --nodes=256 --target_fidelity=0.981 \
  --gain_json=/tmp/gain.json
```

## Phase-15: Convergence & Lock-In

### Simplest
```bash
./build/qallow phase 15 --ticks=800 --eps=5e-6
```

### With Export
```bash
./build/qallow phase 15 --ticks=800 --eps=5e-6 --export=/tmp/phase15.json
```

### With Custom Tolerance
```bash
./build/qallow phase 15 --ticks=1000 --eps=1e-7 --export=/tmp/phase15.json
```

## Full Workflow

### Sequential Execution
```bash
# Phase-14: deterministic fidelity
./build/qallow phase 14 --ticks=600 --nodes=256 --target_fidelity=0.981 \
  --export=/tmp/p14.json

# Phase-15: convergence
./build/qallow phase 15 --ticks=800 --eps=5e-6 --export=/tmp/p15.json

# Check results
echo "=== Phase-14 Metrics ===" && cat /tmp/p14.json
echo "=== Phase-15 Metrics ===" && cat /tmp/p15.json
```

### With QAOA Tuning
```bash
# Phase-14 with QAOA
./build/qallow phase 14 --ticks=600 --nodes=256 --target_fidelity=0.981 \
  --tune_qaoa --qaoa_n=16 --qaoa_p=2 --export=/tmp/p14_qaoa.json

# Phase-15 convergence
./build/qallow phase 15 --ticks=800 --eps=5e-6 --export=/tmp/p15.json
```

## Verification

### Check Phase-14 Output
```bash
./build/qallow phase 14 --ticks=600 --nodes=256 --target_fidelity=0.981 2>&1 | grep "COMPLETE"
# Expected: [PHASE14] COMPLETE fidelity=0.981000 [OK]
```

### Check Phase-15 Output
```bash
./build/qallow phase 15 --ticks=800 --eps=5e-6 2>&1 | grep "COMPLETE"
# Expected: [PHASE15] COMPLETE score=... stability=0.000000
```

### Check Exported Metrics
```bash
# Phase-14 metrics
./build/qallow phase 14 --ticks=600 --nodes=256 --target_fidelity=0.981 \
  --export=/tmp/p14.json && cat /tmp/p14.json

# Phase-15 metrics
./build/qallow phase 15 --ticks=800 --eps=5e-6 --export=/tmp/p15.json && cat /tmp/p15.json
```

## Phase-14 Options Summary

| Option | Default | Purpose |
|--------|---------|---------|
| `--ticks=N` | 500 | Number of iterations |
| `--nodes=N` | 256 | Lattice nodes |
| `--target_fidelity=F` | 0.981 | Success threshold |
| `--alpha=A` | - | Explicit alpha override |
| `--jcsv=FILE` | - | CUDA J-coupling CSV |
| `--gain_base=B` | 0.001 | Base gain |
| `--gain_span=S` | 0.009 | Gain span |
| `--gain_json=FILE` | - | JSON gain file |
| `--tune_qaoa` | - | Enable QAOA tuner |
| `--qaoa_n=N` | 16 | QAOA problem size |
| `--qaoa_p=P` | 2 | QAOA depth |
| `--export=FILE` | - | Export JSON metrics |

## Phase-15 Options Summary

| Option | Default | Purpose |
|--------|---------|---------|
| `--ticks=N` | 400 | Max iterations |
| `--eps=E` | 1e-5 | Convergence tolerance |
| `--export=FILE` | - | Export JSON metrics |

## Gain Priority (Phase-14)

When multiple gain sources are available:

1. **QAOA tuner** (if `--tune_qaoa`)
2. **JSON file** (if `--gain_json`)
3. **CUDA J-coupling** (if `--jcsv`)
4. **CLI alpha** (if `--alpha`)
5. **Closed-form α** (fallback)

## Expected Output

### Phase-14 Success
```
[PHASE14] Coherence-lattice integration
[PHASE14] nodes=256 ticks=600 target_fidelity=0.981
[PHASE14] alpha closed-form = 0.00161134
[PHASE14][0000] fidelity=0.950081
[PHASE14][0050] fidelity=0.953948
...
[PHASE14] COMPLETE fidelity=0.981000 [OK]
```

### Phase-15 Success
```
[PHASE15] Starting convergence & lock-in
[PHASE15] ticks=800 eps=5e-06
[PHASE15][0000] score=0.740000 f=0.845000 s=0.560000
...
[PHASE15][0140] converged score=-0.012481
[PHASE15] COMPLETE score=-0.012481 stability=0.000000
```

## Troubleshooting

### Phase-14 doesn't hit target
- Check `--ticks` is large enough
- Verify `--target_fidelity` is reasonable (0.0-1.0)
- Try explicit `--alpha` if closed-form fails

### Phase-15 doesn't converge
- Increase `--ticks`
- Relax `--eps` (e.g., 1e-5 instead of 5e-6)
- Check previous phase output

### QAOA tuner fails
- Ensure qiskit is installed
- Check Python path
- Falls back to closed-form automatically

## Quick Copy-Paste

```bash
# Build
cmake -S . -B build && cmake --build build --parallel

# Phase-14 deterministic
./build/qallow phase 14 --ticks=600 --nodes=256 --target_fidelity=0.981

# Phase-15 convergence
./build/qallow phase 15 --ticks=800 --eps=5e-6

# Full workflow with export
./build/qallow phase 14 --ticks=600 --nodes=256 --target_fidelity=0.981 --export=/tmp/p14.json
./build/qallow phase 15 --ticks=800 --eps=5e-6 --export=/tmp/p15.json
```

