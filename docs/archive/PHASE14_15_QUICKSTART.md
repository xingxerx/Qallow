# Phase 14 & 15 Quick Start

## One-Line Summary

All phases run through **`qallow phase <N>`** — a unified command group with shared engine. No separate CLI invocations.

## Build

```bash
cd /root/Qallow
cmake -S . -B build && cmake --build build --parallel
```

## Phase-14: Hit 0.981 Fidelity Deterministically

### Simplest Command
```bash
./build/qallow phase 14 --ticks=600 --nodes=256 --target_fidelity=0.981
```

**Output:**
```
[PHASE14] alpha closed-form = 0.00161134
[PHASE14][0000] fidelity=0.950081
...
[PHASE14] COMPLETE fidelity=0.981000 [OK]
```

### With Export
```bash
./build/qallow phase 14 --ticks=600 --nodes=256 --target_fidelity=0.981 \
  --export=/tmp/phase14.json
```

### With QAOA Tuner (Optional)
```bash
./build/qallow phase 14 --ticks=600 --nodes=256 --target_fidelity=0.981 \
  --tune_qaoa --qaoa_n=16 --qaoa_p=2 --export=/tmp/phase14.json
```

### With CUDA J-Coupling (Optional)
```bash
./build/qallow phase 14 --ticks=600 --nodes=256 --target_fidelity=0.981 \
  --jcsv=graph.csv --gain_base=0.001 --gain_span=0.009
```

## Phase-15: Convergence & Lock-In

### Simplest Command
```bash
./build/qallow phase 15 --ticks=800 --eps=5e-6
```

**Output:**
```
[PHASE15][0000] score=0.740000 f=0.845000 s=0.560000
...
[PHASE15][0140] converged score=-0.012481
[PHASE15] COMPLETE score=-0.012481 stability=0.000000
```

### With Export
```bash
./build/qallow phase 15 --ticks=800 --eps=5e-6 --export=/tmp/phase15.json
```

## Full Workflow

```bash
# Phase-14: deterministic fidelity
./build/qallow phase 14 --ticks=600 --nodes=256 --target_fidelity=0.981 \
  --export=/tmp/p14.json

# Phase-15: convergence
./build/qallow phase 15 --ticks=800 --eps=5e-6 --export=/tmp/p15.json

# Check results
cat /tmp/p14.json
cat /tmp/p15.json
```

## Help

```bash
./build/qallow help phase
```

## Success Criteria

✅ Phase-14: `fidelity >= 0.981` with `[OK]` status  
✅ Phase-15: `stability >= 0.0` and converged

## Key Flags

| Flag | Phase | Purpose |
|------|-------|---------|
| `--ticks=N` | 14, 15 | Number of iterations |
| `--target_fidelity=F` | 14 | Success threshold (default: 0.981) |
| `--eps=E` | 15 | Convergence tolerance (default: 1e-5) |
| `--tune_qaoa` | 14 | Enable QAOA tuner |
| `--export=FILE` | 14, 15 | Write JSON metrics |

## Gain Priority (Phase-14)

1. QAOA tuner (if `--tune_qaoa`)
2. JSON file (if `--gain_json`)
3. CUDA J-coupling (if `--jcsv`)
4. CLI alpha (if `--alpha`)
5. Closed-form α (fallback, guarantees target)

## Notes

- **Unified**: All phases (11-15) run through `qallow phase`
- **Shared Engine**: Single underlying quantum simulation
- **Deterministic**: Phase-14 closed-form α hits target in exactly n ticks
- **Optional Tuning**: QAOA tuner is optional; closed-form works standalone
- **Metrics**: Export JSON for pipeline orchestration

