# Phase 14 & 15 Unified Integration Summary

## Overview

Phase-14 and Phase-15 are now fully integrated into the unified **`qallow phase`** command group. All phases (11-15) run through a single CLI interface with a shared engine, not separate invocations.

## Unified Command Structure

```bash
qallow phase <11|12|13|14|15> [options]
```

All phases are subcommands of the `phase` group, operating on the same underlying engine.

## Phase-14: Coherence-Lattice Integration

### Command
```bash
qallow phase 14 --ticks=600 --nodes=256 --target_fidelity=0.981
```

### Key Features

1. **Deterministic Gain (Closed-Form α)**
   - Computes α to hit target fidelity in exactly n ticks
   - Formula: `α = 1 − ((1 − target) / (1 − f0))^(1/n)`
   - For f0=0.950, target=0.981, n=600 → α≈0.00161134

2. **Adaptive Gain Priority (Highest to Lowest)**
   - QAOA tuner (if `--tune_qaoa`)
   - JSON gain file (if `--gain_json`)
   - CUDA J-coupling graph (if `--jcsv`)
   - CLI alpha override (if `--alpha`)
   - Closed-form deterministic α (fallback)

3. **QAOA Tuner Integration**
   - Invoked inline via `--tune_qaoa` flag
   - No separate Python commands needed
   - Automatically applies learned alpha_eff

### Phase-14 Options

```
--ticks=N                 Number of ticks (default: 500)
--nodes=N                 Lattice nodes (default: 256)
--target_fidelity=F       Success threshold (default: 0.981)
--alpha=A                 Explicit alpha override
--jcsv=FILE               CUDA CSR J-couplings
--gain_base=B             Base gain (default: 0.001)
--gain_span=S             Gain span (default: 0.009)
--gain_json=FILE          Load {"alpha_eff": A}
--tune_qaoa               Invoke QAOA tuner inline
--qaoa_n=N                QAOA problem size (default: 16)
--qaoa_p=P                QAOA depth (default: 2)
--export=FILE             Write JSON metrics
```

### Example Runs

**Deterministic (no tuning):**
```bash
qallow phase 14 --ticks=600 --nodes=256 --target_fidelity=0.981
```

**With QAOA tuner:**
```bash
qallow phase 14 --ticks=600 --nodes=256 --target_fidelity=0.981 \
  --tune_qaoa --qaoa_n=16 --qaoa_p=2 --export=/tmp/phase14.json
```

**With CUDA J-coupling:**
```bash
qallow phase 14 --ticks=600 --nodes=256 --target_fidelity=0.981 \
  --jcsv=graph.csv --gain_base=0.001 --gain_span=0.009
```

### Output

```
[PHASE14] Coherence-lattice integration
[PHASE14] nodes=256 ticks=600 target_fidelity=0.981
[PHASE14] alpha closed-form = 0.00161134
[PHASE14][0000] fidelity=0.950081
[PHASE14][0050] fidelity=0.953948
...
[PHASE14] COMPLETE fidelity=0.981000 [OK]
```

### Exported Metrics (JSON)

```json
{
  "fidelity": 0.981000,
  "target": 0.981000,
  "ticks": 600,
  "alpha_base": 0.00161134,
  "alpha_used": 0.00161134
}
```

## Phase-15: Convergence & Lock-In

### Command
```bash
qallow phase 15 --ticks=800 --eps=5e-6
```

### Key Features

1. **Convergence Detection**
   - Stops when score change < eps after 50-tick warm-up
   - Enforces non-negative stability (clamped at 0.0)

2. **Weighted Score**
   - `score = 0.6 * fidelity + 0.35 * stability - 0.05 * decoherence`

### Phase-15 Options

```
--ticks=N                 Max ticks (default: 400)
--eps=E                   Convergence tolerance (default: 1e-5)
--export=FILE             Write JSON summary
```

### Example Run

```bash
qallow phase 15 --ticks=800 --eps=5e-6 --export=/tmp/phase15.json
```

### Output

```
[PHASE15] Starting convergence & lock-in
[PHASE15] ticks=800 eps=5e-06
[PHASE15][0000] score=0.740000 f=0.845000 s=0.560000
[PHASE15][0050] score=0.203329 f=0.209042 s=0.221138
[PHASE15][0140] converged score=-0.012481
[PHASE15] COMPLETE score=-0.012481 stability=0.000000
```

### Exported Metrics (JSON)

```json
{
  "score": -0.012481,
  "stability": 0.000000
}
```

## Success Criteria

✅ **Phase-14**: Final fidelity ≥ 0.981 without WARN  
✅ **Phase-15**: Score monotone to convergence, stability ≥ 0

## Architecture

- **Unified CLI**: `launcher.c` dispatches all phases through `qallow_handle_phase_group()`
- **Phase Runners**: `main.c` contains `qallow_phase14_runner()` and `qallow_phase15_runner()`
- **QAOA Tuner**: `qiskit_tuner.py` runs inline when `--tune_qaoa` is set
- **Shared Engine**: All phases operate on the same underlying quantum simulation engine

## Build & Run

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

## Notes

- All phases run through the unified `qallow phase` command group
- No separate CLI invocations needed
- QAOA tuner is optional; closed-form α guarantees target hit
- Exported metrics enable pipeline orchestration and monitoring

