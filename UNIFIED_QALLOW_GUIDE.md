# Qallow Unified Command System

## 🎯 Overview

Phase 13 has been successfully integrated into the unified Qallow command system! You now have **one command** to rule them all - build, run, and test everything.

## 📁 File Structure

```
/root/Qallow/
├── qallow                    # ✨ NEW: Unified command interface (shell script)
├── build/
│   ├── qallow_unified        # Main unified binary (all phases)
│   └── qallow_phase13        # Standalone Phase 13 test
├── src/
│   └── qallow_phase13.c      # Your 8-thread harmonic test
└── backend/cpu/
    └── phase13_harmonic.c    # Full Phase 13 implementation
```

## 🚀 Quick Start

### Display Help
```bash
./qallow help
```

### Check System Status
```bash
./qallow status
```

### Build Everything
```bash
./qallow build-all       # Builds unified + standalone Phase 13
```

## 📋 Available Commands

### Build Commands
```bash
./qallow build           # Build unified system (build/qallow_unified)
./qallow build-phase13   # Build standalone Phase 13 test (8 threads)
./qallow build-all       # Build everything
```

### Run Commands

#### Unified VM
```bash
./qallow run             # Run full unified VM
```

#### Phase 12 (Elasticity)
```bash
./qallow phase12 --ticks=1000 --eps=0.0001 --log=phase12.csv
```

#### Phase 13 (Harmonic Propagation) - Via Unified System
```bash
./qallow phase13 --nodes=8 --ticks=100 --k=0.001 --log=phase13.csv
./qallow phase13 --nodes=16 --ticks=500 --k=0.001 --log=phase13_large.csv
```

**Direct execution alternative:**
```bash
./build/qallow_unified phase13 --nodes=8 --ticks=100 --k=0.001 --log=test.csv
```

#### Phase 13 Standalone Test (8 Threads)
```bash
./qallow phase13-test    # Quick 8-thread harmonic test
```

**Or compile and run manually:**
```bash
gcc -O3 -march=native -flto -pthread src/qallow_phase13.c -o qallow
./qallow
```

### Management Commands
```bash
./qallow bench           # Run benchmarks
./qallow govern          # Governance audit
./qallow verify          # System verification
./qallow live            # Live interface
./qallow status          # Show system status
```

### Bend Integration
```bash
./qallow bend12 --ticks=100 --eps=0.0001 --log=bend12.csv
./qallow bend13 --nodes=16 --ticks=500 --k=0.001 --log=bend13.csv
```

## 🔧 Phase 13 Details

### What is Phase 13?
Phase 13 implements **harmonic propagation** across multiple "pocket dimensions" with:
- **Multi-pocket coherence**: Synchronization across isolated quantum states
- **Phase alignment**: Coupled oscillators with damping and energy flow
- **Drift minimization**: Automatic convergence to stable equilibrium
- **CSV logging**: Full telemetry output for analysis

### Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `--nodes=N` | Number of pocket dimensions | 8 | 2-32 |
| `--ticks=T` | Simulation iterations | 400 | 1-∞ |
| `--k=K` | Coupling strength | 0.001 | 0.0001-0.01 |
| `--log=PATH` | CSV output file | - | any path |

### Output Metrics

The Phase 13 simulation tracks:
- **avg_coherence**: Mean coherence across all pockets (target: → 1.0)
- **phase_drift**: Phase deviation from mean (target: → 0.0)
- **phase_energy**: Total system energy

### Example Output
```
[PHASE13] Harmonic propagation
[PHASE13] nodes=8 ticks=100 k=0.001000
[PHASE13] log=test_phase13.csv
[PHASE13] Harmonic propagation complete: pockets=8 ticks=100 k=0.001000
[PHASE13] avg_coherence: 0.797500 → 0.891608
[PHASE13] phase_drift  : 0.100000 → 0.002347
```

### CSV Output Format
```csv
tick,avg_coherence,phase_drift,phase_energy
1,0.797643,0.098234,0.024563
2,0.801245,0.092156,0.023189
3,0.806789,0.085432,0.021567
...
```

## 🎨 Visual Flow

```
User Command
     ↓
./qallow [command] [options]
     ↓
     ├──→ build          → ./build.sh → build/qallow_unified
     ├──→ build-phase13  → gcc ... → build/qallow_phase13
     ├──→ phase13        → build/qallow_unified phase13 [args]
     ├──→ phase13-test   → build/qallow_phase13 (8 threads)
     └──→ [other]        → build/qallow_unified [command]
```

## 🧪 Testing Examples

### Quick Test (100 ticks)
```bash
./qallow phase13 --nodes=8 --ticks=100 --k=0.001 --log=quick.csv
```

### Convergence Test (1000 ticks, high coupling)
```bash
./qallow phase13 --nodes=16 --ticks=1000 --k=0.005 --log=converge.csv
```

### Weak Coupling Test (slow drift correction)
```bash
./qallow phase13 --nodes=8 --ticks=500 --k=0.0001 --log=weak.csv
```

### Standalone 8-Thread Test
```bash
./qallow phase13-test
# Output:
# Thread 0 active
# Thread 1 active
# Thread 2 active
# Thread 3 active
# Thread 4 active
# Thread 5 active
# Thread 6 active
# Thread 7 active
# Qallow Phase 13 operational
```

## 🔍 Troubleshooting

### Command not found: ./qallow
```bash
chmod +x qallow
```

### Binary not built
The `./qallow` script will automatically build missing binaries:
```bash
./qallow phase13  # Auto-builds if needed
```

### Manual build
```bash
./build.sh                              # Build unified system
gcc -O3 -march=native -flto -pthread \
    src/qallow_phase13.c \
    -o build/qallow_phase13             # Build standalone
```

### Check what's built
```bash
./qallow status
```

## 📊 CSV Data Analysis

After running Phase 13, analyze the CSV output:

```bash
# View first 10 rows
head test_phase13.csv

# Count total ticks
wc -l test_phase13.csv

# Extract final coherence value
tail -1 test_phase13.csv | cut -d',' -f2

# Plot with gnuplot
gnuplot -e "set terminal png; set output 'coherence.png'; \
    plot 'test_phase13.csv' using 1:2 with lines title 'Coherence'"
```

## 🏗️ Architecture Integration

Phase 13 is fully integrated with:
- ✅ **Unified build system** (`build.sh`)
- ✅ **Main launcher** (`interface/launcher.c`)
- ✅ **Phase 13 runner** (`interface/main.c::qallow_phase13_runner`)
- ✅ **Backend implementation** (`backend/cpu/phase13_harmonic.c`)
- ✅ **Header interface** (`core/include/qallow_phase13.h`)
- ✅ **Unified command** (`./qallow` script)

## 🎯 Summary

| What | Command | Output |
|------|---------|--------|
| **Build Unified** | `./qallow build` | `build/qallow_unified` |
| **Build Standalone** | `./qallow build-phase13` | `build/qallow_phase13` |
| **Run Phase 13 (Full)** | `./qallow phase13 --nodes=8 --ticks=100` | Harmonic simulation |
| **Run Phase 13 (Quick)** | `./qallow phase13-test` | 8-thread test |
| **Check Status** | `./qallow status` | System overview |

## 🚀 Next Steps

1. **Run benchmarks** to measure performance:
   ```bash
   ./qallow phase13 --nodes=8 --ticks=1000 --k=0.001 --log=benchmark.csv
   ```

2. **Analyze convergence** with different coupling strengths:
   ```bash
   for k in 0.0001 0.001 0.01; do
       ./qallow phase13 --nodes=16 --ticks=500 --k=$k --log=k_${k}.csv
   done
   ```

3. **Compare with Bend** implementation:
   ```bash
   ./qallow phase13 --nodes=8 --ticks=100 --log=c_impl.csv
   ./qallow bend13 --nodes=8 --ticks=100 --log=bend_impl.csv
   ```

4. **Scale testing**:
   ```bash
   ./qallow phase13 --nodes=32 --ticks=2000 --k=0.001 --log=scale_test.csv
   ```

---

**Phase 13 is now operational! 🎉**

All functionality is unified under the `./qallow` command for maximum convenience and consistency.
