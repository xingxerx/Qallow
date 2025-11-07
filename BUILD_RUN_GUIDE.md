# Qallow Build & Run Guide

## Quick Start

### Build (Linux/WSL)
```bash
cd ~/Qallow/qallow_vm
bash build_unified.sh [cuda|cpu]
```

**Options:**
- `bash build_unified.sh` - Auto-detect CUDA (use if available)
- `bash build_unified.sh cuda` - Force CUDA build
- `bash build_unified.sh cpu` - CPU-only build

### Build from Windows PowerShell (WSL)
```powershell
wsl bash -c "cd ~/Qallow/qallow_vm && bash build_unified.sh"
```

## What Gets Built

| Binary | Size | Purpose |
|--------|------|---------|
| `build/qallow` | 3.8M | Main CLI interface |
| `build/qallow_unified_cuda` | 3.8M | CUDA-enabled unified executor |
| `build/qallow_ui` | 43K | SDL2-based graphical interface |
| `build/qallow_test_temporal_memory` | 30K | Temporal memory unit tests |
| `build/qallow_unit_cuda_parallel` | 18K | CUDA parallelization tests |
| `build/qallow_unit_ethics` | Various | Ethics module tests |
| `build/qallow_unit_dl_integration` | Various | Deep learning integration tests |

## Running Qallow

### Main Commands

```bash
cd ~/Qallow

# Show help
./build/qallow --help

# Show help for specific group
./build/qallow help run
./build/qallow help phase
./build/qallow help system

# Run unified workflow with integration
./build/qallow run unified --integrate

# Run specific phase
./build/qallow phase 13 --ticks=120

# Run phase with custom parameters
./build/qallow phase 13 --ticks=200 --integrate-phase13-k=0.003

# Run benchmark
./build/qallow run bench

# Run live execution
./build/qallow run live

# Build system
./build/qallow system build

# Verify project
./build/qallow system verify

# Clear build artifacts
./build/qallow system clear
```

### Run Unified Workflow (Recommended)

The unified workflow executes phases 12-15 sequentially with telemetry collection:

```bash
./build/qallow run unified --integrate
```

**Output includes:**
- Phase 12: Elasticity simulation
- Phase 13: Harmonic propagation  
- Phase 14: Lattice entanglement
- Phase 15: Convergence analysis

**Logs generated:**
- `data/logs/phase12.csv` - Phase 12 metrics
- `data/logs/phase13.csv` - Phase 13 metrics
- `data/logs/lattice_integrations.csv` - Lattice convergence data
- `data/logs/phase_summary.json` - Final phase metrics
- `data/logs/telemetry_stream.csv` - System telemetry
- `data/logs/qallow_bench.log` - Benchmark log

### Run Individual Phases

```bash
# Phase 13 (harmonic propagation)
./build/qallow phase 13

# Phase 13 with custom ticks
./build/qallow phase 13 --ticks=200

# Phase 12 (elasticity)
./build/qallow phase 12 --ticks=150

# Phase 14 (lattice entanglement)
./build/qallow phase 14

# Phase 15 (convergence)
./build/qallow phase 15
```

### Run Tests

```bash
# CUDA parallel processing tests
./build/qallow_unit_cuda_parallel

# Temporal memory tests
./build/qallow_test_temporal_memory

# Ethics module tests
./build/qallow_unit_ethics

# All unit tests
ctest --test-dir build --output-on-failure
```

### Run UI (if SDL2 available)

```bash
./build/qallow_ui
```

## Build Configuration

### Build Types

**CUDA Build** (recommended):
- Enables GPU acceleration via NVIDIA CUDA
- Requires: CUDA Toolkit, NVIDIA drivers, GPU hardware
- Provides: 10-100x speedup for quantum simulations

**CPU Build**:
- Pure CPU execution (no GPU required)
- Slower but portable across systems
- Useful for testing and development

### CMake Options

Build configuration is managed by `build_unified.sh`. To customize:

```bash
# Edit qallow_vm/build_unified.sh and modify these lines:
cmake "$PROJECT_ROOT" -DQALLOW_ENABLE_CUDA=ON -DCMAKE_BUILD_TYPE=Release
```

## Troubleshooting

### Build fails with "CMake not found"
```bash
sudo apt install cmake
```

### Build fails with "CUDA not found"
- Install CUDA Toolkit: https://developer.nvidia.com/cuda-downloads
- Or run CPU-only build: `bash build_unified.sh cpu`

### Build takes too long
- Uses parallel build with `nproc` cores (default 16)
- To limit: Edit `build_unified.sh` and change `NUM_CORES` value

### Binaries not found after build
- Verify you're in the correct directory: `cd ~/Qallow`
- Binaries are in `build/` not `qallow_vm/build/`

### Phase runner fails
```bash
# Verify build
./build/qallow system verify

# Clean and rebuild
./build/qallow system clear
cd qallow_vm
bash build_unified.sh
```

## Project Structure

```
~/Qallow/
├── qallow_vm/
│   ├── build_unified.sh     ← Use this to build (Linux/WSL)
│   └── build_unified.bat    ← Windows build script (legacy)
├── build/                   ← Build artifacts directory
│   ├── qallow               ← Main CLI
│   ├── qallow_unified_cuda  ← CUDA unified runner
│   └── [test binaries...]
├── src/                     ← Source code
│   ├── quantum/             ← Quantum module
│   ├── runtime/             ← Runtime infrastructure
│   └── ...
├── backend/
│   ├── cpu/                 ← CPU implementations
│   └── cuda/                ← CUDA implementations
├── data/logs/               ← Phase outputs and telemetry
├── CMakeLists.txt           ← Build configuration
└── README.md                ← Project documentation
```

## Development Workflow

1. **Make code changes** in `src/`, `backend/`, or `algorithms/`
2. **Rebuild**: `cd qallow_vm && bash build_unified.sh`
3. **Run tests**: `ctest --test-dir build --output-on-failure`
4. **Execute workflows**: `./build/qallow run unified --integrate`
5. **Check logs**: `cat data/logs/phase_summary.json`

## Performance Tips

- Use CUDA build for quantum simulations (10-100x faster)
- Increase phase ticks for more detailed metrics: `--ticks=500`
- Monitor telemetry: `tail -f data/logs/telemetry_stream.csv`
- Profile hot paths with: `./build/qallow_unit_cuda_parallel`

## Next Steps

- Read `README.md` for architecture overview
- Check `docs/ARCHITECTURE_SPEC.md` for detailed design
- Review `CONSTITUTION.md` for governance principles
- Explore `scripts/` for utility commands

---

**Last Updated**: November 6, 2025  
**Build Status**: ✅ All binaries compiled successfully with CUDA  
**Tests**: ✅ All unit tests passing
