# Qallow Native C + CUDA Project

A high-performance native C + CUDA implementation of the Qallow quantum-inspired probabilistic system with GPU acceleration.

## Project Structure

```
qallow/
├── include/
│   └── qallow.h              # Core header with type definitions and function declarations
├── src/
│   ├── qallow_kernel.c       # Main VM kernel and orchestration logic
│   └── overlays.c            # Overlay (Orbital, River, Mycelial) implementations
├── emulation/
│   ├── photonic.cu           # GPU photonic simulation (CUDA)
│   └── quantum.cu            # GPU quantum optimizer (CUDA)
├── build.bat                 # Windows batch build script
├── build.ps1                 # Windows PowerShell build script
├── Makefile                  # Unix/Linux Makefile (reference)
└── qallow.exe                # Compiled executable
```

## Features

- **Native C Implementation**: No VM overhead, direct CPU execution
- **GPU Acceleration**: CUDA kernels for photonic simulation and quantum optimization
- **Three Overlays**: Orbital, River, and Mycelial state layers
- **Ethics Framework**: E = S + C + H (Sustainability + Compassion + Harmony)
- **Decoherence Tracking**: System stability monitoring
- **256 Nodes**: Configurable node count per overlay

## Building

### Windows (Recommended)

**Prerequisites:**
- NVIDIA CUDA Toolkit 13.0+
- Visual Studio 2022 Build Tools
- NVIDIA RTX 5080 or compatible GPU

**Build:**
```bash
.\build.bat clean
```

Or with PowerShell:
```powershell
powershell -ExecutionPolicy Bypass -File build.ps1 -Clean
```

### Linux/macOS

```bash
make clean
make
./qallow
```

## Running

```bash
.\qallow.exe
```

**Output Example:**
```
[Qallow] Native start. Nodes=256
[01] Orbital=0.9427 River=0.9998 Mycelial=1.0000 | Global=0.9808 | Deco=0.00040
[02] Orbital=0.9379 River=0.9997 Mycelial=1.0000 | Global=0.9792 | Deco=0.00040
...
[60] Orbital=0.9384 River=0.9955 Mycelial=0.9998 | Global=0.9779 | Deco=0.00038
[Qallow] Done.
```

## Architecture

### Core Components

1. **qallow_kernel.c**: Main VM orchestration
   - `qallow_init()`: Initialize VM state
   - `qallow_tick()`: Execute one simulation step
   - `qallow_global_stability()`: Compute system stability

2. **overlays.c**: Overlay state management
   - `overlay_init()`: Initialize overlay with deterministic seeding
   - `overlay_apply_nudge()`: Soften extremes toward midline
   - `overlay_propagate()`: Ripple effects between overlays
   - `overlay_stability()`: Compute overlay stability (inverse variance)

3. **photonic.cu**: GPU photonic simulation
   - Uses CUDA random number generation (Philox)
   - Generates probabilistic samples in [0,1)
   - Fills orbital overlay with photonic data

4. **quantum.cu**: GPU quantum optimizer
   - Energy-reduction shaping around 0.5
   - Gradient-based optimization
   - Operates on mycelial overlay

### Execution Flow (Per Tick)

1. GPU: Photonic simulation fills orbital overlay
2. CPU: Nudge orbital toward midline (soften extremes)
3. CPU: Ripple orbital → river (5% factor)
4. GPU: Quantum optimizer on mycelial overlay
5. CPU: Ripple river → mycelial (2% factor)
6. CPU: Ethics check & decoherence decay
7. CPU: Safeguard nudges if needed

## Configuration

Edit `include/qallow.h` to adjust:

```c
#define QALLOW_NODES 256  // Number of nodes per overlay
```

Edit `src/qallow_kernel.c` to adjust:

```c
for (int s=0; s<60; ++s)  // Number of simulation ticks
```

## Performance

- **Compilation Time**: ~10-15 seconds (CUDA + MSVC)
- **Execution Time**: ~1-2 seconds for 60 ticks
- **GPU Memory**: ~1 MB per tick
- **CPU Memory**: ~6 KB (256 nodes × 3 overlays × 8 bytes)

## Troubleshooting

### Build Errors

**"Cannot find compiler 'cl.exe'"**
- Install Visual Studio 2022 Build Tools
- Ensure CUDA Toolkit is installed

**"Cannot open include file"**
- Verify include paths in build script
- Check file permissions

### Runtime Errors

**"CUDA out of memory"**
- Reduce `QALLOW_NODES` in header
- Close other GPU applications

**"No CUDA devices found"**
- Verify GPU driver is installed
- Run `nvidia-smi` to check GPU status

## Future Enhancements

- ASCII dashboard with per-tick visualization
- Configurable ethics parameters
- Multi-GPU support
- Snapshot/checkpoint system
- Extended simulation metrics

## License

Part of the Qallow project ecosystem.

