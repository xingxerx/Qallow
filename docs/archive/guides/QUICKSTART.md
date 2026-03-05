# Qallow GPU Project - Quick Start Guide

## ✅ Project Setup Complete!

Your native C + CUDA Qallow project is ready to use.

## 📁 What Was Added

```
D:\Qallow/
├── include/qallow.h              ← Core type definitions
├── src/
│   ├── qallow_kernel.c           ← Main VM logic
│   └── overlays.c                ← Overlay implementations
├── emulation/
│   ├── photonic.cu               ← GPU photonic simulation
│   └── quantum.cu                ← GPU quantum optimizer
├── build.bat                     ← Windows build script
├── build.ps1                     ← PowerShell build script
├── qallow.exe                    ← Compiled executable (ready to run!)
├── README_QALLOW_GPU.md          ← Full documentation
└── QUICKSTART.md                 ← This file
```

## 🚀 Quick Start

### Run the Executable (Already Built!)

```bash
.\qallow.exe
```

You should see 60 ticks of simulation output showing:
- Orbital, River, Mycelial stability values
- Global stability average
- Decoherence decay

### Rebuild After Changes

```bash
.\build.bat clean
```

Or with PowerShell:
```powershell
powershell -ExecutionPolicy Bypass -File build.ps1 -Clean
```

## 🔧 Configuration

### Change Number of Nodes

Edit `include/qallow.h`:
```c
#define QALLOW_NODES 256  // Change to 512, 1024, etc.
```

### Change Number of Ticks

Edit `src/qallow_kernel.c` in the `main()` function:
```c
for (int s=0; s<60; ++s)  // Change 60 to desired tick count
```

### Adjust GPU Architecture

Edit `build.bat` (line 35):
```batch
"%NVCC%" -O2 -arch=sm_89 ...  # sm_89 for RTX 5080
```

Common architectures:
- `sm_89`: RTX 5080, RTX 4090, RTX 4080
- `sm_86`: RTX 3090, RTX 3080
- `sm_75`: RTX 2080, RTX 2070
- `sm_70`: V100, Titan V

## 📊 Understanding the Output

```
[Qallow] Native start. Nodes=256
[01] Orbital=0.9427 River=0.9998 Mycelial=1.0000 | Global=0.9808 | Deco=0.00040
     ^^^^^^^^         ^^^^^^^^         ^^^^^^^^^     ^^^^^^^^     ^^^^^^^^
     Orbital          River            Mycelial      Average       Decoherence
     stability        stability        stability     stability     (decays over time)
```

**Stability Range**: 0.0 (chaotic) to 1.0 (perfect order)

**Decoherence**: Starts at 0.0004, decays by 0.1% per tick
- When decoherence > 0.0015 → system applies safeguard nudges

## 🎯 Key Features

✅ **GPU-Accelerated**: Photonic simulation and quantum optimization run on RTX 5080
✅ **256 Nodes**: Per overlay (configurable)
✅ **Three Overlays**: Orbital, River, Mycelial with ripple propagation
✅ **Ethics Framework**: E = S + C + H (Sustainability + Compassion + Harmony)
✅ **Deterministic**: Reproducible results with seed control
✅ **Fast**: ~1-2 seconds for 60 ticks

## 🐛 Troubleshooting

### Build Fails with "Cannot find compiler"
- Install Visual Studio 2022 Build Tools
- Ensure CUDA Toolkit 13.0+ is installed

### Executable Won't Run
- Check GPU driver: `nvidia-smi`
- Verify CUDA installation: `nvcc --version`

### Want to Modify Code?
1. Edit `.c` or `.cu` files
2. Run `.\build.bat clean` to rebuild
3. Run `.\qallow.exe` to test

## 📚 Next Steps

1. **Explore the Code**:
   - `include/qallow.h` - Type definitions
   - `src/qallow_kernel.c` - Main orchestration
   - `src/overlays.c` - Overlay math
   - `emulation/photonic.cu` - GPU photonic simulation
   - `emulation/quantum.cu` - GPU quantum optimizer

2. **Modify Parameters**:
   - Adjust node count for different problem sizes
   - Change tick count for longer simulations
   - Tune ethics parameters (S, C, H values)

3. **Add Features**:
   - ASCII dashboard visualization
   - Snapshot/checkpoint system
   - Multi-GPU support
   - Extended metrics

## 📖 Full Documentation

See `README_QALLOW_GPU.md` for complete documentation including:
- Architecture details
- Performance metrics
- Advanced configuration
- Future enhancements

## ✨ You're All Set!

Your Qallow GPU project is ready to use. Run `.\qallow.exe` to start!

Questions? Check the README or examine the source code.

