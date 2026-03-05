# Qallow Quick Start Guide - Build & Run

**Last Updated**: 2025-11-11  
**Status**: ✅ All Build Issues Fixed

---

## Prerequisites

### System Dependencies
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y \
    cmake \
    build-essential \
    python3 \
    python3-pip \
    cargo \
    libsdl2-dev \
    libsdl2-ttf-dev \
    fonts-dejavu

# Optional: CUDA support
# Download from https://developer.nvidia.com/cuda-downloads
```

### Python Dependencies
```bash
pip3 install flask flask-cors cirq
```

---

## Quick Health Check

Before building, verify your system is ready:

```bash
./scripts/health_check.sh
```

Expected output:
```
✅ All critical checks passed!
```

---

## Building Qallow

### Option 1: Full Build (Recommended)
```bash
# Configure build
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Build all targets
cmake --build build

# Run tests
ctest --test-dir build
```

### Option 2: CPU-Only Build (No CUDA)
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DQALLOW_ENABLE_CUDA=OFF
cmake --build build
```

### Option 3: Specific Target
```bash
# Build just the core binary
cmake --build build --target qallow

# Build just the UI
cmake --build build --target qallow_ui

# Build unified runner
cmake --build build --target qallow_unified
```

---

## Running Qallow

### 1. C/SDL2 GUI (Lightweight, Native)
```bash
./build/qallow_ui
```
**Features**: Real-time telemetry, phase tracking, ethics visualization  
**Keyboard**: `[0]` = Phase 11 (Cirq), `[Q]` = Quit

### 2. Python Flask Dashboard (Web-based)
```bash
python3 ui/dashboard.py
# Open browser: http://localhost:5000
```
**Features**: Web interface, CSV export, audit logs  
**API Endpoints**: `/api/state`, `/api/telemetry`, `/api/phases`

### 3. Rust Native App (Modern, Type-Safe)
```bash
cd native_app
cargo build --release
./target/release/qallow_native
```
**Features**: Modern UI, cross-platform, type-safe  
**Status**: Recommended for new development

---

## Core Commands

### Run the Mind Engine
```bash
./build/qallow mind --steps=100
```

### Run Quantum Simulation (Phase 11)
```bash
python3 python/quantum/cirq_phase11.py --ticks=10 --simulator=ideal
```

### Run Benchmarks
```bash
./build/qallow bench --duration=60
```

### Run Tests
```bash
ctest --test-dir build --verbose
```

---

## Troubleshooting

### Issue: "Clock skew detected" warning
```bash
./scripts/fix_clock_skew.sh
```

### Issue: CUDA compilation errors
```bash
# Rebuild with CUDA disabled
cmake -B build -DQALLOW_ENABLE_CUDA=OFF
cmake --build build
```

### Issue: Font not found
```bash
# Install fonts
sudo apt-get install fonts-dejavu

# Or specify custom font path
export QALLOW_FONT_PATH=/path/to/font.ttf
```

### Issue: Python dashboard won't start
```bash
# Check syntax
python3 -m py_compile ui/dashboard.py

# Install missing dependencies
pip3 install flask flask-cors

# Run with debug output
QALLOW_DASHBOARD_DEBUG=1 python3 ui/dashboard.py
```

---

## Project Structure

```
Qallow/
├── build/                    # Build output (created by cmake)
├── interface/               # C/SDL2 UI
│   └── qallow_ui.c
├── ui/                      # Python Flask Dashboard
│   ├── dashboard.py
│   ├── templates/
│   └── requirements.txt
├── native_app/              # Rust FLTK Native App
│   ├── Cargo.toml
│   └── src/
├── backend/                 # Core algorithms
│   ├── cpu/                 # CPU implementations
│   ├── cuda/                # CUDA kernels
│   └── neuro/               # Neuromorphic backend
├── python/                  # Python modules
│   └── quantum/
│       └── cirq_phase11.py
├── scripts/                 # Build & utility scripts
│   ├── health_check.sh      # ✅ NEW: System verification
│   ├── fix_clock_skew.sh    # ✅ NEW: Clock sync tool
│   └── build.sh
├── CMakeLists.txt           # ✅ FIXED: CUDA flags
└── README.md
```

---

## Documentation

- **Build System**: See `BUILD_FIXES_SUMMARY.md`
- **UI Strategy**: See `UI_CONSOLIDATION_STRATEGY.md`
- **Python Dashboard**: See `ui/WEB_DASHBOARD_README.md`
- **Rust Native App**: See `native_app/README.md`

---

## Performance Tips

### For Development
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build
```

### For Production
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

### With CUDA Optimization
```bash
cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DQALLOW_ENABLE_CUDA=ON
cmake --build build
```

---

## Getting Help

1. **Check health**: `./scripts/health_check.sh`
2. **Review logs**: Check `build/CMakeFiles/CMakeOutput.log`
3. **Read docs**: See documentation files above
4. **Debug build**: `cmake --build build --verbose`

---

## Next Steps

1. ✅ Run health check: `./scripts/health_check.sh`
2. ✅ Build project: `cmake -B build && cmake --build build`
3. ✅ Run tests: `ctest --test-dir build`
4. ✅ Try a UI: `./build/qallow_ui` or `python3 ui/dashboard.py`
5. ✅ Explore: `./build/qallow --help`

---

## Support

For issues or questions:
1. Check `BUILD_FIXES_SUMMARY.md` for recent fixes
2. Run `./scripts/health_check.sh` to diagnose
3. Review build logs in `build/CMakeFiles/`
4. Check GitHub issues for similar problems

---

**Happy building! 🚀**

