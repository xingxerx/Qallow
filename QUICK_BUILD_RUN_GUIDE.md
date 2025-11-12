# 🚀 Qallow Quick Build & Run Guide

**Last Updated**: 2025-11-12  
**Environment**: GitHub Codespaces / Linux (CPU-only)  
**Status**: ✅ Fully Working

---

## ⚡ TL;DR - Run Now

```bash
cd /workspaces/Qallow
./build/qallow run vm
```

**That's it!** The system is already built and ready to run.

---

## 🔧 Build from Scratch (If Needed)

### **One-Command Build**
```bash
rm -rf build && mkdir build && cd build && \
cmake -DQALLOW_ENABLE_CUDA=OFF .. && \
cmake --build . --parallel $(nproc) && cd ..
```

### **Step-by-Step Build**
```bash
# 1. Clean old build
rm -rf build
mkdir build

# 2. Configure (CPU-only)
cd build
cmake -DQALLOW_ENABLE_CUDA=OFF ..

# 3. Build (uses all CPU cores)
cmake --build . --parallel $(nproc)

# 4. Go back to project root
cd ..
```

**Build Time**: ~30 seconds on 16 cores

---

## 🎮 Main Commands

### **Run the System**
```bash
./build/qallow run vm              # Main VM execution
./build/qallow run bench           # Benchmark mode
./build/qallow run unified         # Phase 12-15 pipeline
```

### **Run Specific Phases**
```bash
./build/qallow phase 11 --shots=1024
./build/qallow phase 12 --ticks=100
./build/qallow phase 13 --ticks=100
```

### **Run Tests**
```bash
cd build
ctest                              # Run all tests
ctest --output-on-failure          # Show test details
```

### **Get Help**
```bash
./build/qallow --help              # Main help
./build/qallow help run            # Help for 'run' commands
./build/qallow help phase          # Help for 'phase' commands
```

---

## 📊 What You'll See

When you run `./build/qallow run vm`, you'll see:

1. **Banner**: Qallow VM initialization
2. **System Info**: CPU mode, node count, max ticks
3. **Dashboard Updates**: Every 50 ticks showing:
   - Overlay Stability (Orbital/River/Mycelial)
   - Ethics Monitoring (Safety/Clarity/Human)
   - Reality Drift tracking
   - Coherence metrics
4. **Completion**: Telemetry summary

**Example Output**:
```
╔════════════════════════════════════════════════════════════╗
║           Qallow VM Dashboard - Tick 1                  ║
╚════════════════════════════════════════════════════════════╝

OVERLAY STABILITY:
Orbital      | ######################################## | 0.9992
River        | ######################################## | 0.9992
Mycelial     | ######################################## | 0.9992

ETHICS MONITORING:
Safety (S)   | ######################################## | 0.9992
Clarity (C)  | ######################################## | 1.0000
Human (H)    | ######################################## | 1.0000
```

---

## 🔍 Available Binaries

After building, these executables are in `./build/`:

| Binary | Purpose |
|--------|---------|
| `qallow` | Main entry point (recommended) |
| `qallow_unified_cpu` | CPU-optimized version |
| `qallow_throughput_bench` | Performance testing |
| `qallow_test_temporal_memory` | Memory system tests |
| `qallow_unit_ethics` | Ethics unit tests |
| `qallow_unit_dl_integration` | DL integration tests |

---

## 🐛 Troubleshooting

### **Build Fails with CMake Cache Error**
```bash
rm -rf build
mkdir build && cd build
cmake -DQALLOW_ENABLE_CUDA=OFF ..
cmake --build . --parallel $(nproc)
```

### **Missing json-c Library**
```bash
sudo apt-get update
sudo apt-get install -y libjson-c-dev
```

### **Python Quantum Framework Warnings**
```bash
source .venv/bin/activate  # If using venv
pip install --upgrade cycler matplotlib cirq cirq
```

### **Permission Denied**
```bash
chmod +x ./build/qallow
```

---

## ✅ Verification Checklist

After building, verify everything works:

```bash
# 1. Check binary exists
ls -lh ./build/qallow

# 2. Run help
./build/qallow --help

# 3. Run tests
cd build && ctest && cd ..

# 4. Run VM (short test)
./build/qallow run vm

# 5. Check version info
./build/qallow system verify
```

All should complete without errors.

---

## 📚 Key Files & Directories

```
/workspaces/Qallow/
├── build/                  # Build output (executables here)
│   ├── qallow             # Main binary ⭐
│   └── qallow_*           # Other binaries
├── core/                   # C/CUDA core implementation
├── interface/              # CLI interface
├── quantum_algorithms/     # Python quantum modules
├── CMakeLists.txt         # Build configuration
├── README.md              # Project overview
└── docs/                  # Documentation
```

---

## 🎯 Common Use Cases

### **Quick Test Run**
```bash
./build/qallow run vm
```

### **Performance Benchmark**
```bash
./build/qallow run bench
```

### **Run with Custom Options**
```bash
./build/qallow run vm --dashboard=100 --integrate phase12 phase13
```

### **Run Specific Phase**
```bash
./build/qallow phase 12 --ticks=200
```

### **Run All Tests**
```bash
cd build && ctest --verbose
```

---

## 🚀 Next Steps

1. ✅ **System is built and working**
2. 📖 Read `README.md` for project overview
3. 🔍 Explore `docs/` for detailed documentation
4. 🧪 Run tests: `cd build && ctest`
5. 🎮 Try different commands: `./build/qallow help`
6. 🐍 Fix Python deps (optional): `pip install --upgrade cycler matplotlib cirq`

---

## 📞 Getting Help

- **Command Help**: `./build/qallow help <command>`
- **Documentation**: See `docs/` directory
- **Issues**: Check `TROUBLESHOOTING_GUIDE.md`
- **Build Logs**: Check `build/` directory

---

**Status**: 🟢 **READY TO USE**

The Qallow system is fully built and operational in CPU mode!

