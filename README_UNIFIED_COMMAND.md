# Qallow Unified Command Interface

## 🎯 One Command to Rule Them All

Welcome to the unified Qallow command interface! Everything you need to build, run, benchmark, and monitor the Qallow VM is now accessible through a single, intuitive command.

---

## 🚀 Quick Start

```bash
# Build both CPU and CUDA versions
./qallow build all

# Run the CPU version
./qallow run cpu

# Benchmark both versions
./qallow bench all

# Check system status
./qallow status

# View telemetry data
./qallow telemetry stream
```

---

## 📋 Command Reference

### Build
```bash
./qallow build              # Build CPU version (default)
./qallow build cpu          # Build CPU version
./qallow build cuda         # Build CUDA version
./qallow build all          # Build both versions
```

### Run
```bash
./qallow run                # Run CPU version (default)
./qallow run cpu            # Run CPU version
./qallow run cuda           # Run CUDA version
```

> ℹ️ `./qallow run` now triggers a fresh build for the selected target before launching, so the VM always executes the latest source changes. The CLI automatically restarts itself after the rebuild, so expect a brief pause before execution resumes.

### Benchmark
```bash
./qallow bench              # Benchmark CPU version (default)
./qallow bench cpu          # Benchmark CPU version
./qallow bench cuda         # Benchmark CUDA version
./qallow bench all          # Benchmark both versions
```

### Telemetry
```bash
./qallow telemetry          # Show stream data (default)
./qallow telemetry stream   # Show real-time tick data
./qallow telemetry bench    # Show benchmark history
./qallow telemetry adapt    # Show adaptive state
./qallow telemetry all      # Show all telemetry
```

### System
```bash
./qallow status             # Show system status
./qallow clean              # Clean build artifacts
./qallow help               # Show help
```

---

## 📊 Output Files

| File | Purpose |
|------|---------|
| `qallow_stream.csv` | Real-time telemetry data |
| `qallow_bench.log` | Benchmark history |
| `adapt_state.json` | Adaptive parameters |
| `build/qallow.exe` | CPU executable |
| `build/qallow_cuda.exe` | CUDA executable |

---

## 🔧 Implementation

### Files Created

1. **qallow_unified.ps1** - Main command script (PowerShell)
2. **qallow.bat** - Batch wrapper for Windows
3. **qallow.cmd** - CMD wrapper for Windows
4. **UNIFIED_COMMAND_GUIDE.md** - Complete documentation
5. **QALLOW_CHEATSHEET.md** - Quick reference
6. **UNIFIED_COMMAND_SUMMARY.md** - Implementation summary

### How It Works

```
./qallow <command> [target]
    ↓
qallow_unified.ps1
    ↓
Route to appropriate function
    ↓
Call build.ps1 / benchmark.ps1 / etc.
    ↓
Generate output files
```

---

## 💡 Usage Examples

### Development Workflow
```bash
# Build and run
./qallow build cpu
./qallow run cpu

# Check status
./qallow status

# View telemetry
./qallow telemetry stream
```

### Benchmarking Workflow
```bash
# Build both versions
./qallow build all

# Benchmark both
./qallow bench all

# View results
./qallow telemetry bench
```

### Full Test Workflow
```bash
# Clean, build, run, benchmark
./qallow clean
./qallow build all
./qallow run cpu
./qallow bench all
./qallow status
./qallow telemetry all
```

---

## 📈 System Status Example

```
========================================
  QALLOW SYSTEM STATUS
========================================

Build Status:
[OK] CPU build ready (219.5 KB)
[OK] CUDA build ready (221.5 KB)

Telemetry Files:
[OK] Stream data: 2 lines
[OK] Benchmark log: 16 lines
[PENDING] Adaptive state: Not configured

System Info:
[CPU] AMD Ryzen 7 7800X3D 8-Core Processor
[CORES] 8
[RAM] 31.2 GB
[GPU] NVIDIA GeForce RTX 5080
```

---

## 🎓 Documentation

- **UNIFIED_COMMAND_GUIDE.md** - Complete user guide with all commands
- **QALLOW_CHEATSHEET.md** - Quick reference card
- **UNIFIED_COMMAND_SUMMARY.md** - Implementation details

---

## ✅ Features

✅ **Single unified interface** - One command for everything
✅ **Smart defaults** - Sensible defaults for common operations
✅ **Comprehensive help** - Built-in help system
✅ **Status reporting** - Real-time system status
✅ **Telemetry management** - Easy access to all data
✅ **Batch wrappers** - Windows compatibility
✅ **Full documentation** - Complete guides and references

---

## 🔍 Troubleshooting

### Command Not Found
```bash
# Use full PowerShell path
./qallow_unified.ps1 build cpu

# Or use batch wrapper
./qallow.bat build cpu
```

### Build Fails
```bash
# Clean and rebuild
./qallow clean
./qallow build cpu
```

### CUDA Issues
```bash
# Check CUDA installation
nvidia-smi

# Verify CUDA Toolkit
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin\nvcc.exe" --version
```

### No Telemetry Data
```bash
# Run simulation first
./qallow run cpu

# Then view telemetry
./qallow telemetry stream
```

---

## 📁 File Structure

```
Qallow/
├── qallow_unified.ps1              ← Main command
├── qallow.bat                      ← Batch wrapper
├── qallow.cmd                      ← CMD wrapper
├── README_UNIFIED_COMMAND.md       ← This file
├── UNIFIED_COMMAND_GUIDE.md        ← Full guide
├── QALLOW_CHEATSHEET.md            ← Quick reference
├── UNIFIED_COMMAND_SUMMARY.md      ← Implementation
├── scripts/
│   ├── build.ps1
│   └── benchmark.ps1
├── build/
│   ├── qallow.exe
│   └── qallow_cuda.exe
├── qallow_stream.csv
├── qallow_bench.log
└── adapt_state.json
```

---

## 🎯 Next Steps

1. **Read the guide:** `UNIFIED_COMMAND_GUIDE.md`
2. **Check the cheatsheet:** `QALLOW_CHEATSHEET.md`
3. **Start using:** `./qallow build all`
4. **Monitor:** `./qallow status`
5. **Analyze:** `./qallow telemetry all`

---

## 📞 Support

For issues or questions:

1. Check `./qallow help`
2. Read `UNIFIED_COMMAND_GUIDE.md`
3. Review `QALLOW_CHEATSHEET.md`
4. Check troubleshooting section above

---

## 🟢 Status

### PRODUCTION READY

All commands tested and operational!

---

**Version:** 1.0.0
**Created:** 2025-10-18
**Status:** Complete and Tested
