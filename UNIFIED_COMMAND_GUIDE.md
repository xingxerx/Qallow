# Qallow Unified Command - Complete Guide

## Overview

The `./qallow` command provides a single, unified interface for all Qallow VM operations. No more juggling multiple scripts - everything is accessible through one command.

---

## Installation

The unified command is already set up! You can use it in three ways:

### Option 1: PowerShell (Recommended)
```bash
./qallow_unified.ps1 <command> [target]
```

### Option 2: Batch Wrapper (Windows)
```bash
./qallow.bat <command> [target]
qallow <command> [target]
```

### Option 3: CMD Wrapper (Windows)
```bash
./qallow.cmd <command> [target]
```

---

## Commands

### 1. Build

Build the Qallow system in CPU, CUDA, or both modes.

```bash
./qallow build cpu              # Build CPU version only
./qallow build cuda             # Build CUDA version only
./qallow build all              # Build both CPU and CUDA
./qallow build                  # Default: CPU
```

**Output:**
- `build/qallow.exe` - CPU executable
- `build/qallow_cuda.exe` - CUDA executable

---

### 2. Run

Execute a simulation with the specified backend.

```bash
./qallow run cpu                # Run CPU version
./qallow run cuda               # Run CUDA version
./qallow run                    # Default: CPU
```

> ℹ️ The run command always triggers a build for the selected backend first, ensuring the binary reflects the latest source code before execution. After the build completes, the CLI respawns itself automatically so the newly compiled binary handles the session.

**Output:**
- `qallow_stream.csv` - Real-time telemetry data
- `qallow_bench.log` - Benchmark entry
- `adapt_state.json` - Adaptive state (if pocket dimensions spawned)

---

### 3. Benchmark

Run performance benchmarks (3 runs by default).

```bash
./qallow bench cpu              # Benchmark CPU version
./qallow bench cuda             # Benchmark CUDA version
./qallow bench all              # Benchmark both versions
./qallow bench                  # Default: CPU
```

**Output:**
- Timing statistics (min, max, average, std dev)
- System information (CPU, cores, RAM, GPU)
- Benchmark log entry in `qallow_bench.log`

---

### 4. Telemetry

View real-time telemetry data and logs.

```bash
./qallow telemetry stream       # View real-time stream data
./qallow telemetry bench        # View benchmark history
./qallow telemetry adapt        # View adaptive state
./qallow telemetry all          # View all telemetry
./qallow telemetry              # Default: stream
```

**Files:**
- `qallow_stream.csv` - Real-time tick data
- `qallow_bench.log` - Benchmark history
- `adapt_state.json` - Adaptive parameters

---

### 5. Status

Display comprehensive system status.

```bash
./qallow status
```

**Shows:**
- Build status (CPU/CUDA executables)
- Telemetry files (stream, bench, adapt)
- System information (CPU, cores, RAM, GPU)

---

### 6. Clean

Remove all build artifacts.

```bash
./qallow clean
```

**Removes:**
- `build/` directory
- All compiled executables
- Object files

---

### 7. Help

Display help information.

```bash
./qallow help                   # Show main help
./qallow help build             # Show build help
./qallow help run               # Show run help
./qallow help bench             # Show benchmark help
./qallow help telemetry         # Show telemetry help
./qallow help status            # Show status help
./qallow help clean             # Show clean help
```

---

## Quick Start Examples

### Complete Workflow

```bash
# 1. Build both versions
./qallow build all

# 2. Run CPU version
./qallow run cpu

# 3. View telemetry
./qallow telemetry stream

# 4. Benchmark both versions
./qallow bench all

# 5. Check system status
./qallow status

# 6. View all telemetry
./qallow telemetry all
```

### Development Workflow

```bash
# Build and run CPU version
./qallow build cpu
./qallow run cpu

# Check status
./qallow status

# View telemetry
./qallow telemetry stream

# Clean and rebuild
./qallow clean
./qallow build cpu
```

### Benchmarking Workflow

```bash
# Build both versions
./qallow build all

# Benchmark CPU
./qallow bench cpu

# Benchmark CUDA
./qallow bench cuda

# View benchmark history
./qallow telemetry bench

# Check status
./qallow status
```

---

## Output Files

### qallow_stream.csv
Real-time telemetry data streamed during execution.

```csv
tick,orbital,river,mycelial,global,deco,mode
0,0.9984,0.9982,0.9984,0.9992,0.00001,CPU
1,0.9985,0.9983,0.9985,0.9993,0.00001,CPU
```

### qallow_bench.log
Benchmark history with timestamps and metrics.

```
timestamp,compile_ms,run_ms,deco,global,mode
2025-10-18 07:56:49,0.0,1.00,0.00001,0.9992,CPU
2025-10-18 08:12:34,0.0,0.95,0.00001,0.9993,CUDA
```

### adapt_state.json
Adaptive reinforcement parameters.

```json
{
  "target_ms": 50.0,
  "last_run_ms": 42.8,
  "threads": 4,
  "learning_rate": 0.0034,
  "human_score": 0.8
}
```

---

## System Status Output

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

## Troubleshooting

### Command Not Found

If `./qallow` doesn't work, try:

```bash
# Use full PowerShell path
./qallow_unified.ps1 build cpu

# Or use batch wrapper
./qallow.bat build cpu

# Or set execution policy
powershell -ExecutionPolicy Bypass -File qallow_unified.ps1 build cpu
```

### Build Fails

```bash
# Clean and rebuild
./qallow clean
./qallow build cpu

# Check Visual Studio
"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
```

### CUDA Build Fails

```bash
# Check CUDA Toolkit
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin\nvcc.exe" --version

# Verify GPU
nvidia-smi
```

### No Telemetry Data

```bash
# Run simulation first
./qallow run cpu

# Then view telemetry
./qallow telemetry stream
```

---

## Advanced Usage

### Custom Build Modes

```bash
# Build with clean
./qallow build cpu              # Rebuilds from scratch

# Build CUDA only
./qallow build cuda
```

### Telemetry Analysis

```bash
# View last 10 lines of stream
./qallow telemetry stream

# View all benchmark history
./qallow telemetry bench

# View adaptive state
./qallow telemetry adapt

# View everything
./qallow telemetry all
```

### Performance Monitoring

```bash
# Run and benchmark
./qallow run cpu
./qallow bench cpu

# Check status
./qallow status

# View telemetry
./qallow telemetry all
```

---

## File Structure

```
Qallow/
├── qallow_unified.ps1         # Main unified command script
├── qallow.bat                 # Batch wrapper
├── qallow.cmd                 # CMD wrapper
├── scripts/
│   ├── build.ps1              # Build script
│   └── benchmark.ps1          # Benchmark script
├── build/
│   ├── qallow.exe             # CPU executable
│   └── qallow_cuda.exe        # CUDA executable
├── qallow_stream.csv          # Real-time telemetry
├── qallow_bench.log           # Benchmark history
└── adapt_state.json           # Adaptive state
```

---

## Environment Variables

The unified command respects these environment variables:

- `CUDA_PATH` - CUDA Toolkit installation path
- `INCLUDE_DIR` - Include directory for compilation
- `BUILD_DIR` - Build output directory

---

## Performance Tips

1. **Use `./qallow bench all`** to compare CPU vs CUDA performance
2. **Check `./qallow status`** to verify builds are ready
3. **View `./qallow telemetry stream`** for real-time metrics
4. **Use `./qallow clean`** before major rebuilds

---

## Integration with CI/CD

```bash
# In CI/CD pipeline
./qallow build all
./qallow bench all
./qallow telemetry all
```

---

## Status Indicators

| Indicator | Meaning |
|-----------|---------|
| `[OK]` | Operation successful |
| `[PENDING]` | Data not yet generated |
| `[MISSING]` | Required file not found |
| `[ERROR]` | Operation failed |
| `[INFO]` | Informational message |

---

## Next Steps

1. **Build:** `./qallow build all`
2. **Run:** `./qallow run cpu`
3. **Benchmark:** `./qallow bench all`
4. **Monitor:** `./qallow status`
5. **Analyze:** `./qallow telemetry all`

---

**Status:** 🟢 **PRODUCTION READY**

All commands operational and tested.

---

**Last Updated:** 2025-10-18  
**Version:** 1.0.0

