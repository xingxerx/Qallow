# Qallow Linux Commands - Unified System

## Quick Start

### Build the Unified System

```bash
# Compile all modules (CPU + CUDA support)
gcc -O2 -Wall -Icore/include \
    backend/cpu/*.c \
    interface/*.c \
    io/adapters/*.c \
    -o qallow_unified -lm

# Or with CUDA support (if CUDA is installed)
nvcc -O2 -arch=sm_89 -Icore/include \
    backend/cpu/*.c \
    backend/cuda/*.cu \
    interface/*.c \
    io/adapters/*.c \
    -o qallow_unified -lcurand -lm
```

### Run the Unified System

```bash
# Run with command routing (same as Windows)
./qallow_unified build
./qallow_unified run
./qallow_unified bench
./qallow_unified govern
./qallow_unified verify
./qallow_unified live
./qallow_unified help
```

---

## Unified Command System (7 Commands)

All commands use the same executable with different arguments:

### 1. Build

```bash
./qallow_unified build
```

**Output**: Shows build status
**Result**: Confirms system is compiled

### 2. Run

```bash
./qallow_unified run
```

**Output**: Executes the VM
**Result**: Runs with adaptive-predictive-temporal loop

### 3. Benchmark

```bash
./qallow_unified bench
```

**Output**: Runs benchmark with logging
**Result**: Performance metrics logged

### 4. Governance Audit

```bash
./qallow_unified govern
```

**Output**: Runs autonomous governance audit
**Result**: Ethics monitoring and validation

### 5. System Verification

```bash
./qallow_unified verify
```

**Output**: System checkpoint
**Result**: Verifies coherence, decoherence, ethics scores

### 6. Live Interface

```bash
./qallow_unified live
```

**Output**: Phase 6 live interface with data ingestion
**Result**: Streams telemetry to CSV file

### 7. Help

```bash
./qallow_unified help
```

**Output**: Shows help message
**Result**: Displays all available commands

---

## Common Workflows

### Development Build & Test

```bash
# Clean and rebuild
rm -f qallow_unified *.o

# Rebuild
gcc -O2 -Wall -Icore/include backend/cpu/*.c interface/*.c io/adapters/*.c -o qallow_unified -lm

# Quick test
./qallow_unified run
```

### Benchmark Run

```bash
# Run benchmark
./qallow_unified bench

# View results
tail -f qallow_stream.csv
```

### Production Run

```bash
# Run in background
nohup ./qallow_unified run > qallow_$(date +%s).log 2>&1 &

# Monitor in background
watch -n 1 'tail -5 qallow_stream.csv'
```

### Continuous Monitoring

```bash
# Run in background and log output
nohup ./qallow_unified live > qallow_$(date +%s).log 2>&1 &

# Check status
ps aux | grep qallow_unified

# View logs
tail -f qallow_*.log
```

---

## Troubleshooting

### Build Fails: GCC Not Found

```bash
# Check GCC installation
gcc --version

# Install GCC
# Ubuntu/Debian:
sudo apt-get install build-essential

# CentOS/RHEL:
sudo yum groupinstall "Development Tools"
```

### Build Fails: CUDA Not Found (Optional)

```bash
# Check CUDA installation
nvcc --version

# Install CUDA Toolkit (optional, CPU-only works fine)
# Ubuntu/Debian:
sudo apt-get install nvidia-cuda-toolkit

# Or download from: https://developer.nvidia.com/cuda-downloads
```

### Runtime Error: Permission Denied

```bash
# Make executable
chmod +x qallow_unified

# Run again
./qallow_unified run
```

### Runtime Error: Command Not Found

```bash
# Make sure you're in the right directory
pwd
ls -la qallow_unified

# Run with full path
./qallow_unified run
```

---

## File Locations

| File | Purpose |
|------|---------|
| `qallow_unified` | Main executable |
| `qallow_stream.csv` | Telemetry output |
| `qallow_bench.log` | Benchmark results |
| `backend/cpu/*.c` | CPU source files |
| `backend/cuda/*.cu` | CUDA kernel files (optional) |
| `core/include/*.h` | Header files |
| `interface/*.c` | Command launcher |
| `io/adapters/*.c` | Data adapters |

---

## Performance Tips

1. **CPU-only build** (fastest to compile): `gcc -O2 -Wall -Icore/include backend/cpu/*.c interface/*.c io/adapters/*.c -o qallow_unified -lm`
2. **CUDA build** (faster execution): `nvcc -O2 -arch=sm_89 -Icore/include backend/cpu/*.c backend/cuda/*.cu interface/*.c io/adapters/*.c -o qallow_unified -lcurand -lm`
3. **Monitor system**: `watch -n 1 nvidia-smi`
4. **Run multiple instances**: `./qallow_unified run & ./qallow_unified run &`

---

## Status

✅ **Linux Build**: Supported
✅ **CUDA Support**: Enabled (sm_89 for RTX 5080)
✅ **CPU Fallback**: Available
✅ **All 7 Commands**: Functional

---

**Version**: Phase 8-10 Complete
**Last Updated**: 2025-10-18

