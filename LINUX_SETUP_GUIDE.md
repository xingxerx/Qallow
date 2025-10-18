# Qallow Linux Setup Guide

## Prerequisites

### Install Build Tools

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y build-essential gcc g++ make

# CentOS/RHEL
sudo yum groupinstall -y "Development Tools"

# Verify installation
gcc --version
```

### Optional: Install CUDA

```bash
# Download from: https://developer.nvidia.com/cuda-downloads
# Or install via package manager:

# Ubuntu/Debian
sudo apt-get install -y nvidia-cuda-toolkit

# Verify installation
nvcc --version
```

---

## Setup Steps

### 1. Copy Files to Linux

From Windows (WSL or SCP):

```bash
# Using SCP
scp -r /mnt/d/Qallow/* user@linux-host:/root/Qallow/

# Or manually copy:
# - backend/cpu/*.c
# - backend/cuda/*.cu (optional)
# - interface/*.c
# - io/adapters/*.c
# - core/include/*.h
# - build_unified_linux.sh
```

### 2. Make Build Script Executable

```bash
cd /root/Qallow
chmod +x build_unified_linux.sh
```

### 3. Build the System

```bash
# CPU-only build (fastest)
./build_unified_linux.sh

# Clean build
./build_unified_linux.sh clean
./build_unified_linux.sh
```

### 4. Verify Build

```bash
# Check executable exists
ls -la qallow_unified

# Check it's executable
file qallow_unified
```

---

## Running Commands

### All 7 Commands

```bash
# Build status
./qallow_unified build

# Execute VM
./qallow_unified run

# Benchmark
./qallow_unified bench

# Governance audit
./qallow_unified govern

# System verification
./qallow_unified verify

# Phase 6 live interface
./qallow_unified live

# Help
./qallow_unified help
```

---

## Troubleshooting

### Error: "No such file or directory"

**Problem**: Build script can't find source files

**Solution**:
```bash
# Verify directory structure
ls -la backend/cpu/
ls -la interface/
ls -la core/include/

# Make sure you're in the right directory
pwd
# Should output: /root/Qallow (or your Qallow directory)
```

### Error: "gcc: command not found"

**Problem**: GCC not installed

**Solution**:
```bash
# Ubuntu/Debian
sudo apt-get install -y build-essential

# CentOS/RHEL
sudo yum groupinstall -y "Development Tools"

# Verify
gcc --version
```

### Error: "implicit declaration of function"

**Problem**: Missing header files or function declarations

**Solution**:
```bash
# Make sure all .h files are in core/include/
ls -la core/include/

# Rebuild
./build_unified_linux.sh clean
./build_unified_linux.sh
```

### Error: "cannot open file"

**Problem**: Linker can't find libraries

**Solution**:
```bash
# Make sure -lm flag is included (math library)
# The build script should handle this automatically

# Try manual build:
gcc -O2 -Wall -Icore/include backend/cpu/*.c interface/*.c io/adapters/*.c -o qallow_unified -lm
```

### Build Succeeds but Executable Won't Run

**Problem**: Missing dependencies or permission issues

**Solution**:
```bash
# Check dependencies
ldd ./qallow_unified

# Make executable
chmod +x qallow_unified

# Run with full path
./qallow_unified run
```

---

## Performance Tips

### CPU-Only Build (Recommended for First Build)

```bash
./build_unified_linux.sh
```

**Pros**: Fast to compile, works everywhere
**Cons**: Slower execution

### CUDA Build (If CUDA Available)

```bash
# Build script auto-detects CUDA
./build_unified_linux.sh

# If CUDA is installed, it will be included automatically
```

**Pros**: Faster execution
**Cons**: Requires CUDA toolkit

### Optimize Compilation

```bash
# Add optimization flags
gcc -O3 -march=native -Wall -Icore/include backend/cpu/*.c interface/*.c io/adapters/*.c -o qallow_unified -lm
```

---

## File Structure

```
/root/Qallow/
├── backend/
│   ├── cpu/
│   │   ├── qallow_kernel.c
│   │   ├── overlay.c
│   │   ├── ethics.c
│   │   ├── ppai.c
│   │   ├── qcp.c
│   │   ├── pocket_dimension.c
│   │   ├── telemetry.c
│   │   ├── adaptive.c
│   │   ├── pocket.c
│   │   ├── govern.c
│   │   ├── ingest.c
│   │   ├── verify.c
│   │   ├── semantic_memory.c
│   │   ├── goal_synthesizer.c
│   │   ├── transfer_engine.c
│   │   ├── self_reflection.c
│   │   └── phase7_core.c
│   └── cuda/
│       ├── ppai_kernels.cu
│       ├── qcp_kernels.cu
│       ├── photonic.cu
│       ├── quantum.cu
│       └── pocket.cu
├── core/
│   └── include/
│       ├── qallow_kernel.h
│       ├── ppai.h
│       ├── qcp.h
│       ├── ethics.h
│       ├── overlay.h
│       ├── sandbox.h
│       ├── telemetry.h
│       ├── pocket.h
│       ├── phase7.h
│       ├── ingest.h
│       └── verify.h
├── interface/
│   ├── launcher.c
│   └── main.c
├── io/
│   └── adapters/
│       ├── net_adapter.c
│       └── sim_adapter.c
├── build_unified_linux.sh
└── qallow_unified (executable, created after build)
```

---

## Quick Start

```bash
# 1. Copy files to Linux
scp -r /mnt/d/Qallow/* user@linux:/root/Qallow/

# 2. Build
cd /root/Qallow
chmod +x build_unified_linux.sh
./build_unified_linux.sh

# 3. Run
./qallow_unified run

# 4. Try other commands
./qallow_unified bench
./qallow_unified verify
./qallow_unified live
```

---

## Status

✅ **Linux Build**: Supported
✅ **CPU Build**: Fully functional
✅ **CUDA Build**: Auto-detected and included
✅ **All 7 Commands**: Working

---

**Version**: Phase 8-10 Complete
**Last Updated**: 2025-10-18

