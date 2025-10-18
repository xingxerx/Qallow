# Linux Build - Quick Fix

## The Problem

```
nvcc fatal: Unknown option '-Wall'
```

## The Solution

Use the new **`build_simple.sh`** script instead of `build_unified_linux.sh`

---

## 3-Step Setup

### 1. Copy Files to Linux

```bash
# Option A: Using tar (easiest)
# On Windows:
tar -czf qallow.tar.gz backend/ interface/ io/ core/ build_simple.sh

# On Linux:
tar xzf qallow.tar.gz
cd /root/Qallow

# Option B: Using git
git clone <your-repo> /root/Qallow
cd /root/Qallow
```

### 2. Build

```bash
chmod +x build_simple.sh
./build_simple.sh
```

### 3. Run

```bash
./qallow_unified run
```

---

## All 7 Commands

```bash
./qallow_unified build    # Build status
./qallow_unified run      # Execute VM
./qallow_unified bench    # Benchmark
./qallow_unified govern   # Governance
./qallow_unified verify   # Verification
./qallow_unified live     # Live interface
./qallow_unified help     # Help
```

---

## What Changed

| Issue | Fix |
|-------|-----|
| NVCC doesn't support `-Wall` | Use `-Xcompiler -Wall` with NVCC |
| Complex build script | Simplified to `build_simple.sh` |
| CUDA errors | Auto-fallback to CPU-only |

---

## Build Output

```
================================
Building Qallow Unified System
================================

[INFO] GCC version: 15.2.1
[INFO] CUDA version: 13.0

[BUILD] Compiling all modules...
  → (all files)

[BUILD] Linking qallow_unified...
[BUILD] Using NVCC (CUDA enabled)

================================
BUILD SUCCESSFUL (CUDA)
================================
```

---

## If Build Fails

```bash
# Install GCC
sudo apt-get install -y build-essential

# Try again
./build_simple.sh

# If CUDA still fails, it will auto-use CPU
```

---

## Files You Need

```
backend/cpu/*.c
backend/cuda/*.cu (optional)
interface/*.c
io/adapters/*.c
core/include/*.h
build_simple.sh
```

---

**Status**: ✅ FIXED - Ready to build on Linux

