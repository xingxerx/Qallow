# Unified Commands - Clarification

## The Issue

There are **multiple build scripts** in the project from different phases:

- ❌ `build_phase4.bat` - Old Phase IV script (not used)
- ❌ `build_phase4_linux.sh` - Old Phase IV Linux script (not used)
- ✅ `scripts/build_wrapper.bat` - **CORRECT** unified system script
- ✅ `qallow.bat` - **CORRECT** unified command wrapper

---

## The Correct Commands

### Windows (What You're Using)

```bash
# Build
.\qallow.bat build

# Run
.\qallow.bat run

# Benchmark
.\qallow.bat bench

# Governance
.\qallow.bat govern

# Verify
.\qallow.bat verify

# Live Interface
.\qallow.bat live

# Help
.\qallow.bat help
```

### Linux (Equivalent)

```bash
# Build the unified system
gcc -O2 -Wall -Icore/include \
    backend/cpu/*.c \
    interface/*.c \
    io/adapters/*.c \
    -o qallow_unified -lm

# Run commands
./qallow_unified build
./qallow_unified run
./qallow_unified bench
./qallow_unified govern
./qallow_unified verify
./qallow_unified live
./qallow_unified help
```

---

## What Each Command Does

| Command | Purpose | Windows | Linux |
|---------|---------|---------|-------|
| build | Compile system | `.\qallow.bat build` | `./qallow_unified build` |
| run | Execute VM | `.\qallow.bat run` | `./qallow_unified run` |
| bench | Benchmark | `.\qallow.bat bench` | `./qallow_unified bench` |
| govern | Governance audit | `.\qallow.bat govern` | `./qallow_unified govern` |
| verify | System check | `.\qallow.bat verify` | `./qallow_unified verify` |
| live | Phase 6 interface | `.\qallow.bat live` | `./qallow_unified live` |
| help | Show help | `.\qallow.bat help` | `./qallow_unified help` |

---

## Build System Architecture

### Windows Build Flow

```
qallow.bat [command]
    ↓
scripts/build_wrapper.bat CPU
    ↓
Visual Studio cl.exe (C compiler)
    ↓
build/qallow.exe
```

### Linux Build Flow

```
gcc -O2 -Wall -Icore/include backend/cpu/*.c interface/*.c io/adapters/*.c -o qallow_unified -lm
    ↓
GCC compiler
    ↓
qallow_unified executable
```

---

## Files to Ignore

These are **old/unused** scripts:

- `build_phase4.bat` - Phase IV only, not unified
- `build_phase4_linux.sh` - Phase IV only, not unified
- `scripts/Makefile` - Old makefile, not used
- `qallow_vm/` - Old directory structure

---

## Files to Use

These are the **correct** unified system files:

- ✅ `scripts/build_wrapper.bat` - Windows build script
- ✅ `qallow.bat` - Windows command wrapper
- ✅ `interface/launcher.c` - Command routing
- ✅ `interface/main.c` - VM execution
- ✅ `backend/cpu/*.c` - Core modules
- ✅ `backend/cuda/*.cu` - GPU kernels (optional)
- ✅ `core/include/*.h` - Headers
- ✅ `io/adapters/*.c` - Data adapters

---

## Quick Reference

### Windows

```bash
# One-liner to build and run
.\qallow.bat build && .\qallow.bat run
```

### Linux

```bash
# One-liner to build and run
gcc -O2 -Wall -Icore/include backend/cpu/*.c interface/*.c io/adapters/*.c -o qallow_unified -lm && ./qallow_unified run
```

---

## Status

✅ **Unified System**: Complete
✅ **7 Commands**: All working
✅ **Windows**: Using `qallow.bat`
✅ **Linux**: Use `gcc` to compile, then run `./qallow_unified`

---

**Version**: Phase 8-10 Complete
**Last Updated**: 2025-10-18

