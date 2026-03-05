# Automated Cleanup System - Qallow

**Status:** ✅ **ACTIVE**  
**Date:** October 23, 2025

---

## Overview

The Qallow project now includes an **automated cleanup system** that ensures a clean, consistent build environment every time you build the project. This system automatically removes redundant files before each build, preventing build issues and maintaining code quality.

---

## How It Works

### 1. **Pre-Build Cleanup Script** (`scripts/pre_build_cleanup.sh`)

Automatically removes:
- ✅ Backup files (`.backup`, `.bak`, `.dup`, `~`)
- ✅ Object files outside `build/` directory (`.o`, `.obj`)
- ✅ Windows-specific files (`.bat`, `.ps1`, `.cmd`, `.exe`)
- ✅ Legacy/demo files (test CSVs, legacy binaries)
- ✅ Verifies essential files exist
- ✅ Checks build directory status

**Runs automatically before every build in CI/CD pipeline.**

### 2. **CI/CD Integration** (`.github/workflows/internal-ci.yml`)

The GitHub Actions workflow now includes:

```yaml
- name: Pre-build cleanup
  run: bash scripts/pre_build_cleanup.sh

- name: Build CPU binary
  run: make ACCELERATOR=CPU -j"$(nproc)" -B
```

**Every CI build automatically runs cleanup first.**

### 3. **Makefile Targets**

New targets added to `Makefile`:

```bash
# Run pre-build cleanup only
make pre-build-cleanup

# Clean build directory
make clean

# Full cleanup (pre-build cleanup + clean)
make clean-all
```

### 4. **Clean Build Script** (`scripts/build_clean.sh`)

Convenience script that runs cleanup + build in one command:

```bash
# Build CPU with auto-detected jobs
./scripts/build_clean.sh

# Build CPU with 4 jobs
./scripts/build_clean.sh CPU 4

# Build CUDA with 8 jobs
./scripts/build_clean.sh CUDA 8
```

---

## Usage

### Option 1: Use the Clean Build Script (Recommended)

```bash
./scripts/build_clean.sh CPU
```

This automatically:
1. Runs pre-build cleanup
2. Builds the project
3. Verifies the binary

### Option 2: Use Makefile Targets

```bash
# Full cleanup + build
make clean-all
make ACCELERATOR=CPU -j$(nproc)

# Or just pre-build cleanup
make pre-build-cleanup
make ACCELERATOR=CPU -j$(nproc)
```

### Option 3: Manual Cleanup

```bash
# Run cleanup manually
bash scripts/pre_build_cleanup.sh

# Then build normally
make ACCELERATOR=CPU -j$(nproc)
```

### Option 4: CI/CD (Automatic)

The cleanup runs automatically in GitHub Actions:
- On every push to `main`, `develop`, or `release/**` branches
- On every pull request
- No manual action needed

---

## What Gets Cleaned

### Removed Files

| Category | Pattern | Reason |
|----------|---------|--------|
| Backup files | `*.backup`, `*.bak`, `*.dup`, `*~` | Development artifacts |
| Object files | `*.o`, `*.obj` (outside build/) | Stale compiled objects |
| Windows files | `*.bat`, `*.ps1`, `*.cmd`, `*.exe` | Platform-specific |
| Demo files | `demo.csv`, `test*.csv`, etc. | Test data |
| Legacy binaries | `qallow_legacy_bin`, etc. | Outdated executables |

### Preserved Files

| Category | Reason |
|----------|--------|
| `build/` directory | Primary build artifacts |
| Source code | All `.c`, `.h`, `.cpp` files |
| Configuration | `config/`, `configs/` directories |
| Documentation | All `.md` files |
| `.git/` directory | Version control |
| `.venv/` directories | Python environments |

---

## Verification

After cleanup, the script verifies:

```
✅ Makefile exists
✅ README.md exists
✅ interface/main.c exists
✅ core/include/phase14.h exists
✅ backend/cpu/phase14_coherence.c exists
✅ Build directory status
✅ No redundant build directories
```

If any essential file is missing, the cleanup fails with an error.

---

## Benefits

1. **Consistency**: Every build starts from a clean state
2. **Reliability**: No stale artifacts causing build failures
3. **Automation**: No manual cleanup needed
4. **CI/CD Integration**: Automatic in GitHub Actions
5. **Disk Space**: Removes unnecessary files (~250MB+ saved)
6. **Development**: Developers can use `./scripts/build_clean.sh` for local builds

---

## Troubleshooting

### Issue: Cleanup removes files I need

**Solution:** Check the cleanup script and add exceptions if needed. The script preserves:
- All source code
- Build artifacts in `build/`
- Configuration files
- Documentation

### Issue: Build still fails after cleanup

**Solution:** 
1. Check that essential files exist: `make pre-build-cleanup`
2. Run full rebuild: `make clean-all && make ACCELERATOR=CPU -j$(nproc)`
3. Check for missing dependencies

### Issue: Cleanup is too aggressive

**Solution:** Modify `scripts/pre_build_cleanup.sh` to add exceptions for specific files or patterns.

---

## Configuration

To customize cleanup behavior, edit `scripts/pre_build_cleanup.sh`:

```bash
# Add new cleanup pattern
cleanup_files "-name '*.myext'" "my extension files"

# Add directory cleanup
cleanup_dirs "my_dir_pattern" "my directories"

# Add exceptions to existing cleanup
find "$PROJECT_ROOT" -type f -name "*.o" \
    ! -path "*/build/*" \
    ! -path "*/.venv/*" \
    ! -path "*/my_exception/*" \
    -delete
```

---

## CI/CD Pipeline

The cleanup is integrated into the GitHub Actions workflow:

```
1. Checkout repository
2. Free up disk space
3. Install dependencies
4. Install CUDA (optional)
5. Check Makefile coverage
6. ✅ PRE-BUILD CLEANUP (NEW)
7. Build CPU binary
8. Verify binary
9. Run smoke tests
10. Accelerator execution
11. Clean up build artifacts
12. Dependency report
```

---

## Summary

The automated cleanup system ensures:
- ✅ Clean builds every time
- ✅ No stale artifacts
- ✅ Consistent CI/CD pipeline
- ✅ Reduced disk usage
- ✅ Reliable development environment

**Status: ACTIVE AND WORKING** ✅

