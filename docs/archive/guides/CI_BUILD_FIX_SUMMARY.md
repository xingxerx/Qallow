# GitHub CI Build Fix - Binary Naming Issue

## Problem

The GitHub Actions workflow was failing with:
```
ERROR: Binary not found at build/CPU/qallow_unified_cpu
```

The CMake build was creating `qallow_unified` but the workflow expected `qallow_unified_cpu`.

## Root Cause

1. **CMakeLists.txt** creates a target named `qallow_unified` without renaming based on accelerator type
2. **Workflow** was using `make ACCELERATOR=CPU` which has Makefile logic to rename binaries, but CMake doesn't
3. **Verification step** was looking for `build/CPU/qallow_unified_cpu` which didn't exist

## Solution

### 1. Updated CMakeLists.txt (Lines 209-215)

Added conditional output name based on CUDA enablement:

```cmake
if(QALLOW_ENABLE_CUDA)
    target_link_libraries(qallow_unified PRIVATE qallow_backend_cuda CUDA::cudart)
    target_compile_definitions(qallow_unified PRIVATE QALLOW_ENABLE_CUDA)
    set_target_properties(qallow_unified PROPERTIES OUTPUT_NAME "qallow_unified_cuda")
else()
    set_target_properties(qallow_unified PROPERTIES OUTPUT_NAME "qallow_unified_cpu")
endif()
```

**Effect**: When building with `-DQALLOW_ENABLE_CUDA=OFF`, the binary is named `qallow_unified_cpu`

### 2. Updated Workflow (.github/workflows/internal-ci.yml)

#### Changed: Build Step (Lines 81-98)

**Before**:
```yaml
- name: Build CPU binary
  run: make ACCELERATOR=CPU -j"$(nproc)" -B
```

**After**:
```yaml
- name: Configure and build (CPU)
  run: |
    set -eux
    rm -rf build/CPU
    cmake -S . -B build/CPU -DQALLOW_ENABLE_CUDA=OFF -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
    echo "After configure:"
    ls -la build/CPU || true
    cmake --build build/CPU -- -j"$(nproc)" VERBOSE=1 2>&1 | tee build/CPU/build.log || (echo "Build failed - see build/CPU/build.log" && tail -n 200 build/CPU/build.log && false)
    echo "After build:"
    ls -la build/CPU || true
    find build/CPU -maxdepth 4 -type f -executable -print -ls || true
    tail -n 200 build/CPU/build.log || true
```

**Improvements**:
- Uses explicit `cmake -S . -B` instead of Makefile wrapper
- Captures verbose build output to `build/CPU/build.log`
- Shows directory listings before and after build
- Searches for executables to verify build succeeded
- Prints last 200 lines of build log for debugging

#### Changed: Verify Step (Lines 100-113)

**Before**:
```yaml
- name: Verify binary exists
  run: |
    if [ ! -f build/CPU/qallow_unified_cpu ]; then
      echo "ERROR: Binary not found at build/CPU/qallow_unified_cpu"
      ls -la build/CPU/ || echo "build/CPU directory does not exist"
      exit 1
    fi
    echo "Binary verified: $(file build/CPU/qallow_unified_cpu)"
```

**After**:
```yaml
- name: Verify binary exists
  run: |
    set -eux
    echo "Listing build/CPU:"
    ls -la build/CPU || true
    echo "Searching for qallow-related executables:"
    find build/CPU -maxdepth 4 -type f -perm /u=x,g=x,o=x -name "*qallow*" -print -ls || true
    if [ -f build/CPU/qallow_unified_cpu ]; then
      echo "Binary verified: $(file build/CPU/qallow_unified_cpu)"
      exit 0
    fi
    echo "ERROR: Expected binary build/CPU/qallow_unified_cpu not found."
    echo "See build/CPU/build.log for build output (if present)."
    exit 1
```

**Improvements**:
- More detailed diagnostics
- Searches for all executables in build directory
- References build log for debugging
- Better error messages

## Testing

To test locally:

```bash
# Clean and configure
rm -rf build/CPU
cmake -S . -B build/CPU -DQALLOW_ENABLE_CUDA=OFF -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

# Build with verbose output
cmake --build build/CPU -- -j$(nproc) VERBOSE=1 2>&1 | tee build/CPU/build.log

# Verify binary
ls -la build/CPU/qallow_unified_cpu
file build/CPU/qallow_unified_cpu
```

## Expected Outcome

✅ CMake configures successfully
✅ Build completes without errors
✅ Binary `build/CPU/qallow_unified_cpu` is created
✅ Workflow verification passes
✅ Smoke tests run successfully

## Files Modified

1. **CMakeLists.txt** - Added conditional output name for qallow_unified target
2. **.github/workflows/internal-ci.yml** - Updated build and verify steps with better diagnostics

## Related Issues

- Binary naming inconsistency between Makefile and CMake
- Lack of verbose build output made debugging difficult
- Workflow verification was too simplistic

## Future Improvements

1. Consider consolidating Makefile and CMake build logic
2. Add build log archiving to workflow artifacts
3. Add more granular build targets for different accelerators

