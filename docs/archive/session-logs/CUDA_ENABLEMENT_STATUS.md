# CUDA Enablement Status

## Summary

**Status**: BLOCKED  
**Date**: 2025-11-13  
**Issue**: CMake CUDA configuration fails due to incomplete NVIDIA runtime library installation

## What Was Attempted

1. **Toolkit Installation**
   - Installed `nvidia-cuda-toolkit` via `apt` (Ubuntu 24.04 repos)
   - Version: CUDA 12.0.140 (release 12.0, V12.0.140)
   - `nvcc` binary: `/usr/bin/nvcc` (verified working)
   - Compiler backend `cicc`: `/usr/lib/nvidia-cuda-toolkit/bin/cicc` (verified present)

2. **CMake Configuration**
   - Attempted: `cmake -S . -B build -DQALLOW_ENABLE_CUDA=ON -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/gcc-12`
   - Result: **FAILED** with error:
     ```
     CMake Error: Failed to extract nvcc implicit link line.
     ```

## Root Cause

The package `libnvidia-compute-535` (NVIDIA runtime library) failed to install due to a **cross-device link** error during `dpkg` backup operation:

```
dpkg: error processing archive libnvidia-compute-535_535.274.02-0ubuntu0.24.04.2_amd64.deb (--unpack):
 unable to make backup link of './usr/lib/x86_64-linux-gnu/libcuda.so.1' before installing new version: Invalid cross-device link
```

This is a known issue in containerized environments where `/usr/lib` may be on a different filesystem or overlay layer than the package manager expects for atomic file replacement.

### Symptoms

- `nvcc` compiler driver works (can print version)
- Internal tools (`cicc`, `cudafe++`) are present
- **CMake cannot complete CUDA language detection** because it relies on parsing `nvcc`'s implicit link flags, which requires the CUDA runtime libraries (`libcudart`, `libcuda`) to be fully installed and linkable
- `ldconfig` shows `libcuda.so.1` registered, but it's a stale/incomplete installation artifact

## Workarounds Attempted

1. **Force-install with `dpkg --force-overwrite`**: Failed (file busy)
2. **Manual file backup/move**: Failed (same cross-device issue)
3. **`dpkg --force-unsafe-io`**: Failed (underlying issue persists)
4. **CMake policy CMP0146**: Not applicable to CMake 3.28

## Current State

- **CPU Build**: OPERATIONAL  
  ```bash
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug -DQALLOW_ENABLE_CUDA=OFF
  ```
- **CUDA Build**: BLOCKED until runtime library issue resolved

## Next Steps to Enable CUDA

### Option 1: Fix dpkg/overlay Issue (Recommended for persistent env)

If running in a Docker container or dev container with overlay FS, the host or container config may need adjustment:

1. **Stop any processes using CUDA libraries** (check with `lsof | grep libcuda`)
2. **Remove stale `libcuda.so*` files manually**:
   ```bash
   sudo rm -f /usr/lib/x86_64-linux-gnu/libcuda.so*
   sudo apt-get install --reinstall libnvidia-compute-535
   ```
3. **Or rebuild container** with NVIDIA runtime pre-installed via `nvidia/cuda` base image

### Option 2: Install CUDA from NVIDIA Official Repo (Cleaner)

Use NVIDIA's .deb repository instead of Ubuntu's multiverse:

```bash
# Add NVIDIA CUDA repo
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-0
```

This approach avoids the Ubuntu-packaged version which may have overlay FS compatibility issues.

### Option 3: Manual CUDA Library Paths (Quick Workaround)

If toolkit is installed but CMake can't find libraries, explicitly set paths:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Debug \
  -DQALLOW_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/gcc-12 \
  -DCMAKE_CUDA_IMPLICIT_LINK_LIBRARIES="cudart_static;rt;pthread;dl" \
  -DCMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES="/usr/lib/x86_64-linux-gnu;/usr/local/cuda/lib64"
```

This bypasses CMake's link line parsing but requires knowing exact lib paths.

## Verification Commands

Once resolved, verify CUDA build with:

```bash
# Configure with CUDA
export PATH="/usr/lib/nvidia-cuda-toolkit/bin:$PATH"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug -DQALLOW_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/gcc-12

# Build CUDA target
cmake --build build --target qallow_unified_cuda --parallel

# Run Phase 14 with CUDA backend
./build/qallow_unified_cuda phase 14 --ticks 2000 --nodes 256
```

Expected output: Phase 14 telemetry showing GPU acceleration engaged.

## References

- CUDA 12.0 Release Notes: https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html
- CMake CUDA Support: https://cmake.org/cmake/help/latest/manual/cmake-language.7.html#cuda-modules
- Known issue: dpkg cross-device link in overlayfs: https://bugs.launchpad.net/ubuntu/+source/dpkg/+bug/1965970

---

**Conclusion**: CUDA toolkit (nvcc 12.0) is installed and functional; CMake CUDA configuration blocked by incomplete runtime library installation due to dpkg/filesystem issue. CPU build remains fully operational.
