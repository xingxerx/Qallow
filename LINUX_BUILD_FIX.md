# Linux Build Fix Instructions

## Problem Fixed
The `backend/cuda/pocket.cu` file had several errors preventing Linux build:
1. `inline __device__` without `static` keyword (causes linkage error in non-RDC mode)
2. Missing `CUDA_OK` macro
3. Missing `k_update` kernel
4. Missing global device pointers
5. Missing `pocket.h` include

## Files Updated (Windows)
- ✅ `backend/cuda/pocket.cu` - Fixed all CUDA compilation errors
- ✅ `core/include/qallow_kernel.h` - Added dashboard + CSV declarations
- ✅ `backend/cpu/qallow_kernel.c` - Implemented dashboard + CSV logging
- ✅ `interface/main.c` - Restored proper VM loop with dashboard/CSV integration

## Quick Sync to Linux

### Option 1: Manual Copy (Recommended)
From Linux (/root/Qallow):
```bash
# Mount Windows drive if not already mounted
sudo mkdir -p /mnt/d
sudo mount -t drvfs D: /mnt/d

# Copy specific fixed files
cp /mnt/d/Qallow/backend/cuda/pocket.cu backend/cuda/pocket.cu
cp /mnt/d/Qallow/core/include/qallow_kernel.h core/include/qallow_kernel.h
cp /mnt/d/Qallow/backend/cpu/qallow_kernel.c backend/cpu/qallow_kernel.c
cp /mnt/d/Qallow/interface/main.c interface/main.c

# Build
./build.sh

# Test
./qallow_unified run
QALLOW_LOG=test.csv ./qallow_unified run
cat test.csv
```

### Option 2: Use WSL from Windows
From PowerShell (Windows):
```powershell
wsl cp /mnt/d/Qallow/backend/cuda/pocket.cu /root/Qallow/backend/cuda/pocket.cu
wsl cp /mnt/d/Qallow/core/include/qallow_kernel.h /root/Qallow/core/include/qallow_kernel.h
wsl cp /mnt/d/Qallow/backend/cpu/qallow_kernel.c /root/Qallow/backend/cpu/qallow_kernel.c
wsl cp /mnt/d/Qallow/interface/main.c /root/Qallow/interface/main.c
wsl bash -c "cd /root/Qallow && ./build.sh"
```

## Key Changes in pocket.cu

### Before (Error):
```cuda
__device__ inline double lcg(uint32_t* s){
```

### After (Fixed):
```cuda
static __device__ __inline__ double lcg(uint32_t* s){
```

### Added:
- `#include <cstdint>` for uint32_t
- `#include "pocket.h"` for pocket_cfg_t
- CUDA_OK macro for error checking
- Global device pointers (d_orb, d_riv, d_myc, etc.)
- k_update kernel for pocket evolution
- extern "C" linkage for host functions

## Expected Linux Build Output

**Success:**
```
[BUILD] Compiling CUDA kernels...
  → photonic.cu
  → pocket.cu          ✓ (FIXED)
  → ppai_kernels.cu
  → qcp_kernels.cu
  → quantum.cu

[BUILD] Linking qallow_unified...
BUILD SUCCESSFUL
```

**Dashboard Test:**
```bash
$ ./qallow_unified run
╔═══════════════════════════════════════════════════════════╗
│           Qallow VM Dashboard - Tick 1                  │
╚═══════════════════════════════════════════════════════════╝

OVERLAY STABILITY:
Orbital      | ######################################## | 0.9992
River        | ######################################## | 0.9991
...
```

**CSV Test:**
```bash
$ QALLOW_LOG=test.csv ./qallow_unified run
[CSV] Logging enabled: test.csv
...
[CSV] Log closed

$ cat test.csv
tick,orbital,river,mycelial,global,decoherence,ethics_S,ethics_C,ethics_H,ethics_total,ethics_pass
1,0.999203,0.999126,0.999214,0.999181,0.000009,0.999181,0.999991,0.800000,2.799172,1
```

## Troubleshooting

### If pocket.cu still has errors:
1. Verify all 4 files were copied
2. Check `backend/cuda/pocket.h` exists
3. Try: `nvcc --version` (should be 13.0)
4. Try: `gcc --version` (should be compatible with CUDA 13.0)

### If linker errors:
1. Check build.sh includes pocket.obj
2. Verify CUDA libraries are linked
3. Run: `./build.sh clean && ./build.sh`

### If dashboard doesn't appear:
1. Check terminal supports UTF-8
2. Verify qallow_kernel.c was copied
3. Check main.c has dashboard call at tick % 100

## What Works Now

✅ Windows build successful (build_phase4.bat)  
✅ Dashboard renders with progress bars  
✅ CSV logging works with QALLOW_LOG env var  
✅ All Phase terminology removed  
✅ pocket.cu compiles without errors  
⏳ Linux build pending sync  

## Next Step

**Copy the 4 files to Linux and run ./build.sh** 🚀
