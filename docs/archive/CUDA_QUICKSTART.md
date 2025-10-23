# CUDA Quick Reference - Qallow VM

## ✅ CUDA Status: FULLY OPERATIONAL

### Your System
- **GPU:** NVIDIA GeForce RTX 5080 (16GB)
- **CUDA:** 13.0.88
- **Driver:** 581.42
- **Architecture:** sm_89

---

## Build & Run Commands

### Windows (PowerShell)
```powershell
# Build CUDA version
.\build.bat

# Run
.\qallow.exe

# Clean and rebuild
.\build.bat clean
.\build.bat
```

### Check GPU Status
```powershell
nvcc --version          # CUDA compiler
nvidia-smi              # GPU info
nvidia-smi -l 1         # Monitor GPU (refresh every 1s)
```

---

## VS Code

Press **Ctrl+Shift+B** → Select:
- **build original CUDA** (recommended)
- **run original CUDA version**

---

## Files

### Working Build System
- `build.bat` ← Use this! ✅
- `qallow.exe` ← Output executable

### CUDA Source
- `backend/cuda/*.cu` ← GPU kernels
- `backend/cpu/*.c` ← CPU implementations
- `core/include/*.h` ← Headers

---

## Project is READY! 🚀

Your CUDA environment is fully configured and working.
Just use `.\build.bat` and `.\qallow.exe` to run!
