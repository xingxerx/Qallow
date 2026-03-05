# Quick Start - Linux

## 1-Minute Setup

```bash
# Copy files from Windows
scp -r /mnt/d/Qallow/* user@linux:/root/Qallow/

# Go to directory
cd /root/Qallow

# Make script executable
chmod +x build_unified_linux.sh

# Build
./build_unified_linux.sh

# Run
./qallow_unified run
```

---

## The 7 Commands

```bash
./qallow_unified build    # Show build status
./qallow_unified run      # Execute VM
./qallow_unified bench    # Run benchmark
./qallow_unified govern   # Governance audit
./qallow_unified verify   # System verification
./qallow_unified live     # Phase 6 live interface
./qallow_unified help     # Show help
```

---

## Common Issues & Fixes

### "gcc: command not found"

```bash
sudo apt-get install -y build-essential
```

### "No such file or directory"

```bash
# Make sure you're in the right directory
pwd
# Should show: /root/Qallow

# Check files exist
ls -la backend/cpu/qallow_kernel.c
ls -la interface/launcher.c
```

### "Permission denied"

```bash
chmod +x build_unified_linux.sh
chmod +x qallow_unified
```

### Build fails with "implicit declaration"

```bash
# This is fixed! Just rebuild:
./build_unified_linux.sh clean
./build_unified_linux.sh
```

---

## File Structure (What You Need)

```
/root/Qallow/
├── backend/cpu/*.c          ← CPU modules
├── backend/cuda/*.cu        ← GPU kernels (optional)
├── interface/*.c            ← Launcher & main
├── io/adapters/*.c          ← Data adapters
├── core/include/*.h         ← Headers
└── build_unified_linux.sh   ← Build script
```

---

## What Gets Built

```
qallow_unified  ← Single executable with all 7 commands
```

---

## Performance

- **CPU-only**: Fast to compile, works everywhere
- **With CUDA**: Auto-detected, faster execution

---

## Status

✅ Build script created
✅ Function declarations fixed
✅ All 7 commands ready
✅ Linux support complete

---

**Version**: Phase 8-10 Complete

