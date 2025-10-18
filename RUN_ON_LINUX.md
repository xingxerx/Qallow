# Run These Commands on Linux

You already have the files copied to `/root/Qallow`. Just run these commands:

## Step 1: Create the Build Script

Copy and paste this entire command into your Linux terminal:

```bash
cat > /root/Qallow/build.sh << 'EOF'
#!/bin/bash
set -e
echo "================================"
echo "Building Qallow Unified System"
echo "================================"
echo ""

if ! command -v gcc &> /dev/null; then
    echo "Error: gcc not found"
    echo "Install with: sudo apt-get install build-essential"
    exit 1
fi

echo "[INFO] GCC version: $(gcc --version | head -n1 | awk '{print $3}')"

CUDA_AVAILABLE=0
if command -v nvcc &> /dev/null; then
    CUDA_AVAILABLE=1
    echo "[INFO] CUDA version: $(nvcc --version | grep release | awk '{print $5}' | tr -d ',')"
else
    echo "[INFO] CUDA not found - building CPU-only version"
fi

echo ""
echo "[BUILD] Compiling all modules..."
echo "--------------------------------"

C_FILES=""
for src in backend/cpu/*.c interface/*.c io/adapters/*.c; do
    if [ -f "$src" ]; then
        echo "  → $(basename $src)"
        C_FILES="$C_FILES $src"
    fi
done

CU_FILES=""
if [ $CUDA_AVAILABLE -eq 1 ]; then
    echo ""
    echo "[BUILD] Compiling CUDA kernels..."
    for src in backend/cuda/*.cu; do
        if [ -f "$src" ]; then
            echo "  → $(basename $src)"
            CU_FILES="$CU_FILES $src"
        fi
    done
fi

echo ""
echo "[BUILD] Linking qallow_unified..."
echo "--------------------------------"

if [ $CUDA_AVAILABLE -eq 1 ] && [ ! -z "$CU_FILES" ]; then
    echo "[BUILD] Using NVCC (CUDA enabled)"
    nvcc -O2 -arch=sm_89 -Icore/include -Xcompiler -Wall $C_FILES $CU_FILES -lcurand -lm -o qallow_unified
    echo ""
    echo "================================"
    echo "BUILD SUCCESSFUL (CUDA)"
    echo "================================"
else
    echo "[BUILD] Using GCC (CPU-only)"
    gcc -O2 -Wall -Icore/include $C_FILES -lm -o qallow_unified
    echo ""
    echo "================================"
    echo "BUILD SUCCESSFUL (CPU)"
    echo "================================"
fi

echo ""
echo "Executable: qallow_unified"
echo ""
echo "Commands:"
echo "  ./qallow_unified build    # Show build status"
echo "  ./qallow_unified run      # Execute VM"
echo "  ./qallow_unified bench    # Run benchmark"
echo "  ./qallow_unified govern   # Governance audit"
echo "  ./qallow_unified verify   # System verification"
echo "  ./qallow_unified live     # Phase 6 live interface"
echo "  ./qallow_unified help     # Show help"
echo ""
EOF
```

## Step 2: Make it Executable and Build

```bash
chmod +x /root/Qallow/build.sh
cd /root/Qallow
./build.sh
```

## Step 3: Run the Commands

```bash
# Execute VM
./qallow_unified run

# Run benchmark
./qallow_unified bench

# System verification
./qallow_unified verify

# Live interface
./qallow_unified live

# Show help
./qallow_unified help
```

---

## All in One

Copy and paste this entire block:

```bash
cat > /root/Qallow/build.sh << 'EOF'
#!/bin/bash
set -e
echo "================================"
echo "Building Qallow Unified System"
echo "================================"
echo ""

if ! command -v gcc &> /dev/null; then
    echo "Error: gcc not found"
    echo "Install with: sudo apt-get install build-essential"
    exit 1
fi

echo "[INFO] GCC version: $(gcc --version | head -n1 | awk '{print $3}')"

CUDA_AVAILABLE=0
if command -v nvcc &> /dev/null; then
    CUDA_AVAILABLE=1
    echo "[INFO] CUDA version: $(nvcc --version | grep release | awk '{print $5}' | tr -d ',')"
else
    echo "[INFO] CUDA not found - building CPU-only version"
fi

echo ""
echo "[BUILD] Compiling all modules..."
echo "--------------------------------"

C_FILES=""
for src in backend/cpu/*.c interface/*.c io/adapters/*.c; do
    if [ -f "$src" ]; then
        echo "  → $(basename $src)"
        C_FILES="$C_FILES $src"
    fi
done

CU_FILES=""
if [ $CUDA_AVAILABLE -eq 1 ]; then
    echo ""
    echo "[BUILD] Compiling CUDA kernels..."
    for src in backend/cuda/*.cu; do
        if [ -f "$src" ]; then
            echo "  → $(basename $src)"
            CU_FILES="$CU_FILES $src"
        fi
    done
fi

echo ""
echo "[BUILD] Linking qallow_unified..."
echo "--------------------------------"

if [ $CUDA_AVAILABLE -eq 1 ] && [ ! -z "$CU_FILES" ]; then
    echo "[BUILD] Using NVCC (CUDA enabled)"
    nvcc -O2 -arch=sm_89 -Icore/include -Xcompiler -Wall $C_FILES $CU_FILES -lcurand -lm -o qallow_unified
    echo ""
    echo "================================"
    echo "BUILD SUCCESSFUL (CUDA)"
    echo "================================"
else
    echo "[BUILD] Using GCC (CPU-only)"
    gcc -O2 -Wall -Icore/include $C_FILES -lm -o qallow_unified
    echo ""
    echo "================================"
    echo "BUILD SUCCESSFUL (CPU)"
    echo "================================"
fi

echo ""
echo "Executable: qallow_unified"
echo ""
echo "Commands:"
echo "  ./qallow_unified build    # Show build status"
echo "  ./qallow_unified run      # Execute VM"
echo "  ./qallow_unified bench    # Run benchmark"
echo "  ./qallow_unified govern   # Governance audit"
echo "  ./qallow_unified verify   # System verification"
echo "  ./qallow_unified live     # Phase 6 live interface"
echo "  ./qallow_unified help     # Show help"
echo ""
EOF
chmod +x /root/Qallow/build.sh
cd /root/Qallow
./build.sh
./qallow_unified run
```

---

## If Build Fails

### "gcc: command not found"

```bash
sudo apt-get update
sudo apt-get install -y build-essential
cd /root/Qallow
./build.sh
```

### "No such file or directory"

```bash
# Check files exist
ls -la /root/Qallow/backend/cpu/
ls -la /root/Qallow/interface/
ls -la /root/Qallow/core/include/

# Check you're in right directory
pwd
# Should show: /root/Qallow
```

### CUDA errors

The script will auto-fallback to CPU-only. Just run:

```bash
cd /root/Qallow
./build.sh
```

---

## Status

✅ Files already copied to `/root/Qallow`
✅ Build script ready to create
✅ All 7 commands ready to use

---

**Next**: Copy the "All in One" command above and paste it into your Linux terminal!

