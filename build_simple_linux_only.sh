#!/bin/bash
# Simple build script for Qallow Unified System
# Works with or without CUDA
# Copy this file to /root/Qallow/ and run: chmod +x build_simple_linux_only.sh && ./build_simple_linux_only.sh

set -e

echo "================================"
echo "Building Qallow Unified System"
echo "================================"
echo ""

# Check for GCC
if ! command -v gcc &> /dev/null; then
    echo "Error: gcc not found"
    echo "Install with: sudo apt-get install build-essential"
    exit 1
fi

echo "[INFO] GCC version: $(gcc --version | head -n1 | awk '{print $3}')"

# Check for CUDA (optional)
CUDA_AVAILABLE=0
if command -v nvcc &> /dev/null; then
    CUDA_AVAILABLE=1
    echo "[INFO] CUDA version: $(nvcc --version | grep release | awk '{print $5}' | tr -d ',')"
else
    echo "[INFO] CUDA not found - building CPU-only version"
fi

echo ""

# Clean if requested
if [ "$1" == "clean" ]; then
    echo "[CLEAN] Removing build artifacts..."
    rm -f qallow_unified
    rm -f *.o
    rm -f *.csv *.log
    echo "Clean complete."
    exit 0
fi

echo "[BUILD] Compiling all modules..."
echo "--------------------------------"

# Collect all C files
C_FILES=""
for src in backend/cpu/*.c interface/*.c io/adapters/*.c; do
    if [ -f "$src" ]; then
        echo "  → $(basename $src)"
        C_FILES="$C_FILES $src"
    fi
done

# Collect CUDA files if available
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

# Build based on CUDA availability
if [ $CUDA_AVAILABLE -eq 1 ] && [ ! -z "$CU_FILES" ]; then
    echo "[BUILD] Using NVCC (CUDA enabled)"
    if nvcc -O2 -arch=sm_89 -Icore/include -Xcompiler -Wall $C_FILES $CU_FILES -lcurand -lm -o qallow_unified; then
        echo ""
        echo "================================"
        echo "BUILD SUCCESSFUL (CUDA)"
        echo "================================"
    else
        echo ""
        echo "================================"
        echo "BUILD FAILED"
        echo "================================"
        exit 1
    fi
else
    echo "[BUILD] Using GCC (CPU-only)"
    if gcc -O2 -Wall -Icore/include $C_FILES -lm -o qallow_unified; then
        echo ""
        echo "================================"
        echo "BUILD SUCCESSFUL (CPU)"
        echo "================================"
    else
        echo ""
        echo "================================"
        echo "BUILD FAILED"
        echo "================================"
        exit 1
    fi
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

