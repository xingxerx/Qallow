#!/bin/bash
# CPU-only build script for Qallow Unified System
# Use this if CUDA compilation fails

set -e

echo "================================"
echo "Building Qallow Unified System"
echo "================================"
echo ""
echo "[INFO] CPU-only build (CUDA skipped)"
echo ""

# Check GCC
if ! command -v gcc &> /dev/null; then
    echo "Error: gcc not found"
    echo "Install with: sudo apt-get install build-essential"
    exit 1
fi

echo "[INFO] GCC version: $(gcc --version | head -n1 | awk '{print $3}')"
echo ""
echo "[BUILD] Compiling all modules..."
echo "--------------------------------"

# Collect C files
C_FILES=""
for src in backend/cpu/*.c interface/*.c io/adapters/*.c; do
    if [ -f "$src" ]; then
        echo "  → $(basename $src)"
        C_FILES="$C_FILES $src"
    fi
done

echo ""
echo "[BUILD] Linking qallow_unified..."
echo "--------------------------------"

# Build with GCC only
if gcc -O2 -Wall -Icore/include $C_FILES -lm -o qallow_unified; then
    echo ""
    echo "================================"
    echo "BUILD SUCCESSFUL (CPU-ONLY)"
    echo "================================"
else
    echo ""
    echo "================================"
    echo "BUILD FAILED"
    echo "================================"
    exit 1
fi

echo ""
echo "Executable: qallow_unified"
echo ""
echo "Run commands:"
echo "  ./qallow_unified build    # Show build status"
echo "  ./qallow_unified run      # Execute VM"
echo "  ./qallow_unified bench    # Run benchmark"
echo "  ./qallow_unified govern   # Governance audit"
echo "  ./qallow_unified verify   # System verification"
echo "  ./qallow_unified live     # Phase 6 live interface"
echo "  ./qallow_unified help     # Show help"
echo ""

