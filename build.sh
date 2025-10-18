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

if [ -f src/qallow_phase13.c ]; then
    echo "  → qallow_phase13.c"
    C_FILES="$C_FILES src/qallow_phase13.c"
fi

for src in algorithms/ethics_core.c algorithms/ethics_learn.c algorithms/ethics_bayes.c; do
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
    nvcc -O2 -arch=sm_89 -Icore/include -DQALLOW_PHASE13_EMBEDDED -Xcompiler -Wall $C_FILES $CU_FILES -lcurand -lm -o qallow_unified
    echo ""
    echo "================================"
    echo "BUILD SUCCESSFUL (CUDA)"
    echo "================================"
else
    echo "[BUILD] Using GCC (CPU-only)"
    gcc -O2 -Wall -Icore/include -DQALLOW_PHASE13_EMBEDDED $C_FILES -lm -o qallow_unified
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
