#!/bin/bash
# Build wrapper script for Qallow VM on Linux
# Sets up environment and compiles for CPU or CUDA

set -e # Exit on error

MODE=${1:-CPU}
BUILD_DIR="build"
INCLUDE_DIR="core/include"
BACKEND_CPU="backend/cpu"
BACKEND_CUDA="backend/cuda"
INTERFACE_DIR="interface"
IO_DIR="io/adapters"

mkdir -p "$BUILD_DIR"

echo "[BUILD] Compiling unified launcher and governance core..."

# List of all C source files
C_FILES=(
    "$INTERFACE_DIR/launcher.c"
    "$INTERFACE_DIR/main.c"
    "$BACKEND_CPU/qallow_kernel.c"
    "$BACKEND_CPU/overlay.c"
    "$BACKEND_CPU/ethics.c"
    "$BACKEND_CPU/ppai.c"
    "$BACKEND_CPU/qcp.c"
    "$BACKEND_CPU/pocket_dimension.c"
    "$BACKEND_CPU/telemetry.c"
    "$BACKEND_CPU/adaptive.c"
    "$BACKEND_CPU/pocket.c"
    "$BACKEND_CPU/govern.c"
    "$BACKEND_CPU/ingest.c"
    "$BACKEND_CPU/verify.c"
    "$BACKEND_CPU/semantic_memory.c"
    "$BACKEND_CPU/goal_synthesizer.c"
    "$BACKEND_CPU/transfer_engine.c"
    "$BACKEND_CPU/self_reflection.c"
    "$BACKEND_CPU/phase7_core.c"
    "$BACKEND_CPU/phase12_elasticity.c" # Added new file
    "$IO_DIR/net_adapter.c"
    "$IO_DIR/sim_adapter.c"
)

# Filter C_FILES to only include files that actually exist. Print warnings for missing files.
EXISTING_C_FILES=()
for f in "${C_FILES[@]}"; do
    if [ -f "$f" ]; then
        EXISTING_C_FILES+=("$f")
    else
        echo "[WARN] Source file not found, skipping: $f"
    fi
done

# Use EXISTING_C_FILES for compilation commands

if [ "$MODE" == "CUDA" ] && [ -x "$(command -v nvcc)" ]; then
    echo "[CUDA] Compiling CUDA-enabled version..."

    # Compile CUDA kernels separately
nvcc -c -O2 -arch=sm_89 -I"$INCLUDE_DIR" "$BACKEND_CUDA/ppai_kernels.cu" -o "$BUILD_DIR/ppai_kernels.o"
nvcc -c -O2 -arch=sm_89 -I"$INCLUDE_DIR" "$BACKEND_CUDA/qcp_kernels.cu" -o "$BUILD_DIR/qcp_kernels.o"

# Link all C and CUDA object files together
    nvcc -O2 -arch=sm_89 \
    "${EXISTING_C_FILES[@]}" \
    "$BUILD_DIR/ppai_kernels.o" \
    "$BUILD_DIR/qcp_kernels.o" \
    -I"$INCLUDE_DIR" \
    -L/usr/local/cuda/lib64 -lcudart -lcurand -lm \
    -o "$BUILD_DIR/qallow_unified"


    # Link CUDA executable with all C files and CUDA object files
    nvcc -O2 -arch=sm_89 \
        "${EXISTING_C_FILES[@]}" \
        "$BUILD_DIR/cuda_kernels.o" \
        -I"$INCLUDE_DIR" \
        -L/usr/local/cuda/lib64 -lcudart -lcurand -lm \
        -o "$BUILD_DIR/qallow_unified"

    echo "[SUCCESS] CUDA build completed: $BUILD_DIR/qallow_unified"

else
    if [ "$MODE" == "CUDA" ]; then
        echo "[WARN] nvcc not found. Falling back to CPU-only build."
    fi
    echo "[CPU] Compiling CPU-only version..."

    gcc -O2 -Wall -Wextra -g -I"$INCLUDE_DIR" -o "$BUILD_DIR/qallow_unified" \
        "${EXISTING_C_FILES[@]}" \
        -lm

    echo "[SUCCESS] CPU build completed: $BUILD_DIR/qallow_unified"
fi

echo "[BUILD] Build process completed successfully"
