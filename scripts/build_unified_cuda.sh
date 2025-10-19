#!/bin/bash
# build_unified_cuda.sh - Complete Qallow build with CUDA + Ethics Monitoring
set -e

echo "════════════════════════════════════════════════════"
echo "   Qallow Unified Build - CUDA + Ethics"
echo "════════════════════════════════════════════════════"
echo ""

# Configuration
BUILD_DIR="build"
INCLUDE_DIR="core/include"
BACKEND_CPU="backend/cpu"
BACKEND_CUDA="backend/cuda"
INTERFACE_DIR="interface"
SRC_DIR="src"
ALGORITHMS_DIR="algorithms"
OUTPUT="$BUILD_DIR/qallow_unified_cuda"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Create build directory
mkdir -p "$BUILD_DIR"
mkdir -p "data/telemetry"
rm -f "$BUILD_DIR"/*.o 2>/dev/null

echo -e "${BLUE}[1/7]${NC} Checking dependencies..."

# Check for gcc
if ! command -v gcc &> /dev/null; then
    echo -e "${RED}Error: gcc not found!${NC}"
    exit 1
fi
echo -e "${GREEN}      ✓${NC} gcc found: $(gcc --version | head -1)"

# Check for nvcc
if ! command -v nvcc &> /dev/null; then
    echo -e "${RED}Error: nvcc not found!${NC}"
    echo "CUDA compiler required for this build"
    exit 1
fi
CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | tr -d ',')
echo -e "${GREEN}      ✓${NC} nvcc found: CUDA $CUDA_VERSION"

# Check for Python3
if ! command -v python3 &> /dev/null; then
    echo -e "${YELLOW}      !${NC} python3 not found - signal collection disabled"
else
    echo -e "${GREEN}      ✓${NC} python3 found: $(python3 --version)"
fi

echo ""
echo -e "${BLUE}[2/7]${NC} Compiling ethics system (CPU)..."

# Compile ethics library
gcc -c -std=c11 -O2 -Wall -I"$INCLUDE_DIR" \
    "$ALGORITHMS_DIR/ethics_core.c" \
    -o "$BUILD_DIR/ethics_core.o"
echo -e "${GREEN}      ✓${NC} ethics_core.c"

gcc -c -std=c11 -O2 -Wall -I"$INCLUDE_DIR" \
    "$ALGORITHMS_DIR/ethics_learn.c" \
    -o "$BUILD_DIR/ethics_learn.o"
echo -e "${GREEN}      ✓${NC} ethics_learn.c"

gcc -c -std=c11 -O2 -Wall -I"$INCLUDE_DIR" \
    "$ALGORITHMS_DIR/ethics_bayes.c" \
    -o "$BUILD_DIR/ethics_bayes.o"
echo -e "${GREEN}      ✓${NC} ethics_bayes.c"

gcc -c -std=c11 -O2 -Wall -I"$INCLUDE_DIR" \
    "$ALGORITHMS_DIR/ethics_feed.c" \
    -o "$BUILD_DIR/ethics_feed.o"
echo -e "${GREEN}      ✓${NC} ethics_feed.c"

echo ""
echo -e "${BLUE}[3/7]${NC} Compiling CPU backend..."

# Find and compile all CPU backend files
CPU_COUNT=0
for source in $BACKEND_CPU/*.c; do
    if [ -f "$source" ]; then
        basename=$(basename "$source" .c)
        if gcc -c -std=c11 -O2 -Wall -Wno-unused -I"$INCLUDE_DIR" -DUSE_CUDA \
            "$source" \
            -o "$BUILD_DIR/${basename}.o"; then
            echo -e "${GREEN}      ✓${NC} $basename.c"
            ((++CPU_COUNT))
        else
            echo -e "${RED}      ✗${NC} $basename.c"
            exit 1
        fi
    fi
done

echo ""
echo -e "${BLUE}[4/7]${NC} Compiling accelerator core..."

gcc -c -std=c11 -O2 -Wall -I"$INCLUDE_DIR" -DQALLOW_PHASE13_EMBEDDED \
    "$SRC_DIR/qallow_phase13.c" \
    -o "$BUILD_DIR/qallow_phase13.o"
echo -e "${GREEN}      ✓${NC} qallow_phase13.c"

echo ""
echo -e "${BLUE}[5/7]${NC} Compiling CUDA kernels..."

# CUDA compilation flags
CUDA_FLAGS="-std=c++11 -O2 -I$INCLUDE_DIR -DUSE_CUDA"
CUDA_ARCH="-arch=sm_89" # Adjust for target GPU

# Compile CUDA kernels
CUDA_COUNT=0
for source in $BACKEND_CUDA/*.cu; do
    if [ -f "$source" ]; then
        basename=$(basename "$source" .cu)
        if nvcc -c $CUDA_FLAGS $CUDA_ARCH \
            "$source" \
            -o "$BUILD_DIR/${basename}_cuda.o"; then
            echo -e "${GREEN}      ✓${NC} $basename.cu"
        else
            echo -e "${RED}      ✗${NC} $basename.cu"
            exit 1
        fi
        ((++CUDA_COUNT))
    fi
done

echo ""
echo -e "${BLUE}[6/7]${NC} Compiling unified interface..."

gcc -c -std=c11 -O2 -Wall -I"$INCLUDE_DIR" -DUSE_CUDA \
    "$INTERFACE_DIR/main.c" \
    -o "$BUILD_DIR/interface_main.o"
echo -e "${GREEN}      ✓${NC} main.c"

gcc -c -std=c11 -O2 -Wall -I"$INCLUDE_DIR" -DUSE_CUDA \
    "$INTERFACE_DIR/launcher.c" \
    -o "$BUILD_DIR/interface_launcher.o"
echo -e "${GREEN}      ✓${NC} launcher.c"

echo ""
echo -e "${BLUE}[7/7]${NC} Linking CUDA executable..."

# Link with CUDA runtime
nvcc -o "$OUTPUT" \
    $BUILD_DIR/*.o \
    -lm -lpthread 2>&1 | grep -v "warning:" || true

if [ -f "$OUTPUT" ]; then
    chmod +x "$OUTPUT"
    echo -e "${GREEN}      ✓${NC} Linked successfully with CUDA runtime"
else
    echo -e "${RED}      ✗${NC} Linking failed"
    exit 1
fi

echo ""
echo -e "${BLUE}[POST]${NC} Verifying GPU access..."

# Test CUDA availability
if nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    echo -e "${GREEN}      ✓${NC} GPU detected: $GPU_NAME"
    echo -e "${GREEN}      ✓${NC} GPU memory: ${GPU_MEM}MB"
else
    echo -e "${YELLOW}      !${NC} nvidia-smi not available (but build succeeded)"
fi

echo ""
echo "════════════════════════════════════════════════════"
echo -e "${GREEN}   CUDA Build Complete!${NC}"
echo "════════════════════════════════════════════════════"
echo ""
echo "Executable: $OUTPUT"
echo "Size: $(du -h $OUTPUT | cut -f1)"
echo "Modules: $CPU_COUNT CPU + $CUDA_COUNT CUDA + 4 Ethics"
echo ""
echo "To run with CUDA acceleration:"
echo "  $OUTPUT"
echo ""
echo "With specific GPU:"
echo "  CUDA_VISIBLE_DEVICES=0 $OUTPUT"
echo ""
echo "Monitor GPU usage:"
echo "  watch -n 1 nvidia-smi"
echo ""
echo "Ethics monitoring:"
echo "  python3 python/collect_signals.py --loop &"
echo ""
