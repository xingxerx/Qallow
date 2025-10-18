#!/bin/bash
# Build script for Qallow Phase IV on Linux
# Handles CUDA compilation with GCC

set -e  # Exit on error

# ANSI color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Directories
CPU_DIR="backend/cpu"
CUDA_DIR="backend/cuda"
INC_DIR="core/include"
BUILD_DIR="build"

# Output
OUTPUT="qallow_phase4"

echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}Building Qallow Phase IV (Linux)${NC}"
echo -e "${BLUE}================================${NC}"
echo ""

# Check for CUDA
if ! command -v nvcc &> /dev/null; then
    echo -e "${RED}Error: nvcc not found${NC}"
    echo "Please install CUDA Toolkit"
    exit 1
fi

# Check for GCC
if ! command -v gcc &> /dev/null; then
    echo -e "${RED}Error: gcc not found${NC}"
    exit 1
fi

# Print versions
echo -e "${GREEN}[INFO]${NC} CUDA version: $(nvcc --version | grep release | awk '{print $5}' | tr -d ',')"
echo -e "${GREEN}[INFO]${NC} GCC version: $(gcc --version | head -n1 | awk '{print $3}')"
echo ""

# Clean if requested
if [ "$1" == "clean" ]; then
    echo -e "${YELLOW}[CLEAN]${NC} Removing build artifacts..."
    rm -rf $BUILD_DIR
    rm -f $OUTPUT
    rm -f *.csv *.txt
    echo -e "${GREEN}Clean complete.${NC}"
    exit 0
fi

# Create build directory
mkdir -p $BUILD_DIR

# Check source files exist
if [ ! -f "$CPU_DIR/qallow_kernel.c" ]; then
    echo -e "${RED}Error: Source files not found in $CPU_DIR${NC}"
    echo "Expected structure:"
    echo "  backend/cpu/*.c"
    echo "  backend/cuda/*.cu"
    echo "  core/include/*.h"
    exit 1
fi

echo -e "${BLUE}[1/3] Compiling CPU modules...${NC}"
echo "--------------------------------"

# Compile CPU files
for src in $CPU_DIR/*.c; do
    if [ -f "$src" ]; then
        obj="$BUILD_DIR/$(basename $src .c).o"
        echo -e "${GREEN}  →${NC} $(basename $src)"
        gcc -O2 -Wall -I$INC_DIR -DCUDA_ENABLED=1 -c "$src" -o "$obj"
    fi
done

echo ""
echo -e "${BLUE}[2/3] Compiling CUDA kernels (sm_89 for RTX 5080)...${NC}"
echo "--------------------------------"

# Compile CUDA files
for src in $CUDA_DIR/*.cu; do
    if [ -f "$src" ]; then
        obj="$BUILD_DIR/$(basename $src .cu)_cu.o"
        echo -e "${GREEN}  →${NC} $(basename $src)"
        nvcc -O2 -arch=sm_89 -I$INC_DIR -DCUDA_ENABLED=1 -c "$src" -o "$obj"
    fi
done

# Compile demo if it exists
if [ -f "phase4_demo.c" ]; then
    echo ""
    echo -e "${YELLOW}[DEMO]${NC} Compiling phase4_demo.c..."
    gcc -O2 -Wall -I$INC_DIR -DCUDA_ENABLED=1 -c "phase4_demo.c" -o "$BUILD_DIR/phase4_demo.o"
fi

echo ""
echo -e "${BLUE}[3/3] Linking ${OUTPUT}...${NC}"
echo "--------------------------------"

# Collect all object files
OBJ_FILES=$(find $BUILD_DIR -name "*.o")

if [ -z "$OBJ_FILES" ]; then
    echo -e "${RED}Error: No object files found${NC}"
    exit 1
fi

# Link with nvcc (handles CUDA runtime)
nvcc -o $OUTPUT $OBJ_FILES -lcurand -lm

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}================================${NC}"
    echo -e "${GREEN}BUILD SUCCESSFUL${NC}"
    echo -e "${GREEN}================================${NC}"
    echo ""
    echo -e "Executable: ${YELLOW}$OUTPUT${NC}"
    echo ""
    echo "Phase IV Features:"
    echo "  - Multi-Pocket Scheduler (16 parallel worldlines)"
    echo "  - Chronometric Prediction Layer"
    echo "  - Temporal Time Bank"
    echo "  - Drift Detection"
    echo ""
    echo "Usage:"
    echo "  ./$OUTPUT              (8 pockets, 100 ticks)"
    echo "  ./$OUTPUT 16 200       (16 pockets, 200 ticks)"
    echo ""
    echo -e "${GREEN}================================${NC}"
else
    echo ""
    echo -e "${RED}================================${NC}"
    echo -e "${RED}BUILD FAILED${NC}"
    echo -e "${RED}================================${NC}"
    exit 1
fi
