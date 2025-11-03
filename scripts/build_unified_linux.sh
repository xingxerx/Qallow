#!/bin/bash
# Build script for Qallow Unified System on Linux
# Compiles all modules: CPU + optional CUDA support

set -e  # Exit on error

# ANSI color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Directories
INC_DIR="core/include"
CPU_DIR="backend/cpu"
CUDA_DIR="backend/cuda"
INTERFACE_DIR="interface"
IO_DIR="io/adapters"

# Output
OUTPUT="qallow_unified"

echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}Building Qallow Unified System${NC}"
echo -e "${BLUE}================================${NC}"
echo ""

# Check for GCC
if ! command -v gcc &> /dev/null; then
    echo -e "${RED}Error: gcc not found${NC}"
    echo "Install with: sudo apt-get install build-essential"
    exit 1
fi

# Print versions
echo -e "${GREEN}[INFO]${NC} GCC version: $(gcc --version | head -n1 | awk '{print $3}')"

# Check for CUDA (optional)
CUDA_AVAILABLE=0
if command -v nvcc &> /dev/null; then
    CUDA_AVAILABLE=1
    echo -e "${GREEN}[INFO]${NC} CUDA version: $(nvcc --version | grep release | awk '{print $5}' | tr -d ',')"
else
    echo -e "${YELLOW}[INFO]${NC} CUDA not found - building CPU-only version"
fi

echo ""

# Clean if requested
if [ "$1" == "clean" ]; then
    echo -e "${YELLOW}[CLEAN]${NC} Removing build artifacts..."
    rm -f $OUTPUT
    rm -f *.o
    rm -f *.csv *.log
    echo -e "${GREEN}Clean complete.${NC}"
    exit 0
fi

# Check source files exist
if [ ! -f "$CPU_DIR/qallow_kernel.c" ]; then
    echo -e "${RED}Error: Source files not found${NC}"
    echo "Expected structure:"
    echo "  backend/cpu/*.c"
    echo "  interface/*.c"
    echo "  io/adapters/*.c"
    echo "  core/include/*.h"
    exit 1
fi

echo -e "${BLUE}[1/2] Compiling all modules...${NC}"
echo "--------------------------------"

# Compile all C files
COMPILE_CMD="gcc -O2 -Wall -I$INC_DIR -DQALLOW_PHASE13_EMBEDDED"

# Add all CPU source files
for src in $CPU_DIR/*.c; do
    if [ -f "$src" ]; then
        echo -e "${GREEN}  →${NC} $(basename $src)"
        COMPILE_CMD="$COMPILE_CMD $src"
    fi
done

# Add interface files
for src in $INTERFACE_DIR/*.c; do
    if [ -f "$src" ]; then
        echo -e "${GREEN}  →${NC} $(basename $src)"
        COMPILE_CMD="$COMPILE_CMD $src"
    fi
done

# Add IO adapter files
for src in $IO_DIR/*.c; do
    if [ -f "$src" ]; then
        echo -e "${GREEN}  →${NC} $(basename $src)"
        COMPILE_CMD="$COMPILE_CMD $src"
    fi
done

ACCELERATOR_SRC="src/qallow_phase13.c"
if [ -f "$ACCELERATOR_SRC" ]; then
    echo -e "${GREEN}  →${NC} $(basename $ACCELERATOR_SRC)"
    COMPILE_CMD="$COMPILE_CMD $ACCELERATOR_SRC"
fi

for src in algorithms/ethics_core.c algorithms/ethics_learn.c algorithms/ethics_bayes.c; do
    if [ -f "$src" ]; then
        echo -e "${GREEN}  →${NC} $(basename $src)"
        COMPILE_CMD="$COMPILE_CMD $src"
    fi
done

# Add CUDA files if available
if [ $CUDA_AVAILABLE -eq 1 ]; then
    echo ""
    echo -e "${BLUE}[1/2] Compiling CUDA kernels...${NC}"
    for src in $CUDA_DIR/*.cu; do
        if [ -f "$src" ]; then
            echo -e "${GREEN}  →${NC} $(basename $src)"
            COMPILE_CMD="$COMPILE_CMD $src"
        fi
    done
    COMPILE_CMD="nvcc -O2 -arch=sm_89 -I$INC_DIR -DQALLOW_PHASE13_EMBEDDED $COMPILE_CMD -lcurand -lm"
else
    COMPILE_CMD="$COMPILE_CMD -lm"
fi

echo ""
echo -e "${BLUE}[2/2] Linking ${OUTPUT}...${NC}"
echo "--------------------------------"

# Execute compile command
if eval "$COMPILE_CMD -o $OUTPUT"; then
    echo ""
    echo -e "${GREEN}================================${NC}"
    echo -e "${GREEN}BUILD SUCCESSFUL${NC}"
    echo -e "${GREEN}================================${NC}"
    echo ""
    echo -e "Executable: ${YELLOW}$OUTPUT${NC}"
    echo ""
    echo "Unified Commands:"
    echo "  ./$OUTPUT build    # Show build status"
    echo "  ./$OUTPUT run      # Execute VM"
    echo "  ./$OUTPUT bench    # Run benchmark"
    echo "  ./$OUTPUT govern   # Governance audit"
    echo "  ./$OUTPUT verify   # System verification"
    echo "  ./$OUTPUT live     # Phase 6 live interface"
    echo "  ./$OUTPUT help     # Show help"
    echo ""
    echo "Features:"
    echo "  - Phases 1-7: Core VM + Governance + AGI"
    echo "  - Phases 8-10: Adaptive-Predictive-Temporal Loop"
    echo "  - Phase 6: Live data integration"
    echo "  - Ethics enforcement (E = S + C + H)"
    echo "  - Sandbox isolation and rollback"
    echo ""
    echo -e "${GREEN}================================${NC}"
else
    echo ""
    echo -e "${RED}================================${NC}"
    echo -e "${RED}BUILD FAILED${NC}"
    echo -e "${RED}================================${NC}"
    exit 1
fi
