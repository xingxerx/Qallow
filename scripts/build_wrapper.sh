#!/bin/bash
# Build wrapper script for Qallow VM on Linux
# Unified build system for CPU and CUDA compilation
# Usage: ./build_wrapper.sh [CPU|CUDA|AUTO]

set -e # Exit on error

# Configuration
MODE=${1:-AUTO}
BUILD_DIR="build"
INCLUDE_DIR="core/include"
BACKEND_CPU="backend/cpu"
BACKEND_CUDA="backend/cuda"
INTERFACE_DIR="interface"
IO_DIR="io/adapters"
OUTPUT="$BUILD_DIR/qallow_unified"

# ANSI color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Create build directory
mkdir -p "$BUILD_DIR"

echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}Qallow Unified Build System${NC}"
echo -e "${BLUE}================================${NC}"
echo ""

# Detect CUDA availability
CUDA_AVAILABLE=0
if command -v nvcc &> /dev/null; then
    CUDA_AVAILABLE=1
    CUDA_VERSION=$(nvcc --version | grep release | awk '{print $5}' | tr -d ',')
    echo -e "${GREEN}[INFO]${NC} CUDA detected: $CUDA_VERSION"
else
    echo -e "${YELLOW}[INFO]${NC} CUDA not found - CPU-only mode available"
fi

# Determine build mode
if [ "$MODE" == "AUTO" ]; then
    if [ $CUDA_AVAILABLE -eq 1 ]; then
        MODE="CUDA"
        echo -e "${GREEN}[AUTO]${NC} Using CUDA mode"
    else
        MODE="CPU"
        echo -e "${YELLOW}[AUTO]${NC} Using CPU-only mode"
    fi
fi

# Validate mode
if [ "$MODE" != "CPU" ] && [ "$MODE" != "CUDA" ]; then
    echo -e "${RED}[ERROR]${NC} Invalid mode: $MODE"
    echo "Usage: $0 [CPU|CUDA|AUTO]"
    exit 1
fi

# Check if CUDA is requested but not available
if [ "$MODE" == "CUDA" ] && [ $CUDA_AVAILABLE -eq 0 ]; then
    echo -e "${RED}[ERROR]${NC} CUDA requested but nvcc not found"
    echo "Install CUDA or use CPU mode"
    exit 1
fi

echo ""
echo -e "${BLUE}[1/3] Collecting source files...${NC}"
echo "--------------------------------"

# Collect C source files
C_FILES=()
for f in "$INTERFACE_DIR"/*.c "$BACKEND_CPU"/*.c "$IO_DIR"/*.c; do
    if [ -f "$f" ]; then
        C_FILES+=("$f")
        echo -e "${GREEN}  →${NC} $(basename "$f")"
    fi
done

ACCELERATOR_SRC="src/qallow_phase13.c"
if [ -f "$ACCELERATOR_SRC" ]; then
    C_FILES+=("$ACCELERATOR_SRC")
    echo -e "${GREEN}  →${NC} $(basename "$ACCELERATOR_SRC")"
fi

ALGO_SOURCES=(
    "algorithms/ethics_core.c"
    "algorithms/ethics_learn.c"
    "algorithms/ethics_bayes.c"
)
for algo in "${ALGO_SOURCES[@]}"; do
    if [ -f "$algo" ]; then
        C_FILES+=("$algo")
        echo -e "${GREEN}  →${NC} $(basename "$algo")"
    fi
done

if [ ${#C_FILES[@]} -eq 0 ]; then
    echo -e "${RED}[ERROR]${NC} No C source files found"
    exit 1
fi

echo ""
echo -e "${BLUE}[2/3] Compiling...${NC}"
echo "--------------------------------"

C_OBJECTS=()
CUDA_OBJECTS=()

if [ "$MODE" == "CUDA" ]; then
    echo -e "${GREEN}[CUDA]${NC} Compiling CUDA kernels..."

    # Compile CUDA kernels
    for cu_file in "$BACKEND_CUDA"/*.cu; do
        if [ -f "$cu_file" ]; then
            obj_name=$(echo "${cu_file%.cu}" | sed 's/[^A-Za-z0-9_]/_/g')
            obj_file="$BUILD_DIR/${obj_name}.o"
            echo -e "${GREEN}  →${NC} $(basename "$cu_file")"
            nvcc -c -O2 -arch=sm_89 -I"$INCLUDE_DIR" "$cu_file" -o "$obj_file"
            CUDA_OBJECTS+=("$obj_file")
        fi
    done
fi

# Compile C sources
for c_file in "${C_FILES[@]}"; do
    obj_name=$(echo "${c_file%.c}" | sed 's/[^A-Za-z0-9_]/_/g')
    obj_file="$BUILD_DIR/${obj_name}.o"
    echo -e "${GREEN}  →${NC} $(basename "$c_file")"
    extra_flags=()
    if [ "$c_file" = "$ACCELERATOR_SRC" ]; then
        extra_flags+=("-DQALLOW_PHASE13_EMBEDDED")
    fi
    gcc -c -O2 -Wall -Wextra -g -I"$INCLUDE_DIR" "${extra_flags[@]}" "$c_file" -o "$obj_file"
    C_OBJECTS+=("$obj_file")
done

if [ "$MODE" == "CUDA" ]; then
    echo -e "${GREEN}[CUDA]${NC} Linking with CUDA support..."
else
    echo -e "${GREEN}[CPU]${NC} Linking CPU-only version..."
fi

echo ""
echo -e "${BLUE}[3/3] Building executable...${NC}"
echo "--------------------------------"

if [ "$MODE" == "CUDA" ]; then
    nvcc -O2 -arch=sm_89 -I"$INCLUDE_DIR" "${C_OBJECTS[@]}" "${CUDA_OBJECTS[@]}" -L/usr/local/cuda/lib64 -lcudart -lcurand -lm -o "$OUTPUT"
else
    gcc -O2 -Wall -Wextra -g -I"$INCLUDE_DIR" "${C_OBJECTS[@]}" -lm -o "$OUTPUT"
fi

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}================================${NC}"
    echo -e "${GREEN}BUILD SUCCESSFUL${NC}"
    echo -e "${GREEN}================================${NC}"
    echo ""
    echo -e "Mode:       ${YELLOW}$MODE${NC}"
    echo -e "Output:     ${YELLOW}$OUTPUT${NC}"
    echo -e "Size:       ${YELLOW}$(du -h $OUTPUT | cut -f1)${NC}"
    echo ""
else
    echo ""
    echo -e "${RED}================================${NC}"
    echo -e "${RED}BUILD FAILED${NC}"
    echo -e "${RED}================================${NC}"
    exit 1
fi

echo "[BUILD] Build process completed successfully"
