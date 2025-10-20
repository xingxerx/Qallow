#!/bin/bash
# Build wrapper script for Qallow VM on Linux
# Unified build system for CPU and CUDA compilation
# Usage: ./build_wrapper.sh [CPU|CUDA|AUTO]

set -e # Exit on error

# Configuration
MODE=${1:-AUTO}
BUILD_DIR="build"
INCLUDE_DIR="core/include"
INCLUDE_DIR_ALT="include"
BACKEND_CPU="backend/cpu"
BACKEND_CUDA="backend/cuda"
INTERFACE_DIR="interface"
IO_DIR="io/adapters"
OUTPUT="$BUILD_DIR/qallow_unified"

COMMON_INCLUDES=("-I." "-I${INCLUDE_DIR}" "-Iruntime" "-I${INCLUDE_DIR_ALT}" "-Iethics" "-I/usr/local/cuda/include" "-I/opt/cuda/targets/x86_64-linux/include")
COMMON_DEFINES=("-D_POSIX_C_SOURCE=200809L" "-D_DEFAULT_SOURCE")
TORCH_LINK_FLAGS=""
CUDA_ARCH="${CUDA_ARCH:-sm_90}"

if [ -n "$USE_LIBTORCH" ]; then
    : "${LIBTORCH_HOME:=/opt/libtorch}"
    COMMON_INCLUDES+=("-I${LIBTORCH_HOME}/include" "-I${LIBTORCH_HOME}/include/torch/csrc/api/include")
    COMMON_DEFINES+=("-DUSE_LIBTORCH")
    : "${LIBTORCH_LIB:=${LIBTORCH_HOME}/lib}"
    TORCH_LINK_FLAGS="-L${LIBTORCH_LIB} -Wl,-rpath,${LIBTORCH_LIB} -ltorch_cpu -ltorch -lc10"
    if [ -n "$USE_LIBTORCH_CUDA" ]; then
        TORCH_LINK_FLAGS+=" -ltorch_cuda"
    fi
fi

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

# Apply CUDA preprocessor definition based on final mode
if [ "$MODE" == "CUDA" ]; then
    COMMON_DEFINES+=("-DCUDA_ENABLED=1")
else
    COMMON_DEFINES+=("-DCUDA_ENABLED=0")
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

RUNTIME_SOURCES=(
    "runtime/meta_introspect.c"
)
for runtime_src in "${RUNTIME_SOURCES[@]}"; do
    if [ -f "$runtime_src" ]; then
        C_FILES+=("$runtime_src")
        echo -e "${GREEN}  →${NC} $(basename "$runtime_src")"
    fi
done

if [ ${#C_FILES[@]} -eq 0 ]; then
    echo -e "${RED}[ERROR]${NC} No C source files found"
    exit 1
fi

# Collect C++ runtime sources
CPP_FILES=()
for cpp in src/runtime/*.cpp; do
    if [ -f "$cpp" ]; then
        CPP_FILES+=("$cpp")
        echo -e "${GREEN}  →${NC} $(basename "$cpp")"
    fi
done

DL_CPP="runtime/dl_integration.cpp"
if [ -f "$DL_CPP" ]; then
    CPP_FILES+=("$DL_CPP")
    echo -e "${GREEN}  →${NC} $(basename "$DL_CPP")"
fi

echo ""
echo -e "${BLUE}[2/3] Compiling...${NC}"
echo "--------------------------------"

C_OBJECTS=()
CUDA_OBJECTS=()
CPP_OBJECTS=()

if [ "$MODE" == "CUDA" ]; then
    echo -e "${GREEN}[CUDA]${NC} Compiling CUDA kernels..."

    # Compile CUDA kernels
    for cu_file in "$BACKEND_CUDA"/*.cu; do
        if [ -f "$cu_file" ]; then
            obj_name=$(echo "${cu_file%.cu}" | sed 's/[^A-Za-z0-9_]/_/g')
            obj_file="$BUILD_DIR/${obj_name}.o"
            echo -e "${GREEN}  →${NC} $(basename "$cu_file")"
            nvcc -c -O2 -arch="${CUDA_ARCH}" "${COMMON_INCLUDES[@]}" "${COMMON_DEFINES[@]}" "$cu_file" -o "$obj_file"
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
    gcc -c -O2 -std=c11 -Wall -Wextra -g "${COMMON_INCLUDES[@]}" "${COMMON_DEFINES[@]}" "${extra_flags[@]}" "$c_file" -o "$obj_file"
    C_OBJECTS+=("$obj_file")
done

# Compile C++ sources
for cpp_file in "${CPP_FILES[@]}"; do
    obj_name=$(echo "${cpp_file%.cpp}" | sed 's/[^A-Za-z0-9_]/_/g')
    obj_file="$BUILD_DIR/${obj_name}.o"
    echo -e "${GREEN}  →${NC} $(basename "$cpp_file")"
    g++ -c -O2 -std=c++17 -Wall -Wextra -g "${COMMON_INCLUDES[@]}" "${COMMON_DEFINES[@]}" "$cpp_file" -o "$obj_file"
    CPP_OBJECTS+=("$obj_file")
done

if [ "$MODE" == "CUDA" ]; then
    echo -e "${GREEN}[CUDA]${NC} Linking with CUDA support..."
else
    echo -e "${GREEN}[CPU]${NC} Linking CPU-only version..."
fi

echo ""
echo -e "${BLUE}[3/3] Building executable...${NC}"
echo "--------------------------------"

LINK_OBJECTS=("${C_OBJECTS[@]}" "${CPP_OBJECTS[@]}")

if [ "$MODE" == "CUDA" ]; then
    nvcc -O2 -arch="${CUDA_ARCH}" "${COMMON_INCLUDES[@]}" "${COMMON_DEFINES[@]}" "${LINK_OBJECTS[@]}" "${CUDA_OBJECTS[@]}" -L/usr/local/cuda/lib64 -lcudart -lcurand -lm $TORCH_LINK_FLAGS -o "$OUTPUT"
else
    g++ -O2 -Wall -Wextra -g "${COMMON_INCLUDES[@]}" "${COMMON_DEFINES[@]}" "${LINK_OBJECTS[@]}" -lm $TORCH_LINK_FLAGS -o "$OUTPUT"
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
