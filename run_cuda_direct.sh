#!/bin/bash

################################################################################
#                                                                              #
#                    QALLOW DIRECT CUDA EXECUTOR                             #
#                    Skip Agent Lightning Overhead - GPU First                #
#                                                                              #
################################################################################

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[ℹ]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[⚠]${NC} $1"
}

log_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_header() {
    echo ""
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║  $1"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo ""
}

################################################################################
# SETUP
################################################################################

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$PROJECT_ROOT/build"

print_header "QALLOW DIRECT CUDA EXECUTOR v1.0"

# Load CUDA environment
log_info "Loading CUDA environment..."
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Verify CUDA
if ! command -v nvcc &> /dev/null; then
    log_error "CUDA not available in PATH"
    exit 1
fi

CUDA_VERSION=$(nvcc --version | grep release | awk '{print $NF}')
log_success "CUDA found: $CUDA_VERSION"

# Verify build directory
if [ ! -d "$BUILD_DIR" ]; then
    log_warning "Build directory not found. Building now..."
    cd "$PROJECT_ROOT"
    rm -rf build
    mkdir -p build
    cd build
    cmake .. -DWITH_CUDA=ON
    make -j$(nproc)
    cd "$PROJECT_ROOT"
fi

################################################################################
# VERIFY EXECUTABLES
################################################################################

print_header "VERIFYING CUDA EXECUTABLES"

EXECUTABLES=(
    "qallow_unified_cuda"
    "qallow_unit_cuda_parallel"
    "test_kernels"
)

for exe in "${EXECUTABLES[@]}"; do
    if [ -f "$BUILD_DIR/$exe" ]; then
        SIZE=$(ls -lh "$BUILD_DIR/$exe" | awk '{print $5}')
        log_success "$exe ($SIZE)"
    else
        log_warning "$exe not found"
    fi
done

################################################################################
# PARSE ARGUMENTS
################################################################################

COMMAND=${1:-help}
TICKS=${2:-120}

################################################################################
# EXECUTION MODES
################################################################################

case "$COMMAND" in
    
    help)
        cat << 'EOF'
╔═══════════════════════════════════════════════════════════════════════════╗
║               QALLOW DIRECT CUDA EXECUTOR - USAGE                        ║
╚═══════════════════════════════════════════════════════════════════════════╝

Direct GPU execution - skips Agent Lightning overhead for pure performance.

USAGE:
  ./run_cuda_direct.sh <command> [options]

COMMANDS:

  help              Show this help message
  
  run [ticks]       Run unified CUDA executable
                    Example: ./run_cuda_direct.sh run 120
  
  test              Run CUDA unit tests
                    Example: ./run_cuda_direct.sh test
  
  bench             Run performance benchmarks
                    Example: ./run_cuda_direct.sh bench
  
  kernels           Run CUDA kernel tests
                    Example: ./run_cuda_direct.sh kernels
  
  verify            Verify CUDA installation
                    Example: ./run_cuda_direct.sh verify
  
  profile           Run with profiling (if available)
                    Example: ./run_cuda_direct.sh profile
  
  clean             Clean build artifacts
                    Example: ./run_cuda_direct.sh clean
  
  rebuild           Clean and rebuild with CUDA
                    Example: ./run_cuda_direct.sh rebuild

EXAMPLES:

  # Quick CUDA test (1 iteration, 120 ticks)
  ./run_cuda_direct.sh run 120

  # Run full tests
  ./run_cuda_direct.sh test

  # Benchmark performance
  ./run_cuda_direct.sh bench

  # Verify CUDA setup
  ./run_cuda_direct.sh verify

ENVIRONMENT:

  CUDA automatically configured:
    - PATH includes /usr/local/cuda/bin
    - LD_LIBRARY_PATH includes /usr/local/cuda/lib64
  
  GPU selection:
    export CUDA_VISIBLE_DEVICES=0  (for specific GPU)

PERFORMANCE:

  Expected speedup with GPU:
    - Quantum simulations: 10-100x faster
    - Parallel operations: Near-linear scaling
    - Memory usage: GPU VRAM instead of system RAM

For more information:
  See: AGENT_LIGHTNING_ANALYSIS.md
  See: CUDA_QUICK_START.md

EOF
        ;;
    
    run)
        print_header "RUNNING QALLOW UNIFIED WITH CUDA"
        
        if [ ! -f "$BUILD_DIR/qallow_unified_cuda" ]; then
            log_error "qallow_unified_cuda not found. Run: ./run_cuda_direct.sh rebuild"
            exit 1
        fi
        
        log_info "Executable: $BUILD_DIR/qallow_unified_cuda"
        log_info "Ticks: $TICKS"
        log_info "GPU Memory: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
        echo ""
        
        cd "$PROJECT_ROOT"
        "$BUILD_DIR/qallow_unified_cuda" run unified --integrate-ticks="$TICKS"
        
        log_success "Execution completed"
        ;;
    
    test)
        print_header "RUNNING CUDA UNIT TESTS"
        
        if [ ! -f "$BUILD_DIR/qallow_unit_cuda_parallel" ]; then
            log_error "qallow_unit_cuda_parallel not found. Run: ./run_cuda_direct.sh rebuild"
            exit 1
        fi
        
        cd "$BUILD_DIR"
        log_info "Running CUDA parallel tests..."
        echo ""
        
        ./qallow_unit_cuda_parallel
        
        log_success "Tests completed"
        ;;
    
    kernels)
        print_header "RUNNING CUDA KERNEL TESTS"
        
        if [ ! -f "$BUILD_DIR/test_kernels" ]; then
            log_error "test_kernels not found. Run: ./run_cuda_direct.sh rebuild"
            exit 1
        fi
        
        cd "$BUILD_DIR"
        log_info "Running CUDA kernel tests..."
        echo ""
        
        ./test_kernels
        
        log_success "Kernel tests completed"
        ;;
    
    bench)
        print_header "RUNNING PERFORMANCE BENCHMARKS"
        
        if [ ! -f "$BUILD_DIR/qallow_throughput_bench" ]; then
            log_error "qallow_throughput_bench not found"
            exit 1
        fi
        
        cd "$BUILD_DIR"
        log_info "Running throughput benchmark..."
        echo ""
        
        ./qallow_throughput_bench
        
        log_success "Benchmark completed"
        ;;
    
    verify)
        print_header "VERIFYING CUDA INSTALLATION"
        
        log_info "CUDA Compiler:"
        nvcc --version
        echo ""
        
        log_info "CUDA Compiler Location:"
        which nvcc
        echo ""
        
        log_info "CUDA Library Path:"
        echo $LD_LIBRARY_PATH | grep -o '[^:]*cuda[^:]*' || log_warning "CUDA not in LD_LIBRARY_PATH"
        echo ""
        
        if command -v nvidia-smi &> /dev/null; then
            log_info "GPU Information:"
            nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
        else
            log_warning "nvidia-smi not available (may not have GPU drivers)"
        fi
        echo ""
        
        log_info "Build Artifacts:"
        for exe in "${EXECUTABLES[@]}"; do
            if [ -f "$BUILD_DIR/$exe" ]; then
                echo "  ✓ $exe"
            else
                echo "  ✗ $exe (missing)"
            fi
        done
        ;;
    
    profile)
        print_header "RUNNING WITH PROFILING"
        
        if [ ! -f "$BUILD_DIR/qallow_unified_cuda" ]; then
            log_error "qallow_unified_cuda not found"
            exit 1
        fi
        
        log_info "Profiling with nvprof..."
        
        if command -v nvprof &> /dev/null; then
            cd "$PROJECT_ROOT"
            nvprof --print-gpu-trace "$BUILD_DIR/qallow_unified_cuda" run unified --integrate-ticks="$TICKS"
        else
            log_warning "nvprof not found. Trying ncu (NVIDIA Compute Profiler)..."
            if command -v ncu &> /dev/null; then
                cd "$PROJECT_ROOT"
                ncu "$BUILD_DIR/qallow_unified_cuda" run unified --integrate-ticks="$TICKS"
            else
                log_error "No profiler found. Install NVIDIA profiling tools."
                exit 1
            fi
        fi
        ;;
    
    clean)
        print_header "CLEANING BUILD ARTIFACTS"
        
        log_info "Removing build directory: $BUILD_DIR"
        rm -rf "$BUILD_DIR"
        
        log_success "Clean complete"
        ;;
    
    rebuild)
        print_header "REBUILDING WITH CUDA"
        
        log_info "Cleaning..."
        rm -rf "$BUILD_DIR"
        mkdir -p "$BUILD_DIR"
        cd "$BUILD_DIR"
        
        log_info "Configuring CMake with CUDA..."
        cmake .. -DWITH_CUDA=ON
        
        log_info "Building (using $(nproc) cores)..."
        make -j$(nproc)
        
        cd "$PROJECT_ROOT"
        log_success "Rebuild complete"
        ;;
    
    *)
        log_error "Unknown command: $COMMAND"
        echo "Run './run_cuda_direct.sh help' for usage"
        exit 1
        ;;
esac

################################################################################
