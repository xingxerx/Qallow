#!/bin/bash

################################################################################
#                                                                              #
#                      CUDA INSTALLATION VERIFICATION SCRIPT                  #
#                  Post-Installation Verification & Diagnostics              #
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
# VERIFICATION CHECKS
################################################################################

print_header "CUDA INSTALLATION VERIFICATION"

# Ensure environment is loaded
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

echo ""
log_info "Environment Setup"
echo "  PATH includes CUDA: $(echo $PATH | grep -q '/usr/local/cuda' && echo '✓' || echo '✗')"
echo "  LD_LIBRARY_PATH: ${LD_LIBRARY_PATH:-not set}"

# Check 1: CUDA Compiler
print_header "Check 1: CUDA Compiler (nvcc)"

if command -v nvcc &> /dev/null; then
    log_success "nvcc found in PATH"
    NVCC_VERSION=$(nvcc --version | grep release | awk '{print $NF}')
    log_success "CUDA Version: $NVCC_VERSION"
    nvcc --version
else
    log_error "nvcc not found"
    echo "  Searched locations:"
    echo "    - /usr/local/cuda/bin/nvcc"
    echo "    - PATH: $PATH"
fi

echo ""

# Check 2: CUDA Libraries
print_header "Check 2: CUDA Runtime Libraries"

CUDA_LIB_DIR="/usr/local/cuda/lib64"

if [ -d "$CUDA_LIB_DIR" ]; then
    log_success "CUDA library directory found: $CUDA_LIB_DIR"
    echo ""
    log_info "Key CUDA Libraries:"
    for lib in libcuda.so libcudart.so libcublas.so libcufft.so libcurand.so; do
        if [ -f "$CUDA_LIB_DIR/$lib"* ]; then
            COUNT=$(ls -1 "$CUDA_LIB_DIR/$lib"* 2>/dev/null | wc -l)
            log_success "$lib (variants: $COUNT)"
        else
            log_warning "$lib not found"
        fi
    done
else
    log_error "CUDA library directory not found at $CUDA_LIB_DIR"
fi

echo ""

# Check 3: CUDA Include Files
print_header "Check 3: CUDA Include Files"

CUDA_INC_DIR="/usr/local/cuda/include"

if [ -d "$CUDA_INC_DIR" ]; then
    log_success "CUDA include directory found: $CUDA_INC_DIR"
    FILE_COUNT=$(ls -1 "$CUDA_INC_DIR" | wc -l)
    log_success "Total header files: $FILE_COUNT"
    echo ""
    log_info "Key CUDA Headers:"
    for header in cuda.h cuda_runtime.h cublas.h cufft.h curand.h; do
        if [ -f "$CUDA_INC_DIR/$header" ]; then
            log_success "$header"
        else
            log_warning "$header not found"
        fi
    done
else
    log_error "CUDA include directory not found at $CUDA_INC_DIR"
fi

echo ""

# Check 4: GPU Detection
print_header "Check 4: GPU Device Detection"

if command -v nvidia-smi &> /dev/null; then
    log_success "nvidia-smi found"
    echo ""
    nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader | while read -r line; do
        log_success "GPU: $line"
    done
else
    log_warning "nvidia-smi not found (may not have GPU drivers in WSL)"
    log_info "This is expected in WSL. Check host Windows for GPU."
fi

echo ""

# Check 5: CMake CUDA Support
print_header "Check 5: CMake CUDA Integration"

if command -v cmake &> /dev/null; then
    log_success "CMake found"
    CMAKE_VERSION=$(cmake --version | head -1)
    log_success "$CMAKE_VERSION"
    
    # Create a test CMakeLists.txt
    TEST_DIR=$(mktemp -d)
    cat > "$TEST_DIR/CMakeLists.txt" << 'CMAKEOF'
cmake_minimum_required(VERSION 3.18)
project(CUDATest LANGUAGES CUDA)
message(STATUS "CUDA Found: ${CUDA_FOUND}")
message(STATUS "CUDA Compiler: ${CMAKE_CUDA_COMPILER}")
message(STATUS "CUDA Toolkit: ${CUDA_TOOLKIT_ROOT_DIR}")
CMAKEOF
    
    log_info "Testing CMake CUDA support..."
    if cd "$TEST_DIR" && cmake . > /dev/null 2>&1; then
        log_success "CMake CUDA support verified"
    else
        log_warning "CMake CUDA test inconclusive"
    fi
    rm -rf "$TEST_DIR"
else
    log_warning "CMake not found"
fi

echo ""

# Check 6: Qallow Build
print_header "Check 6: Qallow Build Configuration"

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$PROJECT_DIR/build"

if [ -d "$BUILD_DIR/CMakeFiles" ]; then
    log_success "Qallow build directory found: $BUILD_DIR"
    
    if grep -q "CUDA" "$BUILD_DIR/CMakeCache.txt" 2>/dev/null; then
        log_success "Build configured with CUDA support"
    else
        log_warning "CUDA not detected in CMake cache"
    fi
    
    if [ -f "$BUILD_DIR/qallow_unified" ]; then
        log_success "Qallow executable built successfully"
        SIZE=$(ls -lh "$BUILD_DIR/qallow_unified" | awk '{print $5}')
        log_success "Binary size: $SIZE"
    else
        log_warning "Qallow executable not found"
    fi
else
    log_warning "Build directory not configured (run enable_cuda.sh first)"
fi

echo ""

# Check 7: Environment Variables
print_header "Check 7: Shell Configuration Files"

log_info "Checking ~/.bashrc..."
if grep -q "export PATH=/usr/local/cuda/bin" ~/.bashrc; then
    log_success "CUDA PATH configured in ~/.bashrc"
else
    log_warning "CUDA PATH not found in ~/.bashrc"
fi

if [ -f ~/.zshrc ]; then
    log_info "Checking ~/.zshrc..."
    if grep -q "export PATH=/usr/local/cuda/bin" ~/.zshrc; then
        log_success "CUDA PATH configured in ~/.zshrc"
    else
        log_warning "CUDA PATH not found in ~/.zshrc"
    fi
fi

echo ""

# Summary
print_header "✅ VERIFICATION SUMMARY"

echo "Next Steps:"
echo ""
echo "1. Source environment (if not already done):"
echo "   source ~/.bashrc"
echo ""
echo "2. Rebuild Qallow:"
echo "   cd $PROJECT_DIR"
echo "   rm -rf build && mkdir build && cd build"
echo "   cmake .. -DWITH_CUDA=ON"
echo "   make -j\$(nproc)"
echo ""
echo "3. Test CUDA execution:"
echo "   cd $PROJECT_DIR"
echo "   ./run_with_improvement.sh 10 120 cuda"
echo ""
echo "4. Verify CUDA is being used in output:"
echo "   ./run_with_improvement.sh 10 120 cuda 2>&1 | grep -i 'cuda\\|gpu\\|device'"
echo ""

################################################################################
