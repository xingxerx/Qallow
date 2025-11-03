#!/bin/bash
# Real Computer System Build and Test Script

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"
INSTALL_DIR="$SCRIPT_DIR/install"

echo "╔════════════════════════════════════════════════════════╗"
echo "║  Qallow Real Hardware Execution System                ║"
echo "║  Build & Test Script                                   ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

# Check dependencies
echo "=== Checking Dependencies ==="

if ! command -v cmake &> /dev/null; then
    print_error "CMake not found. Install with: sudo apt-get install cmake"
    exit 1
fi
print_status "CMake found: $(cmake --version | head -n1)"

if ! command -v nvcc &> /dev/null; then
    print_warning "CUDA not found. GPU workloads will not be available."
    print_warning "Install NVIDIA CUDA Toolkit from: https://developer.nvidia.com/cuda-downloads"
else
    print_status "CUDA found: $(nvcc --version | grep release)"
fi

if ! command -v python3 &> /dev/null; then
    print_error "Python3 not found. Install with: sudo apt-get install python3 python3-dev"
    exit 1
fi
print_status "Python3 found: $(python3 --version)"

# Check for Cirq
if python3 -c "import cirq" 2>/dev/null; then
    print_status "Cirq framework found: $(python3 -c 'import cirq; print(cirq.__version__)')"
else
    print_warning "Cirq not installed. Quantum workloads will not be available."
    print_warning "Install with: pip install cirq"
fi

echo ""
echo "=== Building Real Computer System ==="

# Create build directory
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure
print_status "Configuring with CMake..."
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
    2>&1 | grep -E "^--|CUDA|Python" || true

# Build
print_status "Compiling (using $(nproc) cores)..."
make -j$(nproc) 2>&1 | tail -20

# Check build success
if [ -f "$BUILD_DIR/real_computer_demo" ]; then
    print_status "Build successful!"
else
    print_error "Build failed!"
    exit 1
fi

echo ""
echo "=== Running Real Hardware Demo ==="

# Verify executable exists and is executable
if [ ! -x "$BUILD_DIR/real_computer_demo" ]; then
    print_error "Executable not found or not executable"
    exit 1
fi

print_status "Launching real hardware execution demo..."
echo ""

"$BUILD_DIR/real_computer_demo"

DEMO_EXIT=$?

echo ""
if [ $DEMO_EXIT -eq 0 ]; then
    print_status "Demo completed successfully"
else
    print_error "Demo exited with code $DEMO_EXIT"
    exit $DEMO_EXIT
fi

echo ""
echo "=== Build Summary ==="

# List built artifacts
echo "Build artifacts:"
ls -lh "$BUILD_DIR"/*.a 2>/dev/null || true
ls -lh "$BUILD_DIR"/real_computer_demo 2>/dev/null || true

echo ""
echo "=== Next Steps ==="
echo "1. Run demo again: $BUILD_DIR/real_computer_demo"
echo "2. Link libraries in your code:"
echo "   gcc -I$SCRIPT_DIR my_code.c -L$BUILD_DIR -lreal_computer -lcuda -lcirq_quantum -lpython3 -lm"
echo "3. Integrate into Qallow phases 13-15"
echo ""
print_status "Real Hardware system ready for deployment!"
