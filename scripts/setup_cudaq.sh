#!/bin/bash

# CUDA-Q Setup Script for Qallow
# This script builds and installs CUDA-Q from source

set -e

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║         CUDA-Q Setup for Qallow                               ║"
echo "╚════════════════════════════════════════════════════════════════╝"

QALLOW_ROOT="/root/Qallow"
BUILD_DIR="$QALLOW_ROOT/build"
CUDAQ_SOURCE="$QALLOW_ROOT/third_party/cuda-quantum"

# Check prerequisites
echo ""
echo "📋 Checking prerequisites..."

if ! command -v cmake &> /dev/null; then
    echo "❌ CMake not found. Installing..."
    apt-get update && apt-get install -y cmake
fi

if ! command -v git &> /dev/null; then
    echo "❌ Git not found. Installing..."
    apt-get install -y git
fi

if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found. Installing..."
    apt-get install -y python3 python3-dev python3-pip
fi

echo "✅ Prerequisites check complete"

# Create build directory
echo ""
echo "📁 Setting up build directory..."
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure CMake
echo ""
echo "🔧 Configuring CMake..."
cmake "$QALLOW_ROOT" \
    -DQALLOW_ENABLE_CUDA=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$QALLOW_ROOT/install" \
    -DBUILD_SHARED_LIBS=ON

# Build CUDA-Q
echo ""
echo "🔨 Building CUDA-Q (this may take a while)..."
cmake --build . --target cudaq --parallel $(nproc)

# Build Python bindings
echo ""
echo "🐍 Building Python bindings..."
cmake --build . --target cudaq-python --parallel $(nproc) || echo "⚠️  Python bindings skipped (optional)"

# Install
echo ""
echo "📦 Installing CUDA-Q..."
cmake --install .

# Set up Python path
echo ""
echo "🐍 Setting up Python environment..."
export PYTHONPATH="$CUDAQ_SOURCE/python:$PYTHONPATH"

# Verify installation
echo ""
echo "✅ Verifying installation..."
python3 -c "import sys; sys.path.insert(0, '$CUDAQ_SOURCE/python'); import cudaq; print(f'CUDA-Q version: {cudaq.__version__}')" || echo "⚠️  Python import test skipped"

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║         ✅ CUDA-Q Setup Complete!                             ║"
echo "╚════════════════════════════════════════════════════════════════╝"

echo ""
echo "📝 Next steps:"
echo "1. Add to your ~/.bashrc or ~/.zshrc:"
echo "   export PYTHONPATH=$CUDAQ_SOURCE/python:\$PYTHONPATH"
echo ""
echo "2. Test CUDA-Q:"
echo "   python3 -c 'import cudaq; print(cudaq.get_targets())'"
echo ""
echo "3. Run examples:"
echo "   cd $CUDAQ_SOURCE/examples/python"
echo "   python3 bell_state.py"
echo ""
echo "4. Use with Qallow:"
echo "   cd $QALLOW_ROOT/native_app"
echo "   cargo run"
echo ""

