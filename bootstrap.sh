#!/usr/bin/env bash
################################################################################
# Qallow Self-Provisioning Bootstrap Script
#
# This script automatically sets up the Qallow codebase from a fresh clone by:
# 1. Initializing git submodules
# 2. Creating and activating Python virtual environment
# 3. Installing Python dependencies
# 4. Configuring and building CMake targets
# 5. Running verification tests
#
# Usage:
#   chmod +x bootstrap.sh
#   ./bootstrap.sh [--cuda] [--skip-tests] [--no-python]
#
################################################################################

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Parse command-line arguments
ENABLE_CUDA=${ENABLE_CUDA:-true}
SKIP_TESTS=false
NO_PYTHON=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --cuda)
            ENABLE_CUDA=true
            shift
            ;;
        --no-cuda)
            ENABLE_CUDA=false
            shift
            ;;
        --skip-tests)
            SKIP_TESTS=true
            shift
            ;;
        --no-python)
            NO_PYTHON=true
            shift
            ;;
        --help)
            echo "Usage: ./bootstrap.sh [OPTIONS]"
            echo "Options:"
            echo "  --cuda              Enable CUDA (default: true)"
            echo "  --no-cuda           Disable CUDA"
            echo "  --skip-tests        Skip running tests after build"
            echo "  --no-python         Skip Python virtual environment setup"
            echo "  --help              Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Get absolute path to script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║            Qallow Self-Provisioning Bootstrap                                  ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════════════════════╝${NC}"

# ============================================================================
# Step 1: Git Submodules
# ============================================================================
echo -e "\n${BLUE}[1/5]${NC} ${YELLOW}Initializing git submodules...${NC}"
if ! git submodule update --init --recursive 2>&1 | tail -5; then
    echo -e "${RED}✗ Submodule initialization failed${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Submodules initialized${NC}"

# ============================================================================
# Step 2: Python Virtual Environment & Dependencies
# ============================================================================
if [ "$NO_PYTHON" = false ]; then
    echo -e "\n${BLUE}[2/5]${NC} ${YELLOW}Setting up Python virtual environment...${NC}"
    
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}✗ Python3 not found. Install Python 3.8+ or use --no-python${NC}"
        exit 1
    fi
    
    # Create venv if it doesn't exist
    if [ ! -d ".venv" ]; then
        echo "Creating virtual environment..."
        python3 -m venv .venv
    fi
    
    # Activate venv
    source .venv/bin/activate
    echo "Virtual environment activated"
    
    # Upgrade pip
    echo "Upgrading pip..."
    pip install --upgrade pip setuptools wheel 2>&1 | tail -3
    
    # Install base requirements
    if [ -f "requirements.txt" ]; then
        echo "Installing Python dependencies (requirements.txt)..."
        pip install -r requirements.txt 2>&1 | tail -10
    fi
    
    # Install optional dependencies
    if [ -f "requirements-dev.txt" ]; then
        echo "Installing development dependencies..."
        pip install -r requirements-dev.txt 2>&1 | tail -5
    fi
    
    if [ "$ENABLE_CUDA" = true ] && [ -f "requirements-gpu.txt" ]; then
        echo "Installing GPU dependencies..."
        pip install -r requirements-gpu.txt 2>&1 | tail -5
    fi
    
    if [ -f "requirements-web.txt" ]; then
        echo "Installing web dependencies..."
        pip install -r requirements-web.txt 2>&1 | tail -5
    fi
    
    echo -e "${GREEN}✓ Python environment ready (activate with: source .venv/bin/activate)${NC}"
else
    echo -e "\n${BLUE}[2/5]${NC} ${YELLOW}Skipping Python setup (--no-python)${NC}"
fi

# ============================================================================
# Step 3: Download Assets & Data
# ============================================================================
echo -e "\n${BLUE}[3/5]${NC} ${YELLOW}Fetching assets and data...${NC}"
if [ -f "scripts/fetch_assets.py" ]; then
    if [ "$NO_PYTHON" = false ]; then
        python3 scripts/fetch_assets.py || echo -e "${YELLOW}⚠ Asset fetch completed with warnings${NC}"
    fi
elif [ -f "scripts/download_dependencies.sh" ]; then
    bash scripts/download_dependencies.sh || echo -e "${YELLOW}⚠ Asset download completed with warnings${NC}"
else
    echo "No asset download script found (optional)"
fi
echo -e "${GREEN}✓ Assets ready${NC}"

# ============================================================================
# Step 4: CMake Configuration & Build
# ============================================================================
echo -e "\n${BLUE}[4/5]${NC} ${YELLOW}Configuring and building CMake targets...${NC}"

# Create build directory
mkdir -p build
cd build

# Configure CMake
echo "Configuring CMake..."
if [ "$ENABLE_CUDA" = true ]; then
    cmake -DQALLOW_ENABLE_CUDA=ON .. 2>&1 | tail -10
else
    cmake -DQALLOW_ENABLE_CUDA=OFF .. 2>&1 | tail -10
fi

# Build with all available cores
NPROC=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
echo "Building with $NPROC parallel jobs..."
if ! cmake --build . --parallel "$NPROC" 2>&1 | tail -20; then
    echo -e "${RED}✗ Build failed${NC}"
    exit 1
fi

cd ..
echo -e "${GREEN}✓ Build complete${NC}"

# ============================================================================
# Step 5: Verification Tests
# ============================================================================
if [ "$SKIP_TESTS" = false ]; then
    echo -e "\n${BLUE}[5/5]${NC} ${YELLOW}Running verification tests...${NC}"
    cd build
    
    if ! ctest --output-on-failure 2>&1 | tail -30; then
        echo -e "${YELLOW}⚠ Some tests failed (non-fatal for bootstrap)${NC}"
    else
        echo -e "${GREEN}✓ All tests passed${NC}"
    fi
    
    cd ..
else
    echo -e "\n${BLUE}[5/5]${NC} ${YELLOW}Skipping tests (--skip-tests)${NC}"
fi

# ============================================================================
# Summary
# ============================================================================
echo -e "\n${GREEN}╔════════════════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                    Bootstrap Complete! ✅                                      ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════════════════════════╝${NC}"

echo ""
echo -e "Next steps:"
if [ "$NO_PYTHON" = false ]; then
    echo -e "  1. Activate Python venv:  ${BLUE}source .venv/bin/activate${NC}"
fi
echo -e "  2. Run demo:               ${BLUE}./build/qallow${NC}"
echo -e "  3. Run tests:              ${BLUE}cd build && ctest${NC}"
echo -e "  4. View docs:              ${BLUE}cat README.md${NC}"
echo ""
echo -e "For more info, see: ${BLUE}docs/BOOTSTRAP_GUIDE.md${NC}"
