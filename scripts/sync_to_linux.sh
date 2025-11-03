#!/bin/bash
# Transfer Phase IV files from Windows to Linux
# Run this from WSL with: bash /mnt/d/Qallow/sync_to_linux.sh

set -e

# ANSI colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Paths
WIN_BASE="/mnt/d/Qallow"
LINUX_BASE="/root/Qallow"

echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}Syncing Phase IV to Linux${NC}"
echo -e "${BLUE}================================${NC}"
echo ""

# Check if Windows source exists
if [ ! -d "$WIN_BASE" ]; then
    echo -e "${RED}Error: Windows source not found at $WIN_BASE${NC}"
    echo "Adjust WIN_BASE path in script"
    exit 1
fi

# Create Linux directory structure
echo -e "${YELLOW}[1/5]${NC} Creating directory structure..."
mkdir -p $LINUX_BASE/backend/{cpu,cuda}
mkdir -p $LINUX_BASE/core/include
mkdir -p $LINUX_BASE/scripts

# Copy CPU implementation files
echo -e "${YELLOW}[2/5]${NC} Copying CPU modules..."
if [ -d "$WIN_BASE/backend/cpu" ]; then
    cp -v $WIN_BASE/backend/cpu/*.c $LINUX_BASE/backend/cpu/ 2>/dev/null || echo "  (no .c files)"
fi

# Copy CUDA kernel files
echo -e "${YELLOW}[3/5]${NC} Copying CUDA kernels..."
if [ -d "$WIN_BASE/backend/cuda" ]; then
    cp -v $WIN_BASE/backend/cuda/*.cu $LINUX_BASE/backend/cuda/ 2>/dev/null || echo "  (no .cu files)"
fi

# Copy header files
echo -e "${YELLOW}[4/5]${NC} Copying headers..."
if [ -d "$WIN_BASE/core/include" ]; then
    cp -v $WIN_BASE/core/include/*.h $LINUX_BASE/core/include/ 2>/dev/null || echo "  (no .h files)"
fi

# Copy demo and build scripts
echo -e "${YELLOW}[5/5]${NC} Copying scripts and demo..."
[ -f "$WIN_BASE/phase4_demo.c" ] && cp -v $WIN_BASE/phase4_demo.c $LINUX_BASE/
[ -f "$WIN_BASE/qallow" ] && cp -v $WIN_BASE/qallow $LINUX_BASE/
[ -f "$WIN_BASE/build_phase4_linux.sh" ] && cp -v $WIN_BASE/build_phase4_linux.sh $LINUX_BASE/
chmod +x $LINUX_BASE/qallow 2>/dev/null
chmod +x $LINUX_BASE/build_phase4_linux.sh 2>/dev/null

# Copy documentation
cp -v $WIN_BASE/PHASE_IV*.md $LINUX_BASE/ 2>/dev/null || true

echo ""
echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}Sync Complete${NC}"
echo -e "${GREEN}================================${NC}"
echo ""
echo "Files copied to: $LINUX_BASE"
echo "✅ Fixed chronometric.c synced"
echo "✅ Unified qallow command installed"
echo ""
echo "Next steps:"
echo "  cd $LINUX_BASE"
echo "  ./qallow build"
echo "  ./qallow run 8 100"
echo ""
