#!/bin/bash
#
# Clean Build Script
# Automatically runs pre-build cleanup before building
# Ensures a consistent, clean build every time
#
# Usage: ./scripts/build_clean.sh [ACCELERATOR] [JOBS]
# Examples:
#   ./scripts/build_clean.sh              # Build CPU with auto-detected jobs
#   ./scripts/build_clean.sh CPU 4        # Build CPU with 4 jobs
#   ./scripts/build_clean.sh CUDA 8       # Build CUDA with 8 jobs
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Parameters
ACCELERATOR="${1:-CPU}"
JOBS="${2:-$(nproc)}"

# Validate accelerator
if [[ ! "$ACCELERATOR" =~ ^(CPU|CUDA)$ ]]; then
    echo -e "${RED}❌ Invalid ACCELERATOR: $ACCELERATOR${NC}"
    echo "Usage: $0 [CPU|CUDA] [JOBS]"
    exit 1
fi

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║              CLEAN BUILD SCRIPT                               ║${NC}"
echo -e "${BLUE}║                                                                ║${NC}"
echo -e "${BLUE}║  Accelerator: $ACCELERATOR${NC}"
echo -e "${BLUE}║  Jobs: $JOBS${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Step 1: Pre-build cleanup
echo -e "${YELLOW}Step 1: Running pre-build cleanup...${NC}"
bash "$PROJECT_ROOT/scripts/pre_build_cleanup.sh"
echo ""

# Step 2: Build
echo -e "${YELLOW}Step 2: Building with ACCELERATOR=$ACCELERATOR...${NC}"
cd "$PROJECT_ROOT"
make ACCELERATOR="$ACCELERATOR" -j"$JOBS" -B
BUILD_STATUS=$?

if [ $BUILD_STATUS -eq 0 ]; then
    echo ""
    echo -e "${GREEN}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║              ✅ BUILD SUCCESSFUL ✅                            ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    
    # Show binary info
    BINARY="build/$ACCELERATOR/qallow_unified_${ACCELERATOR,,}"
    if [ -f "$BINARY" ]; then
        SIZE=$(ls -lh "$BINARY" | awk '{print $5}')
        echo -e "${GREEN}Binary: $BINARY ($SIZE)${NC}"
        file "$BINARY" | sed 's/^/  /'
    fi
    exit 0
else
    echo ""
    echo -e "${RED}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${RED}║              ❌ BUILD FAILED ❌                                ║${NC}"
    echo -e "${RED}╚════════════════════════════════════════════════════════════════╝${NC}"
    exit 1
fi

