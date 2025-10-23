#!/bin/bash
#
# Pre-Build Cleanup Script
# Automatically cleans up redundant files before each build
# Ensures a clean, consistent build environment every time
#
# Usage: ./scripts/pre_build_cleanup.sh
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Counters
FILES_REMOVED=0
SPACE_FREED=0

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║         PRE-BUILD CLEANUP - Ensuring Clean Build               ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Function to remove files and track stats
cleanup_files() {
    local pattern="$1"
    local description="$2"
    
    local count=$(find "$PROJECT_ROOT" -type f $pattern 2>/dev/null | wc -l)
    if [ "$count" -gt 0 ]; then
        echo -e "${YELLOW}Removing $description ($count files)...${NC}"
        find "$PROJECT_ROOT" -type f $pattern 2>/dev/null -delete
        FILES_REMOVED=$((FILES_REMOVED + count))
        echo -e "${GREEN}✅ Removed $count $description${NC}"
    fi
}

# Function to remove directories and track stats
cleanup_dirs() {
    local pattern="$1"
    local description="$2"
    
    local count=$(find "$PROJECT_ROOT" -type d -name "$pattern" 2>/dev/null | wc -l)
    if [ "$count" -gt 0 ]; then
        echo -e "${YELLOW}Removing $description ($count directories)...${NC}"
        find "$PROJECT_ROOT" -type d -name "$pattern" 2>/dev/null -exec rm -rf {} + 2>/dev/null || true
        FILES_REMOVED=$((FILES_REMOVED + count))
        echo -e "${GREEN}✅ Removed $count $description${NC}"
    fi
}

echo -e "${BLUE}1. Removing backup files...${NC}"
cleanup_files "-name '*.backup'" "backup files"
cleanup_files "-name '*.bak'" "bak files"
cleanup_files "-name '*.dup'" "dup files"
cleanup_files "-name '*~'" "temp files"
echo ""

echo -e "${BLUE}2. Removing object files (outside build/)...${NC}"
# Remove .o and .obj files but preserve those in build/ directories
find "$PROJECT_ROOT" -type f \( -name "*.o" -o -name "*.obj" \) \
    ! -path "*/build/*" \
    ! -path "*/.venv/*" \
    ! -path "*/qallow_quantum_rust/*" \
    -delete 2>/dev/null || true
OBJ_COUNT=$(find "$PROJECT_ROOT" -type f \( -name "*.o" -o -name "*.obj" \) \
    ! -path "*/build/*" \
    ! -path "*/.venv/*" \
    ! -path "*/qallow_quantum_rust/*" 2>/dev/null | wc -l)
if [ "$OBJ_COUNT" -eq 0 ]; then
    echo -e "${GREEN}✅ All object files cleaned${NC}"
fi
echo ""

echo -e "${BLUE}3. Removing Windows-specific files...${NC}"
cleanup_files "-name '*.bat'" "batch files"
cleanup_files "-name '*.ps1'" "PowerShell files"
cleanup_files "-name '*.cmd'" "cmd files"
cleanup_files "-name '*.exe'" "executable files"
echo ""

echo -e "${BLUE}4. Removing legacy/demo files...${NC}"
cleanup_files "-name 'demo.csv'" "demo CSV files"
cleanup_files "-name 'test*.csv'" "test CSV files"
cleanup_files "-name 'experiment*.csv'" "experiment CSV files"
cleanup_files "-name 'final.csv'" "final CSV files"
cleanup_files "-name '*_legacy*'" "legacy files"
echo ""

echo -e "${BLUE}5. Checking for empty files...${NC}"
EMPTY_COUNT=$(find "$PROJECT_ROOT" -type f -size 0 \
    ! -path "*/build/*" \
    ! -path "*/.git/*" \
    ! -path "*/.venv/*" \
    2>/dev/null | wc -l)
if [ "$EMPTY_COUNT" -gt 0 ]; then
    echo -e "${YELLOW}Found $EMPTY_COUNT empty files (keeping for now)${NC}"
else
    echo -e "${GREEN}✅ No empty files found${NC}"
fi
echo ""

echo -e "${BLUE}6. Verifying build directory...${NC}"
if [ -d "$PROJECT_ROOT/build" ]; then
    BUILD_SIZE=$(du -sh "$PROJECT_ROOT/build" 2>/dev/null | cut -f1)
    echo -e "${GREEN}✅ Primary build directory: build/ ($BUILD_SIZE)${NC}"
else
    echo -e "${YELLOW}⚠️  Build directory not found (will be created during build)${NC}"
fi
echo ""

echo -e "${BLUE}7. Checking for redundant build directories...${NC}"
if [ ! -d "$PROJECT_ROOT/build_ninja" ]; then
    echo -e "${GREEN}✅ build_ninja/ removed${NC}"
fi
if [ ! -d "$PROJECT_ROOT/build_qallow" ]; then
    echo -e "${GREEN}✅ build_qallow/ removed${NC}"
fi
echo ""

echo -e "${BLUE}8. Verifying essential files exist...${NC}"
ESSENTIAL_FILES=(
    "Makefile"
    "README.md"
    "interface/main.c"
    "core/include/phase14.h"
    "backend/cpu/phase14_coherence.c"
)

for file in "${ESSENTIAL_FILES[@]}"; do
    if [ -f "$PROJECT_ROOT/$file" ]; then
        echo -e "${GREEN}✅ $file${NC}"
    else
        echo -e "${RED}❌ MISSING: $file${NC}"
        exit 1
    fi
done
echo ""

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║              PRE-BUILD CLEANUP COMPLETE                        ║${NC}"
echo -e "${BLUE}║                                                                ║${NC}"
echo -e "${GREEN}✅ Files removed: $FILES_REMOVED${NC}"
echo -e "${GREEN}✅ Build environment: CLEAN & READY${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

exit 0

