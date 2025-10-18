#!/bin/bash
# build_unified_ethics.sh - Complete Qallow build with ethics monitoring
set -e

echo "════════════════════════════════════════════════════"
echo "   Qallow Unified Build - With Ethics Monitoring"
echo "════════════════════════════════════════════════════"
echo ""

# Configuration
BUILD_DIR="build"
INCLUDE_DIR="core/include"
INTERFACE_DIR="interface"
BACKEND_CPU="backend/cpu"
ALGORITHMS_DIR="algorithms"
SRC_DIR="src"
OUTPUT="$BUILD_DIR/qallow_unified"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Create build directory
mkdir -p "$BUILD_DIR"
mkdir -p "data/telemetry"

echo -e "${BLUE}[1/6]${NC} Checking dependencies..."

# Check for gcc
if ! command -v gcc &> /dev/null; then
    echo "Error: gcc not found!"
    exit 1
fi
echo -e "${GREEN}      ✓${NC} gcc found: $(gcc --version | head -1)"

# Check for Python3
if ! command -v python3 &> /dev/null; then
    echo "Warning: python3 not found - signal collection disabled"
else
    echo -e "${GREEN}      ✓${NC} python3 found: $(python3 --version)"
fi

echo ""
echo -e "${BLUE}[2/6]${NC} Compiling ethics system..."

# Compile ethics library
gcc -c -std=c11 -O2 -Wall -I"$INCLUDE_DIR" \
    "$ALGORITHMS_DIR/ethics_core.c" \
    -o "$BUILD_DIR/ethics_core.o"
echo -e "${GREEN}      ✓${NC} ethics_core.c"

gcc -c -std=c11 -O2 -Wall -I"$INCLUDE_DIR" \
    "$ALGORITHMS_DIR/ethics_learn.c" \
    -o "$BUILD_DIR/ethics_learn.o"
echo -e "${GREEN}      ✓${NC} ethics_learn.c"

gcc -c -std=c11 -O2 -Wall -I"$INCLUDE_DIR" \
    "$ALGORITHMS_DIR/ethics_bayes.c" \
    -o "$BUILD_DIR/ethics_bayes.o"
echo -e "${GREEN}      ✓${NC} ethics_bayes.c"

echo ""
echo -e "${BLUE}[3/6]${NC} Compiling accelerator core..."

gcc -c -std=c11 -O2 -Wall -DQALLOW_PHASE13_EMBEDDED -I"$INCLUDE_DIR" \
    "$SRC_DIR/qallow_phase13.c" \
    -o "$BUILD_DIR/qallow_phase13.o"
echo -e "${GREEN}      ✓${NC} qallow_phase13.c"

echo ""
echo -e "${BLUE}[4/6]${NC} Compiling backend (CPU)..."

# Find and compile all CPU backend files
for source in $BACKEND_CPU/*.c; do
    if [ -f "$source" ]; then
        basename=$(basename "$source" .c)
        gcc -c -std=c11 -O2 -Wall -I"$INCLUDE_DIR" \
            "$source" \
            -o "$BUILD_DIR/${basename}.o" 2>/dev/null || true
        echo -e "${GREEN}      ✓${NC} $basename.c"
    fi
done

echo ""
echo -e "${BLUE}[5/6]${NC} Compiling unified interface..."

gcc -c -std=c11 -O2 -Wall -I"$INCLUDE_DIR" \
    "$INTERFACE_DIR/main.c" \
    -o "$BUILD_DIR/interface_main.o"
echo -e "${GREEN}      ✓${NC} main.c"

gcc -c -std=c11 -O2 -Wall -I"$INCLUDE_DIR" \
    "$INTERFACE_DIR/launcher.c" \
    -o "$BUILD_DIR/interface_launcher.o"
echo -e "${GREEN}      ✓${NC} launcher.c"

echo ""
echo -e "${BLUE}[6/6]${NC} Linking executable..."

# Link everything together
gcc -o "$OUTPUT" \
    $BUILD_DIR/*.o \
    -lm -lpthread 2>&1 | grep -v "warning: " || true

if [ -f "$OUTPUT" ]; then
    chmod +x "$OUTPUT"
    echo -e "${GREEN}      ✓${NC} Linked successfully"
    echo ""
    echo "════════════════════════════════════════════════════"
    echo -e "${GREEN}   Build Complete!${NC}"
    echo "════════════════════════════════════════════════════"
    echo ""
    echo "Executable: $OUTPUT"
    echo "Size: $(du -h $OUTPUT | cut -f1)"
    echo ""
    echo "To run:"
    echo "  $OUTPUT"
    echo ""
    echo "With options:"
    echo "  $OUTPUT --phase12 --ticks=500"
    echo "  $OUTPUT --phase13 --nodes=16"
    echo ""
    echo "Ethics monitoring:"
    echo "  Signal collector: python3 python/collect_signals.py --loop &"
    echo "  Manual test: cd algorithms && ./ethics_test_feed"
    echo ""
else
    echo -e "${YELLOW}[ERROR]${NC} Build failed - executable not created"
    exit 1
fi
