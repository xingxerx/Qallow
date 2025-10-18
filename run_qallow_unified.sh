#!/bin/bash
# run_qallow_unified.sh - Complete launcher for Qallow with ethics monitoring

BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo ""
echo -e "${BLUE}════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}   Qallow Unified Application Launcher${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════${NC}"
echo ""

# Check if built
if [ ! -f "build/qallow_unified" ]; then
    echo -e "${YELLOW}Executable not found. Building...${NC}"
    ./scripts/build_unified_ethics.sh
    echo ""
fi

# Setup environment
mkdir -p data/telemetry
mkdir -p data/snapshots

# Set default human feedback if not exists
if [ ! -f "data/human_feedback.txt" ]; then
    echo "0.75" > data/human_feedback.txt
    echo -e "${GREEN}✓${NC} Created default human feedback (0.75)"
fi

# Collect initial signals
echo -e "${GREEN}✓${NC} Collecting initial hardware signals..."
python3 python/collect_signals.py 2>/dev/null

echo ""
echo -e "${BLUE}Starting Qallow VM...${NC}"
echo ""

# Run with passed arguments
./build/qallow_unified "$@"

EXIT_CODE=$?

echo ""
echo -e "${BLUE}════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}Session Complete${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════${NC}"
echo ""
echo "Exit code: $EXIT_CODE"
echo ""
echo "Logs:"
echo "  Ethics audit:    data/ethics_audit.log"
echo "  Signal history:  data/telemetry/collection.log"
echo "  Latest signals:  data/telemetry/current_signals.json"
echo ""
echo "View ethics decisions:"
echo "  tail -20 data/ethics_audit.log"
echo ""
echo "Adjust human feedback:"
echo "  echo '0.85' > data/human_feedback.txt"
echo ""

exit $EXIT_CODE
