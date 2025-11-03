#!/bin/bash

################################################################################
# Quick Start: Recursive Improvement with CUDA + Agent Lightning
# 
# This is the main entry point. It:
# 1. Validates environment
# 2. Runs recursive improvement engine
# 3. Applies fixes automatically
# 4. Regenerates improvements each cycle
################################################################################

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Print startup banner
cat << 'BANNER'

 ██████╗  █████╗ ██╗     ██╗      ██████╗ ██╗    ██╗
██╔═══██╗██╔══██╗██║     ██║     ██╔═══██╗██║    ██║
██║   ██║███████║██║     ██║     ██║   ██║██║ █╗ ██║
██║▄▄██║██╔══██║██║     ██║     ██║   ██║██║███╗██║
╚██████╔╝██║  ██║███████╗███████╗╚██████╔╝╚███╔███╔╝
 ╚══▀▀═╝ ╚═╝  ╚═╝╚══════╝╚══════╝ ╚═════╝  ╚══╝╚══╝

 Recursive Improvement Engine v1.0
 CUDA + Agent Lightning + Auto-Fix Integration

BANNER

echo "Starting from: $PROJECT_ROOT"
echo ""

# Check if we're in the right directory
if [ ! -f "$PROJECT_ROOT/CMakeLists.txt" ]; then
    echo "ERROR: CMakeLists.txt not found. Are you in the Qallow project root?"
    exit 1
fi

# Default parameters
ITERATIONS=${1:-10}
TICKS=${2:-120}
MODE=${3:-cuda}  # cuda or cpu

echo "Configuration:"
echo "  Iterations: $ITERATIONS"
echo "  Ticks per phase: $TICKS"
echo "  Mode: $MODE (CUDA-accelerated)" 
echo ""

# Run the recursive improvement engine
cd "$PROJECT_ROOT"

exec ./run_recursive_improvement.sh \
    --iterations "$ITERATIONS" \
    --ticks "$TICKS" \
    $([ "$MODE" = "cpu" ] && echo "--no-cuda" || echo "")
