#!/bin/bash

################################################################################
# Qallow Recursive Improvement Engine - Main Run Script
# 
# Orchestrates CUDA compilation, Agent Lightning integration, and
# automatic error detection + fixes in a continuous improvement loop
#
# Usage: ./run_recursive_improvement.sh [--iterations 10] [--ticks 120] [--cuda]
################################################################################

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
ITERATIONS=10
TICKS=120
USE_CUDA=true
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${PROJECT_ROOT}/improvement_reports"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --iterations)
            ITERATIONS="$2"
            shift 2
            ;;
        --ticks)
            TICKS="$2"
            shift 2
            ;;
        --no-cuda)
            USE_CUDA=false
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create log directory
mkdir -p "$LOG_DIR"

# Function to print headers
print_header() {
    echo ""
    echo -e "${BLUE}===============================================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}===============================================================================${NC}"
    echo ""
}

# Function to print status
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

# Function to print error
print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# Function to print warning
print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

################################################################################
# Phase 1: Environment Setup
################################################################################

print_header "PHASE 1: Environment Setup"

# Set environment variables
export QALLOW_LOG_LEVEL=INFO
export QALLOW_TELEMETRY_ENABLED=1
export QALLOW_PROFILE_ENABLED=1

if [ "$USE_CUDA" = true ]; then
    print_status "CUDA support enabled"
    export QALLOW_USE_CUDA=1
    export CUDA_VISIBLE_DEVICES=0
else
    print_status "CPU mode enabled"
    export QALLOW_USE_CUDA=0
fi

# Check Python environment
if ! command -v python3 &> /dev/null; then
    print_error "Python3 not found"
    exit 1
fi
print_status "Python3 found: $(python3 --version)"

# Check for required Python packages
print_status "Checking Python dependencies..."
python3 << 'PYEOF'
import sys

required_packages = ['pathlib', 'dataclasses', 'logging']
optional_packages: list[str] = []

for pkg in required_packages:
    try:
        __import__(pkg)
    except ImportError:
        print(f"ERROR: Required package not found: {pkg}")
        sys.exit(1)

for pkg in optional_packages:
    try:
        __import__(pkg)
        print(f"  ✓ {pkg} available")
    except ImportError:
        print(f"  ! {pkg} not installed (optional)")

print("  ✓ All required packages available")
PYEOF

################################################################################
# Phase 2: Dependency Check
################################################################################

print_header "PHASE 2: Dependency Check"

# Check CMake
if ! command -v cmake &> /dev/null; then
    print_error "CMake not found. Install with: apt-get install cmake"
    exit 1
fi
print_status "CMake found: $(cmake --version | head -1)"

# Check C compiler
if ! command -v gcc &> /dev/null; then
    print_error "GCC not found. Install with: apt-get install build-essential"
    exit 1
fi
print_status "GCC found: $(gcc --version | head -1)"

# Check CUDA if enabled
if [ "$USE_CUDA" = true ]; then
    if command -v nvcc &> /dev/null; then
        print_status "NVCC found: $(nvcc --version | tail -1)"
    else
        print_warning "CUDA toolkit not found. Building with CPU backend."
        USE_CUDA=false
    fi
fi

################################################################################
# Phase 3: Build Project
################################################################################

print_header "PHASE 3: Build Project (CUDA: $USE_CUDA)"

cd "$PROJECT_ROOT"

# Clean build directory
print_status "Cleaning previous build..."
rm -rf build

# Configure CMake
print_status "Configuring CMake..."
if [ "$USE_CUDA" = true ]; then
    cmake -S . -B build -DQALLOW_ENABLE_CUDA=ON \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        > "$LOG_DIR/cmake_config_${TIMESTAMP}.log" 2>&1
else
    cmake -S . -B build -DQALLOW_ENABLE_CUDA=OFF \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        > "$LOG_DIR/cmake_config_${TIMESTAMP}.log" 2>&1
fi

if [ $? -ne 0 ]; then
    print_error "CMake configuration failed"
    tail -20 "$LOG_DIR/cmake_config_${TIMESTAMP}.log"
    exit 1
fi

# Build
print_status "Building project..."
NUM_CORES=$(nproc)
print_status "Using $NUM_CORES cores"

cmake --build build --parallel $NUM_CORES \
    > "$LOG_DIR/build_${TIMESTAMP}.log" 2>&1

if [ $? -ne 0 ]; then
    print_error "Build failed"
    tail -30 "$LOG_DIR/build_${TIMESTAMP}.log"
    exit 1
fi

print_status "Build successful"

################################################################################
# Phase 4: Run Recursive Improvement Engine
################################################################################

print_header "PHASE 4: Recursive Improvement Engine"
print_status "Starting improvement loop with $ITERATIONS iterations"
print_status "Phase ticks: $TICKS"

cd "$PROJECT_ROOT"

# Run the Python engine
python3 recursive_improvement_engine.py \
    --iterations $ITERATIONS \
    --ticks $TICKS \
    $([ "$USE_CUDA" = true ] && echo "--cuda") \
    2>&1 | tee "$LOG_DIR/improvement_${TIMESTAMP}.log"

ENGINE_EXIT_CODE=$?

################################################################################
# Phase 5: Generate Report
################################################################################

print_header "PHASE 5: Generate Report"

# Find latest improvement report
LATEST_REPORT=$(ls -t "$LOG_DIR"/recursive_improvement_*.json 2>/dev/null | head -1)

if [ -n "$LATEST_REPORT" ]; then
    print_status "Latest report: $LATEST_REPORT"
    
    # Extract summary with Python
    python3 << PYSUMMARY
import json
from pathlib import Path

with open('$LATEST_REPORT', 'r') as f:
    data = json.load(f)

results = data['results']
total = len(results)
successful = sum(1 for r in results if r['success'])
avg_reward = sum(r['agent_reward'] for r in results) / total if total > 0 else 0

print(f"\n  Total Iterations: {total}")
print(f"  Successful: {successful}/{total}")
print(f"  Average Reward: {avg_reward:.4f}")

# Find best iteration
best = max(results, key=lambda x: x['agent_reward'])
print(f"  Best Iteration: #{best['iteration']} (Reward: {best['agent_reward']:.4f})")

# Count improvements
all_improvements = [imp for r in results for imp in r['improvements_applied']]
print(f"  Improvements Applied: {len(set(all_improvements))}")

PYSUMMARY
fi

################################################################################
# Phase 6: Cleanup & Summary
################################################################################

print_header "FINAL SUMMARY"

print_status "Engine exit code: $ENGINE_EXIT_CODE"
print_status "Logs saved to: $LOG_DIR"

# List all reports
echo ""
print_status "Recent improvement reports:"
ls -lh "$LOG_DIR"/recursive_improvement_*.json 2>/dev/null | tail -3 | awk '{print "  " $9 " (" $5 ")"}'

echo ""
if [ $ENGINE_EXIT_CODE -eq 0 ]; then
    print_status "✨ Recursive improvement completed successfully!"
    print_status "The project has been improved across multiple iterations"
    print_status "using CUDA acceleration and Agent Lightning RL training."
else
    print_warning "Engine completed with warnings or errors"
    print_warning "Check logs for details: $LOG_DIR"
fi

echo ""
print_header "Next Steps"
echo ""
echo "1. Review improvement report:"
echo "   cat $LATEST_REPORT | jq '.results[] | {iteration, success, reward: .agent_reward}'"
echo ""
echo "2. Inspect detailed logs:"
echo "   tail -50 $LOG_DIR/improvement_${TIMESTAMP}.log"
echo ""
echo "3. Run again with different parameters:"
echo "   ./run_recursive_improvement.sh --iterations 15 --ticks 150"
echo ""
echo "4. Check build artifacts:"
echo "   ls -la build/qallow*"
echo ""

exit $ENGINE_EXIT_CODE
