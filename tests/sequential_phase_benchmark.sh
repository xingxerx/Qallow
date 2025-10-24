#!/bin/bash

# Sequential Phase Benchmark Script
# Tests phases in strict order (Phase 1 → 2 → ... → 16) to measure cumulative latency
# Inspired by Heron's high-throughput circuits and Willow's error correction
# Output: data/logs/sequential_benchmark.csv

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
LOG_DIR="${SCRIPT_DIR}/data/logs"
BENCHMARK_LOG="${LOG_DIR}/sequential_benchmark.csv"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo "                    Sequential Phase Benchmark"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

# Check if build directory exists
if [ ! -d "$BUILD_DIR" ]; then
    echo -e "${YELLOW}⚠ Build directory not found. Building...${NC}"
    cd "$SCRIPT_DIR"
    mkdir -p build
    cd build
    cmake ..
    make -j$(nproc)
fi

# Create log directory
mkdir -p "$LOG_DIR"

# Initialize benchmark CSV
echo "phase_id,phase_name,latency_ms,coherence_score,memory_mb,status" > "$BENCHMARK_LOG"

echo -e "${BLUE}Starting sequential phase execution...${NC}\n"

# Array of phase demos
PHASES=(
    "phase01_demo:Phase 1 - Initialization"
    "phase02_demo:Phase 2 - Ingest"
    "phase03_demo:Phase 3 - Adaptive"
    "phase04_demo:Phase 4 - Chronometric"
    "phase05_demo:Phase 5 - Govern"
    "phase06_demo:Phase 6 - Overlay"
    "phase07_demo:Phase 7 - Pocket"
    "phase08_demo:Phase 8 - Ethics"
    "phase09_demo:Phase 9 - Ethics Reasoning"
    "phase10_demo:Phase 10 - Ethics Feedback"
    "phase11_demo:Phase 11 - Quantum"
    "phase12_demo:Phase 12 - Elasticity"
    "phase13_demo:Phase 13 - Harmonic"
)

TOTAL_LATENCY=0
PHASE_COUNT=0

for phase_entry in "${PHASES[@]}"; do
    IFS=':' read -r PHASE_BIN PHASE_NAME <<< "$phase_entry"
    PHASE_NUM=$(echo "$PHASE_NAME" | grep -oP 'Phase \K[0-9]+')
    
    PHASE_PATH="${BUILD_DIR}/${PHASE_BIN}"
    
    if [ ! -f "$PHASE_PATH" ]; then
        echo -e "${YELLOW}⚠ ${PHASE_NAME} binary not found: ${PHASE_PATH}${NC}"
        echo "$PHASE_NUM,$PHASE_NAME,0,0,0,SKIPPED" >> "$BENCHMARK_LOG"
        continue
    fi
    
    echo -e "${BLUE}Running ${PHASE_NAME}...${NC}"
    
    # Measure execution time and memory
    START_TIME=$(date +%s%N)
    
    # Run phase with timeout (30 seconds per phase)
    if timeout 30s "$PHASE_PATH" > /tmp/phase_${PHASE_NUM}_output.txt 2>&1; then
        END_TIME=$(date +%s%N)
        LATENCY_MS=$(( (END_TIME - START_TIME) / 1000000 ))
        
        # Extract coherence score if available
        COHERENCE=$(grep -oP 'coherence[_:]?\s*[=:]\s*\K[0-9.]+' /tmp/phase_${PHASE_NUM}_output.txt | head -1 || echo "0")
        
        # Estimate memory usage (simplified)
        MEMORY_MB=$(ps aux | grep "$PHASE_BIN" | grep -v grep | awk '{print $6/1024}' | head -1 || echo "0")
        
        echo -e "  ${GREEN}✓${NC} Completed in ${LATENCY_MS}ms (coherence: ${COHERENCE})"
        echo "$PHASE_NUM,$PHASE_NAME,$LATENCY_MS,$COHERENCE,$MEMORY_MB,PASSED" >> "$BENCHMARK_LOG"
        
        TOTAL_LATENCY=$((TOTAL_LATENCY + LATENCY_MS))
        PHASE_COUNT=$((PHASE_COUNT + 1))
    else
        echo -e "  ${YELLOW}⚠${NC} Phase timed out or failed"
        echo "$PHASE_NUM,$PHASE_NAME,0,0,0,FAILED" >> "$BENCHMARK_LOG"
    fi
    
    rm -f /tmp/phase_${PHASE_NUM}_output.txt
done

echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo "                        Benchmark Results"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

if [ $PHASE_COUNT -gt 0 ]; then
    AVG_LATENCY=$((TOTAL_LATENCY / PHASE_COUNT))
    echo -e "Phases Executed:     ${GREEN}${PHASE_COUNT}${NC}"
    echo -e "Total Latency:       ${GREEN}${TOTAL_LATENCY}ms${NC}"
    echo -e "Average Latency:     ${GREEN}${AVG_LATENCY}ms${NC}"
    echo ""
    echo -e "Benchmark Log:       ${BLUE}${BENCHMARK_LOG}${NC}"
    echo ""
    
    # Display summary table
    echo "Phase Summary:"
    echo "─────────────────────────────────────────────────────────────────────────────"
    tail -n +2 "$BENCHMARK_LOG" | column -t -s',' | head -20
    echo ""
else
    echo -e "${YELLOW}⚠ No phases executed successfully${NC}"
fi

echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

# Generate performance insights
echo "Performance Insights:"
echo "─────────────────────────────────────────────────────────────────────────────"

# Find slowest phase
SLOWEST=$(tail -n +2 "$BENCHMARK_LOG" | sort -t',' -k3 -rn | head -1)
if [ -n "$SLOWEST" ]; then
    SLOWEST_NAME=$(echo "$SLOWEST" | cut -d',' -f2)
    SLOWEST_TIME=$(echo "$SLOWEST" | cut -d',' -f3)
    echo "Slowest Phase:       ${SLOWEST_NAME} (${SLOWEST_TIME}ms)"
fi

# Find fastest phase
FASTEST=$(tail -n +2 "$BENCHMARK_LOG" | grep "PASSED" | sort -t',' -k3 -n | head -1)
if [ -n "$FASTEST" ]; then
    FASTEST_NAME=$(echo "$FASTEST" | cut -d',' -f2)
    FASTEST_TIME=$(echo "$FASTEST" | cut -d',' -f3)
    echo "Fastest Phase:       ${FASTEST_NAME} (${FASTEST_TIME}ms)"
fi

# Calculate optimization potential
if [ $TOTAL_LATENCY -gt 0 ]; then
    OPT_POTENTIAL=$((TOTAL_LATENCY / 10))
    echo "Optimization Potential: ~${OPT_POTENTIAL}ms (10% improvement target)"
fi

echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

# Recommendations
echo "Recommendations:"
echo "─────────────────────────────────────────────────────────────────────────────"
echo "1. Review slowest phases for optimization opportunities"
echo "2. Profile with NVIDIA Nsight: nsys profile -o profile.qdrep ./phase_demo"
echo "3. Compare with Heron's 150,000 CLOPS baseline"
echo "4. Analyze coherence scores for quantum phases (11-13)"
echo ""

echo -e "${GREEN}✓ Sequential benchmark complete!${NC}"
echo ""

