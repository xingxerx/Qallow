#!/bin/bash
# run_qallow_cuda.sh - Launch Qallow with CUDA acceleration and ethics monitoring

BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo ""
echo -e "${BLUE}════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}   Qallow CUDA-Accelerated Launcher${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════${NC}"
echo ""

# Check if CUDA executable exists
if [ ! -f "build/qallow_unified_cuda" ]; then
    echo -e "${YELLOW}CUDA executable not found. Building...${NC}"
    ./scripts/build_unified_cuda.sh
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

# Check GPU
echo -e "${GREEN}GPU Status:${NC}"
if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.used,memory.total,temperature.gpu --format=csv,noheader)
    echo "  $GPU_INFO"
    
    # Collect GPU metrics for ethics
    GPU_TEMP=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    GPU_MEM_TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)
    GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' %')
    
    echo "  Temperature: ${GPU_TEMP}°C"
    echo "  Memory: ${GPU_MEM}MB / ${GPU_MEM_TOTAL}MB"
    echo "  Utilization: ${GPU_UTIL}%"
    
    # Warn if GPU is hot or memory is high
    if [ "$GPU_TEMP" -gt 80 ]; then
        echo -e "${RED}  ⚠  GPU temperature high!${NC}"
    fi
    
    if [ "$GPU_MEM" -gt $((GPU_MEM_TOTAL * 90 / 100)) ]; then
        echo -e "${YELLOW}  ⚠  GPU memory usage high${NC}"
    fi
else
    echo -e "${YELLOW}  ! nvidia-smi not available${NC}"
fi

# Collect initial signals (including GPU metrics)
echo ""
echo -e "${GREEN}✓${NC} Collecting hardware signals (CPU + GPU)..."
python3 python/collect_signals.py 2>/dev/null

echo ""
echo -e "${BLUE}Starting Qallow with CUDA acceleration...${NC}"
echo ""

# Run with passed arguments
./build/qallow_unified_cuda "$@"

EXIT_CODE=$?

# Final GPU check
echo ""
if command -v nvidia-smi &> /dev/null; then
    echo -e "${BLUE}Final GPU Status:${NC}"
    nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,memory.used --format=csv,noheader
fi

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
echo "GPU monitoring:"
echo "  watch -n 1 nvidia-smi"
echo "  nvidia-smi dmon -s pucvmet"
echo ""
echo "Adjust human feedback:"
echo "  echo '0.85' > data/human_feedback.txt"
echo ""

exit $EXIT_CODE
