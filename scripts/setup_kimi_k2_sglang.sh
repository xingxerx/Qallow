#!/bin/bash
# Setup and run Kimi-K2 with SGLang
# Local inference without API keys
# Supports single GPU and multi-GPU setups

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  Kimi-K2 Setup with SGLang                                ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"

# Configuration
MODEL_NAME="${1:-moonshotai/Kimi-K2-Instruct}"
PORT="${2:-30000}"
TP_SIZE="${3:-1}"
MASTER_IP="${4:-127.0.0.1}"

echo -e "${YELLOW}Configuration:${NC}"
echo "  Model:              $MODEL_NAME"
echo "  Port:               $PORT"
echo "  Tensor Parallel:    $TP_SIZE"
echo "  Master IP:          $MASTER_IP"
echo ""

# Check if SGLang is installed
if ! python -c "import sglang" 2>/dev/null; then
    echo -e "${YELLOW}SGLang not found. Installing...${NC}"
    pip install sglang
fi

# Create data directories
mkdir -p data/kimi_k2
mkdir -p data/logs

echo -e "${YELLOW}Starting SGLang server...${NC}"
echo ""

# Start SGLang with Kimi-K2 configuration
python -m sglang.launch_server \
    --model-path "$MODEL_NAME" \
    --tp "$TP_SIZE" \
    --port "$PORT" \
    --trust-remote-code \
    --tool-call-parser kimi_k2 \
    --max-running-requests 1024

echo -e "${GREEN}✓ SGLang server started on http://localhost:$PORT${NC}"

