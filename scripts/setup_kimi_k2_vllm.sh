#!/bin/bash
# Setup and run Kimi-K2 with vLLM
# Local inference without API keys
# Supports single GPU and multi-GPU setups

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  Kimi-K2 Setup with vLLM                                  ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"

# Configuration
MODEL_NAME="${1:-moonshotai/Kimi-K2-Instruct}"
PORT="${2:-8000}"
TENSOR_PARALLEL="${3:-1}"
GPU_MEMORY_UTIL="${4:-0.9}"

echo -e "${YELLOW}Configuration:${NC}"
echo "  Model:              $MODEL_NAME"
echo "  Port:               $PORT"
echo "  Tensor Parallel:    $TENSOR_PARALLEL"
echo "  GPU Memory Util:    $GPU_MEMORY_UTIL"
echo ""

# Check if vLLM is installed
if ! command -v vllm &> /dev/null; then
    echo -e "${YELLOW}vLLM not found. Installing...${NC}"
    pip install vllm>=0.10.0
fi

# Check if model exists locally
if [ -d "$MODEL_NAME" ]; then
    echo -e "${GREEN}✓ Using local model at: $MODEL_NAME${NC}"
    MODEL_PATH="$MODEL_NAME"
else
    echo -e "${YELLOW}Model will be downloaded from HuggingFace${NC}"
    MODEL_PATH="$MODEL_NAME"
fi

# Create data directories
mkdir -p data/kimi_k2
mkdir -p data/logs

echo -e "${YELLOW}Starting vLLM server...${NC}"
echo ""

# Start vLLM with Kimi-K2 configuration
vllm serve "$MODEL_PATH" \
    --port "$PORT" \
    --served-model-name kimi-k2 \
    --trust-remote-code \
    --tensor-parallel-size "$TENSOR_PARALLEL" \
    --gpu-memory-utilization "$GPU_MEMORY_UTIL" \
    --enable-auto-tool-choice \
    --tool-call-parser kimi_k2 \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 256

echo -e "${GREEN}✓ vLLM server started on http://localhost:$PORT${NC}"

