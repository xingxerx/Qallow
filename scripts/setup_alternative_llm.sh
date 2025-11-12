#!/bin/bash
# Alternative LLM Setup - Compatible with 16GB VRAM
# Uses Qwen2.5 or Llama2 instead of Kimi-K2 for better compatibility

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  Alternative LLM Setup (Compatible with 16GB VRAM)        ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Choose model
MODEL_CHOICE="${1:-qwen}"
PORT="${2:-8000}"

case "$MODEL_CHOICE" in
    qwen)
        MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
        echo -e "${BLUE}Selected: Qwen2.5-7B (Recommended)${NC}"
        ;;
    llama)
        MODEL_NAME="meta-llama/Llama-2-7b-chat-hf"
        echo -e "${BLUE}Selected: Llama 2 7B${NC}"
        ;;
    mistral)
        MODEL_NAME="mistralai/Mistral-7B-Instruct-v0.2"
        echo -e "${BLUE}Selected: Mistral 7B${NC}"
        ;;
    *)
        echo -e "${RED}Unknown model: $MODEL_CHOICE${NC}"
        echo "Usage: $0 [qwen|llama|mistral] [port]"
        exit 1
        ;;
esac

echo ""
echo -e "${YELLOW}Configuration:${NC}"
echo "  Model:              $MODEL_NAME"
echo "  Port:               $PORT"
echo "  GPU Memory Util:    0.75"
echo "  Quantization:       fp8"
echo ""

# Check if vLLM is installed
if ! command -v vllm &> /dev/null; then
    echo -e "${YELLOW}vLLM not found. Installing...${NC}"
    pip install vllm>=0.10.0
fi

# Create data directories
mkdir -p data/llm
mkdir -p data/logs

echo -e "${BLUE}System Information:${NC}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "GPU info not available"
echo ""

echo -e "${YELLOW}Starting vLLM server...${NC}"
echo -e "${BLUE}Note: First run will download the model (~15-20GB). This may take 5-15 minutes.${NC}"
echo ""

# Start vLLM with compatible model
vllm serve "$MODEL_NAME" \
    --port "$PORT" \
    --served-model-name llm \
    --trust-remote-code \
    --gpu-memory-utilization 0.75 \
    --quantization fp8 \
    --max-model-len 4096 \
    --max-num-batched-tokens 4096 \
    --max-num-seqs 128 \
    --dtype bfloat16 \
    --disable-log-stats

echo -e "${GREEN}✓ vLLM server started on http://localhost:$PORT${NC}"
echo -e "${YELLOW}API Documentation: http://localhost:$PORT/docs${NC}"

