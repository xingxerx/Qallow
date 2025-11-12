#!/bin/bash
# Kimi-K2 Setup with vLLM - Optimized for 16GB VRAM
# Uses quantization and reduced context length for better memory efficiency

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  Kimi-K2 Setup with vLLM (Optimized for 16GB VRAM)        ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Configuration for 16GB VRAM
MODEL_NAME="${1:-moonshotai/Kimi-K2-Instruct}"
PORT="${2:-8000}"
TENSOR_PARALLEL="${3:-1}"
GPU_MEMORY_UTIL="${4:-0.65}"
QUANTIZATION="${5:-fp8}"
MAX_MODEL_LEN="${6:-4096}"
MAX_BATCH_TOKENS="${7:-4096}"

echo -e "${YELLOW}Configuration (Optimized for 16GB VRAM):${NC}"
echo "  Model:              $MODEL_NAME"
echo "  Port:               $PORT"
echo "  Tensor Parallel:    $TENSOR_PARALLEL"
echo "  GPU Memory Util:    $GPU_MEMORY_UTIL (conservative)"
echo "  Quantization:       $QUANTIZATION"
echo "  Max Model Len:      $MAX_MODEL_LEN (reduced from 131K)"
echo "  Max Batch Tokens:   $MAX_BATCH_TOKENS"
echo ""

# Check if vLLM is installed
if ! command -v vllm &> /dev/null; then
    echo -e "${YELLOW}vLLM not found. Installing...${NC}"
    pip install vllm>=0.10.0
fi

# Create data directories
mkdir -p data/kimi_k2
mkdir -p data/logs

echo -e "${BLUE}System Information:${NC}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "GPU info not available"
echo ""

echo -e "${YELLOW}Starting vLLM server with optimized settings...${NC}"
echo -e "${BLUE}Note: First run will download the model (~50GB). This may take 10-30 minutes.${NC}"
echo ""

# Start vLLM with optimized Kimi-K2 configuration
# Key optimizations:
# - fp8 quantization: Reduces model size by ~75%
# - max-model-len: Reduced from 131K to 4K for memory efficiency
# - gpu-memory-utilization: Conservative 0.65 to avoid OOM
# - max-num-batched-tokens: Reduced for stability
vllm serve "$MODEL_NAME" \
    --port "$PORT" \
    --served-model-name kimi-k2 \
    --trust-remote-code \
    --tensor-parallel-size "$TENSOR_PARALLEL" \
    --gpu-memory-utilization "$GPU_MEMORY_UTIL" \
    --enable-auto-tool-choice \
    --tool-call-parser kimi_k2 \
    --quantization "$QUANTIZATION" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-batched-tokens "$MAX_BATCH_TOKENS" \
    --max-num-seqs 128 \
    --dtype bfloat16 \
    --disable-log-stats \
    --disable-frontend

echo -e "${GREEN}✓ vLLM server started on http://localhost:$PORT${NC}"
echo -e "${YELLOW}API Documentation: http://localhost:$PORT/docs${NC}"

