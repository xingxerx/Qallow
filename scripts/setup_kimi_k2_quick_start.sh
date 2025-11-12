#!/bin/bash
# Quick start script for Kimi-K2 integration
# Sets up everything needed to run Kimi-K2 locally

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  Kimi-K2 Quick Start Setup                                ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Step 1: Check Python
echo -e "${YELLOW}[1/5] Checking Python installation...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}✗ Python 3 not found${NC}"
    exit 1
fi
PYTHON_VERSION=$(python3 --version | awk '{print $2}')
echo -e "${GREEN}✓ Python $PYTHON_VERSION found${NC}"
echo ""

# Step 2: Install dependencies
echo -e "${YELLOW}[2/5] Installing Kimi-K2 dependencies...${NC}"
pip install -q openai>=1.0.0
pip install -q vllm>=0.10.0
echo -e "${GREEN}✓ Dependencies installed${NC}"
echo ""

# Step 3: Create directories
echo -e "${YELLOW}[3/5] Creating data directories...${NC}"
mkdir -p data/kimi_k2
mkdir -p data/logs
mkdir -p config
echo -e "${GREEN}✓ Directories created${NC}"
echo ""

# Step 4: Check CUDA (optional)
echo -e "${YELLOW}[4/5] Checking GPU availability...${NC}"
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    echo -e "${GREEN}✓ Found $GPU_COUNT GPU(s)${NC}"
else
    echo -e "${YELLOW}⚠ No NVIDIA GPU detected (CPU inference will be slow)${NC}"
fi
echo ""

# Step 5: Display next steps
echo -e "${YELLOW}[5/5] Setup complete!${NC}"
echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  Next Steps                                               ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BLUE}Option 1: Start vLLM server (recommended)${NC}"
echo "  bash scripts/setup_kimi_k2_vllm.sh"
echo ""
echo -e "${BLUE}Option 2: Start SGLang server${NC}"
echo "  bash scripts/setup_kimi_k2_sglang.sh"
echo ""
echo -e "${BLUE}Option 3: Start chat server (requires vLLM running)${NC}"
echo "  export QALLOW_CHAT_BACKEND=kimi_k2"
echo "  export KIMI_K2_BASE_URL=http://localhost:8000/v1"
echo "  cd python/chat_server"
echo "  uvicorn main:app --host 0.0.0.0 --port 8008"
echo ""
echo -e "${BLUE}Option 4: Test with Python${NC}"
echo "  python3 -c \"from python.agents.kimi_k2_agent import create_kimi_k2_agent; agent = create_kimi_k2_agent(); print(agent.chat('Hello!'))\""
echo ""
echo -e "${GREEN}Documentation:${NC}"
echo "  - Config: config/kimi_k2.yaml"
echo "  - Agent: python/agents/kimi_k2_agent.py"
echo "  - Chat Server: python/chat_server/main.py"
echo ""

