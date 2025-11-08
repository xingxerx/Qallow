#!/bin/bash
#
# Quick Start Script for Qallow Ollama Agent
# Sets up Ollama and runs a test optimization
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  Qallow Ollama Agent - Quick Start                        ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Step 1: Check if Ollama is installed
echo -e "${BLUE}[1/5] Checking Ollama installation...${NC}"
if command -v ollama &> /dev/null; then
    echo -e "${GREEN}✓ Ollama is installed${NC}"
    ollama --version
else
    echo -e "${YELLOW}Ollama not found. Installing...${NC}"
    curl -fsSL https://ollama.com/install.sh | sh
    echo -e "${GREEN}✓ Ollama installed${NC}"
fi
echo ""

# Step 2: Start Ollama service
echo -e "${BLUE}[2/5] Starting Ollama service...${NC}"
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Ollama already running${NC}"
else
    echo -e "${YELLOW}Starting Ollama...${NC}"
    ollama serve > /tmp/ollama.log 2>&1 &
    OLLAMA_PID=$!
    echo $OLLAMA_PID > /tmp/ollama.pid
    sleep 3
    
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Ollama started (PID: $OLLAMA_PID)${NC}"
    else
        echo -e "${RED}✗ Failed to start Ollama${NC}"
        exit 1
    fi
fi
echo ""

# Step 3: Pull model
echo -e "${BLUE}[3/5] Checking for model...${NC}"
MODEL="llama2:13b"  # Use 13B for quick start

if ollama list | grep -q "$MODEL"; then
    echo -e "${GREEN}✓ Model $MODEL already available${NC}"
else
    echo -e "${YELLOW}Pulling $MODEL (this may take a few minutes)...${NC}"
    ollama pull $MODEL
    echo -e "${GREEN}✓ Model downloaded${NC}"
fi
echo ""

# Step 4: Test agent
echo -e "${BLUE}[4/5] Testing Ollama agent...${NC}"
echo -e "${YELLOW}Running QAOA optimization with 16 nodes...${NC}"

python3 -m python.agents.qallow_agent_ollama \
  --task qaoa_optimize \
  --model $MODEL \
  --nodes 16 \
  --target 0.95 \
  --no-ethics

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Agent test successful${NC}"
else
    echo -e "${RED}✗ Agent test failed${NC}"
    exit 1
fi
echo ""

# Step 5: Show next steps
echo -e "${BLUE}[5/5] Setup complete!${NC}"
echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  Success! Ollama Agent is ready                          ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BLUE}Next Steps:${NC}"
echo ""
echo -e "1. Run Phase 14 with Ollama agent:"
echo -e "   ${YELLOW}./build/qallow phase 14 --nodes=256 --target_fidelity=0.981 --agent-ollama${NC}"
echo ""
echo -e "2. Start chat server:"
echo -e "   ${YELLOW}export QALLOW_CHAT_BACKEND=ollama${NC}"
echo -e "   ${YELLOW}cd python/chat_server && uvicorn main:app --host 0.0.0.0 --port 8008${NC}"
echo ""
echo -e "3. Test chat API:"
echo -e "   ${YELLOW}curl -X POST http://localhost:8008/chat -H 'Content-Type: application/json' -d '{\"message\": \"Hello\", \"backend\": \"ollama\"}'${NC}"
echo ""
echo -e "4. For larger models (70B), use:"
echo -e "   ${YELLOW}ollama pull llama2:70b${NC}"
echo -e "   ${YELLOW}./build/qallow phase 14 --agent-ollama --ollama-model=llama2:70b${NC}"
echo ""
echo -e "${BLUE}Output Files:${NC}"
echo -e "  - Agent log:  ${YELLOW}data/quantum/agent_output.jsonl${NC}"
echo -e "  - Gain JSON:  ${YELLOW}data/quantum/ollama_gain.json${NC}"
echo ""
echo -e "${BLUE}Documentation:${NC}"
echo -e "  - Full guide: ${YELLOW}docs/OLLAMA_AGENT_GUIDE.md${NC}"
echo -e "  - API docs:   ${YELLOW}http://localhost:8008/docs${NC} (when server running)"
echo ""
echo -e "${GREEN}Happy optimizing! 🚀${NC}"
echo ""

