#!/bin/bash
#
# Simple Setup Script for Qallow Ollama Agent
# No sudo required - guides you through manual steps
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  Qallow Ollama Agent - Simple Setup                      ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check we're in the right directory
if [ ! -f "python/agents/qallow_agent_ollama.py" ]; then
    echo -e "${RED}Error: Must run from Qallow root directory${NC}"
    echo -e "${YELLOW}Run: cd ~/Qallow && ./setup_simple.sh${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Running from correct directory: $(pwd)${NC}"
echo ""

# Step 1: Check Ollama
echo -e "${BLUE}[1/5] Checking Ollama installation...${NC}"
if command -v ollama &> /dev/null; then
    echo -e "${GREEN}✓ Ollama is installed${NC}"
    ollama --version
else
    echo -e "${YELLOW}⚠ Ollama not found${NC}"
    echo ""
    echo -e "${BLUE}Please install Ollama manually:${NC}"
    echo -e "${YELLOW}curl -fsSL https://ollama.com/install.sh | sh${NC}"
    echo ""
    echo -e "Then run this script again."
    exit 1
fi
echo ""

# Step 2: Check if Ollama is running
echo -e "${BLUE}[2/5] Checking Ollama service...${NC}"
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Ollama is running${NC}"
else
    echo -e "${YELLOW}⚠ Ollama not running. Starting...${NC}"
    ollama serve > /tmp/ollama.log 2>&1 &
    OLLAMA_PID=$!
    echo $OLLAMA_PID > /tmp/ollama.pid
    sleep 3
    
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Ollama started (PID: $OLLAMA_PID)${NC}"
    else
        echo -e "${RED}✗ Failed to start Ollama${NC}"
        echo -e "${YELLOW}Try manually: ollama serve &${NC}"
        exit 1
    fi
fi
echo ""

# Step 3: Check for model
echo -e "${BLUE}[3/5] Checking for models...${NC}"
MODEL="llama2:7b"

if ollama list | grep -q "$MODEL"; then
    echo -e "${GREEN}✓ Model $MODEL is available${NC}"
else
    echo -e "${YELLOW}⚠ Model $MODEL not found${NC}"
    echo ""
    echo -e "${BLUE}Pulling model (this may take 5-10 minutes)...${NC}"
    ollama pull $MODEL
    echo -e "${GREEN}✓ Model downloaded${NC}"
fi
echo ""

# Step 4: Test Python module
echo -e "${BLUE}[4/5] Testing Python module...${NC}"
export PYTHONPATH=$(pwd)

if python3 -c "from python.agents.qallow_agent_ollama import OllamaAgent; print('OK')" 2>/dev/null; then
    echo -e "${GREEN}✓ Python module imports successfully${NC}"
else
    echo -e "${RED}✗ Python module import failed${NC}"
    echo -e "${YELLOW}Checking Python dependencies...${NC}"
    
    # Check if requirements file exists
    if [ -f "config/requirements.txt" ]; then
        echo -e "${BLUE}Installing dependencies...${NC}"
        pip3 install -r config/requirements.txt --user
    fi
fi
echo ""

# Step 5: Run test
echo -e "${BLUE}[5/5] Testing agent...${NC}"
echo -e "${YELLOW}Running QAOA optimization with 16 nodes (quick test)...${NC}"
echo ""

python3 -m python.agents.qallow_agent_ollama \
  --task qaoa_optimize \
  --model $MODEL \
  --nodes 16 \
  --target 0.95 \
  --no-ethics

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ Agent test successful!${NC}"
else
    echo ""
    echo -e "${RED}✗ Agent test failed${NC}"
    echo -e "${YELLOW}Check logs above for errors${NC}"
    exit 1
fi
echo ""

# Success message
echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  Setup Complete! 🎉                                       ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BLUE}Next Steps:${NC}"
echo ""
echo -e "1. Run Phase 14 with agent:"
echo -e "   ${YELLOW}./build/qallow phase 14 --agent-ollama --ollama-model=$MODEL${NC}"
echo ""
echo -e "2. Try a larger model (better quality):"
echo -e "   ${YELLOW}ollama pull llama2:13b${NC}"
echo -e "   ${YELLOW}./build/qallow phase 14 --agent-ollama --ollama-model=llama2:13b${NC}"
echo ""
echo -e "3. Start chat server:"
echo -e "   ${YELLOW}export QALLOW_CHAT_BACKEND=ollama${NC}"
echo -e "   ${YELLOW}cd python/chat_server && uvicorn main:app --port 8008${NC}"
echo ""
echo -e "4. Read the docs:"
echo -e "   ${YELLOW}cat HOW_TO_RUN.md${NC}"
echo -e "   ${YELLOW}cat MANUAL_SETUP.md${NC}"
echo ""
echo -e "${GREEN}Happy optimizing! 🚀${NC}"
echo ""

