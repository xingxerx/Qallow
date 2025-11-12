#!/bin/bash
#
# Ollama Setup for Supercomputer/Multi-GPU Systems
# Supports: Distributed inference with Ray, MPI, and multi-node clusters
# Models: Llama2-70B, DeepSeek-V3, and other large models
#
# Usage:
#   ./scripts/setup_ollama_supercomputer.sh [OPTIONS]
#
# Options:
#   --model MODEL       Model to pull (default: llama2:70b)
#   --num-gpu N         Number of GPUs (default: 8)
#   --distributed       Enable distributed setup with Ray
#   --mpi               Enable MPI for multi-node
#   --head-node         This is the head node (for distributed)
#   --worker-node ADDR  This is a worker node, connect to ADDR
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default configuration
MODEL="llama2:70b"
NUM_GPU=8
DISTRIBUTED=false
MPI=false
HEAD_NODE=false
WORKER_NODE=""
OLLAMA_HOST="http://localhost:11434"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --num-gpu)
            NUM_GPU="$2"
            shift 2
            ;;
        --distributed)
            DISTRIBUTED=true
            shift
            ;;
        --mpi)
            MPI=true
            shift
            ;;
        --head-node)
            HEAD_NODE=true
            shift
            ;;
        --worker-node)
            WORKER_NODE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  Qallow Ollama Supercomputer Setup                        ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}Configuration:${NC}"
echo -e "  Model:        ${YELLOW}${MODEL}${NC}"
echo -e "  GPUs:         ${YELLOW}${NUM_GPU}${NC}"
echo -e "  Distributed:  ${YELLOW}${DISTRIBUTED}${NC}"
echo -e "  MPI:          ${YELLOW}${MPI}${NC}"
echo -e "  Head Node:    ${YELLOW}${HEAD_NODE}${NC}"
echo -e "  Worker Node:  ${YELLOW}${WORKER_NODE:-N/A}${NC}"
echo ""

# Check if running on CUDA-capable system
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}✗ nvidia-smi not found. CUDA not available.${NC}"
    echo -e "${YELLOW}  This script is designed for NVIDIA GPU systems.${NC}"
    exit 1
fi

# Display GPU info
echo -e "${GREEN}GPU Information:${NC}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | head -n $NUM_GPU
echo ""

# Step 1: Install Ollama
echo -e "${BLUE}[1/6] Installing Ollama...${NC}"
if command -v ollama &> /dev/null; then
    echo -e "${GREEN}✓ Ollama already installed${NC}"
    ollama --version
else
    echo -e "${YELLOW}Installing Ollama...${NC}"
    curl -fsSL https://ollama.com/install.sh | sh
    echo -e "${GREEN}✓ Ollama installed${NC}"
fi
echo ""

# Step 2: Configure Ollama for multi-GPU
echo -e "${BLUE}[2/6] Configuring Ollama for ${NUM_GPU} GPUs...${NC}"

# Set environment variables
export OLLAMA_NUM_GPU=$NUM_GPU
export OLLAMA_HOST=$OLLAMA_HOST

# For large models, increase context size
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_NUM_PARALLEL=1

# Create systemd override for persistent config (if systemd is available)
if command -v systemctl &> /dev/null; then
    echo -e "${YELLOW}Creating systemd override...${NC}"
    sudo mkdir -p /etc/systemd/system/ollama.service.d/
    sudo tee /etc/systemd/system/ollama.service.d/override.conf > /dev/null <<EOF
[Service]
Environment="OLLAMA_NUM_GPU=${NUM_GPU}"
Environment="OLLAMA_HOST=${OLLAMA_HOST}"
Environment="OLLAMA_MAX_LOADED_MODELS=1"
EOF
    sudo systemctl daemon-reload
    echo -e "${GREEN}✓ Systemd override created${NC}"
fi
echo ""

# Step 3: Start Ollama service
echo -e "${BLUE}[3/6] Starting Ollama service...${NC}"

# Check if already running
if curl -s $OLLAMA_HOST/api/tags > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Ollama already running${NC}"
else
    if command -v systemctl &> /dev/null; then
        sudo systemctl start ollama
        sleep 3
    else
        # Start in background
        ollama serve > /tmp/ollama.log 2>&1 &
        OLLAMA_PID=$!
        echo $OLLAMA_PID > /tmp/ollama.pid
        sleep 5
    fi
    
    # Verify
    if curl -s $OLLAMA_HOST/api/tags > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Ollama service started${NC}"
    else
        echo -e "${RED}✗ Failed to start Ollama${NC}"
        exit 1
    fi
fi
echo ""

# Step 4: Pull model
echo -e "${BLUE}[4/6] Pulling model: ${MODEL}...${NC}"
echo -e "${YELLOW}This may take a while for large models (70B+ params)${NC}"

if ollama list | grep -q "$MODEL"; then
    echo -e "${GREEN}✓ Model ${MODEL} already available${NC}"
else
    echo -e "${YELLOW}Downloading ${MODEL}...${NC}"
    ollama pull $MODEL
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Model ${MODEL} downloaded successfully${NC}"
    else
        echo -e "${RED}✗ Failed to download model${NC}"
        exit 1
    fi
fi
echo ""

# Step 5: Distributed setup (optional)
if [ "$DISTRIBUTED" = true ]; then
    echo -e "${BLUE}[5/6] Setting up distributed inference with Ray...${NC}"
    
    # Install Ray if not present
    if ! python3 -c "import ray" 2>/dev/null; then
        echo -e "${YELLOW}Installing Ray...${NC}"
        pip install -q "ray[default]" "ray[serve]"
        echo -e "${GREEN}✓ Ray installed${NC}"
    else
        echo -e "${GREEN}✓ Ray already installed${NC}"
    fi
    
    if [ "$HEAD_NODE" = true ]; then
        echo -e "${YELLOW}Starting Ray head node...${NC}"
        ray start --head --num-gpus=$NUM_GPU --dashboard-host=0.0.0.0
        
        # Get head node address
        HEAD_ADDR=$(ray status | grep "Ray runtime started" -A 1 | tail -n 1 | awk '{print $NF}')
        echo -e "${GREEN}✓ Ray head node started${NC}"
        echo -e "${BLUE}Head node address: ${HEAD_ADDR}${NC}"
        echo -e "${YELLOW}On worker nodes, run:${NC}"
        echo -e "  ./scripts/setup_ollama_supercomputer.sh --worker-node ${HEAD_ADDR} --num-gpu N"
        
    elif [ -n "$WORKER_NODE" ]; then
        echo -e "${YELLOW}Connecting to head node: ${WORKER_NODE}${NC}"
        ray start --address=$WORKER_NODE --num-gpus=$NUM_GPU
        echo -e "${GREEN}✓ Ray worker node connected${NC}"
    fi
else
    echo -e "${BLUE}[5/6] Skipping distributed setup${NC}"
fi
echo ""

# Step 6: Test inference
echo -e "${BLUE}[6/6] Testing inference...${NC}"

TEST_PROMPT="You are a quantum computing expert. In one sentence, explain QAOA."

echo -e "${YELLOW}Sending test prompt to ${MODEL}...${NC}"

RESPONSE=$(curl -s $OLLAMA_HOST/api/generate -d "{
  \"model\": \"$MODEL\",
  \"prompt\": \"$TEST_PROMPT\",
  \"stream\": false,
  \"options\": {
    \"num_gpu\": $NUM_GPU,
    \"temperature\": 0.3
  }
}" | python3 -c "import sys, json; print(json.load(sys.stdin).get('response', 'ERROR'))")

if [ -n "$RESPONSE" ] && [ "$RESPONSE" != "ERROR" ]; then
    echo -e "${GREEN}✓ Inference test successful${NC}"
    echo -e "${BLUE}Response:${NC} ${RESPONSE:0:200}..."
else
    echo -e "${RED}✗ Inference test failed${NC}"
    exit 1
fi
echo ""

# Summary
echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  Setup Complete!                                          ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BLUE}Ollama Configuration:${NC}"
echo -e "  Host:         ${OLLAMA_HOST}"
echo -e "  Model:        ${MODEL}"
echo -e "  GPUs:         ${NUM_GPU}"
echo -e "  Status:       ${GREEN}Running${NC}"
echo ""
echo -e "${BLUE}Next Steps:${NC}"
echo ""
echo -e "1. Test the Qallow agent:"
echo -e "   ${YELLOW}python3 -m python.agents.qallow_agent_ollama --model ${MODEL} --num-gpu ${NUM_GPU} --task qaoa_optimize${NC}"
echo ""
echo -e "2. Start the chat server:"
echo -e "   ${YELLOW}cd python/chat_server && uvicorn main:app --host 0.0.0.0 --port 8008${NC}"
echo ""
echo -e "3. Run Phase 14 with agent:"
echo -e "   ${YELLOW}./build/qallow phase 14 --agent-ollama${NC}"
echo ""
echo -e "4. Monitor GPU usage:"
echo -e "   ${YELLOW}watch -n 1 nvidia-smi${NC}"
echo ""

if [ "$DISTRIBUTED" = true ] && [ "$HEAD_NODE" = true ]; then
    echo -e "${BLUE}Distributed Setup:${NC}"
    echo -e "  Ray dashboard: ${YELLOW}http://localhost:8265${NC}"
    echo -e "  Ray status:    ${YELLOW}ray status${NC}"
    echo ""
fi

echo -e "${GREEN}For more information, see:${NC}"
echo -e "  - ${BLUE}docs/DEEPSEEK_INTEGRATION.md${NC}"
echo -e "  - ${BLUE}python/agents/qallow_agent_ollama.py${NC}"
echo ""

