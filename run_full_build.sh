#!/usr/bin/env bash
################################################################################
# Qallow Full Build Runner
#
# Runs entire project with CUDA + Cirq-Q + All Phases + Fast Agent
#
# Usage:
#   chmod +x run_full_build.sh
#   ./run_full_build.sh [OPTIONS]
#
# Options:
#   --no-agent          Don't start fast agent (just run phases)
#   --agent-only        Only run fast agent (no phases)
#   --phases-only       Only run unified phases (no agent)
#   --quick             Skip Phase 11 quantum bridge (faster)
#   --help              Show this help message
#
################################################################################

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Configuration
RUN_AGENT=true
RUN_PHASES=true
INCLUDE_PHASE11=true
MAX_ITERATIONS=500
AGENT_PID=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --no-agent)
            RUN_AGENT=false
            shift
            ;;
        --agent-only)
            RUN_PHASES=false
            shift
            ;;
        --phases-only)
            RUN_AGENT=false
            shift
            ;;
        --quick)
            INCLUDE_PHASE11=false
            shift
            ;;
        --help)
            echo "Usage: ./run_full_build.sh [OPTIONS]"
            echo "Options:"
            echo "  --no-agent         Don't start fast agent"
            echo "  --agent-only       Only run agent"
            echo "  --phases-only      Only run phases"
            echo "  --quick            Skip Phase 11 (faster)"
            echo "  --help             Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Header
echo -e "${BLUE}╔════════════════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║    Qallow Full Build - CUDA + Cirq-Q + All Phases + Fast Agent                ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# ============================================================================
# 1. Check Prerequisites
# ============================================================================
echo -e "${CYAN}[Setup]${NC} ${YELLOW}Checking prerequisites...${NC}"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}✗ Python 3 not found${NC}"
    exit 1
fi
python3 --version
echo -e "${GREEN}✓${NC} Python 3 found"

# Check CMake
if ! command -v cmake &> /dev/null; then
    echo -e "${RED}✗ CMake not found${NC}"
    exit 1
fi
cmake --version | head -1
echo -e "${GREEN}✓${NC} CMake found"

# Check CUDA (optional)
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✓${NC} NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1
    CUDA_AVAILABLE=true
else
    echo -e "${YELLOW}⚠${NC} No NVIDIA GPU detected (CPU fallback)"
    CUDA_AVAILABLE=false
fi

echo ""

# ============================================================================
# 2. Setup Environment
# ============================================================================
if [[ ! -d ".venv" ]]; then
    echo -e "${CYAN}[Setup]${NC} ${YELLOW}Setting up Python environment (first time)...${NC}"
    ./bootstrap.sh --cuda
else
    echo -e "${GREEN}✓${NC} Python environment already exists"
fi

echo ""
echo -e "${CYAN}[Setup]${NC} ${YELLOW}Activating virtual environment...${NC}"
source .venv/bin/activate
echo -e "${GREEN}✓${NC} Virtual environment activated"

echo ""

# ============================================================================
# 3. Set Environment Variables
# ============================================================================
echo -e "${CYAN}[Setup]${NC} ${YELLOW}Configuring environment variables...${NC}"

if [[ "$CUDA_AVAILABLE" == true ]]; then
    export QALLOW_ENABLE_CUDA=ON
    echo -e "${GREEN}✓${NC} CUDA enabled"
else
    export QALLOW_ENABLE_CUDA=OFF
    echo -e "${YELLOW}⚠${NC} CUDA disabled (using CPU)"
fi

export QALLOW_CIRQ=1
echo -e "${GREEN}✓${NC} Cirq-Q enabled"

export QALLOW_PROFILE_SCOPE=1
echo -e "${GREEN}✓${NC} Profiling enabled"

export QALLOW_LOG_LEVEL=INFO
echo -e "${GREEN}✓${NC} Log level set to INFO"

echo ""

# ============================================================================
# 4. Build Check
# ============================================================================
if [[ ! -f "build/qallow" ]]; then
    echo -e "${CYAN}[Build]${NC} ${YELLOW}Building project (first time)...${NC}"
    cmake -S . -B build -DQALLOW_ENABLE_CUDA=$QALLOW_ENABLE_CUDA
    cmake --build build --parallel $(nproc)
    echo -e "${GREEN}✓${NC} Build successful"
else
    echo -e "${GREEN}✓${NC} Binary already built (build/qallow)"
fi

echo ""

# ============================================================================
# 5. Run Unified Phases (Optional)
# ============================================================================
if [[ "$RUN_PHASES" == true ]]; then
    echo -e "${CYAN}[Phases]${NC} ${YELLOW}Running unified pipeline...${NC}"
    echo ""
    
    if [[ "$INCLUDE_PHASE11" == true ]]; then
        echo -e "${BLUE}Running: ./build/qallow run vm --integrate phase11${NC}"
        ./build/qallow run vm --integrate phase11
    else
        echo -e "${BLUE}Running: ./build/qallow run unified (Phase 11 skipped)${NC}"
        ./build/qallow run unified
    fi
    
    echo ""
    echo -e "${GREEN}✓${NC} Unified phases complete"
    echo -e "${GREEN}✓${NC} Results saved to: data/logs/"
    echo ""
fi

# ============================================================================
# 6. Start Fast Agent (Optional)
# ============================================================================
if [[ "$RUN_AGENT" == true ]]; then
    echo -e "${CYAN}[Agent]${NC} ${YELLOW}Starting fast agent in background...${NC}"
    echo ""
    
    # Check if agent script exists
    if [[ ! -f "lightning_agent_fast.py" ]]; then
        echo -e "${RED}✗ lightning_agent_fast.py not found${NC}"
        exit 1
    fi
    
    # Start agent in background
    python3 lightning_agent_fast.py \
        --fast \
        --use-cuda \
        --daemon \
        --max-iterations=$MAX_ITERATIONS > agent_daemon.log 2>&1 &
    
    AGENT_PID=$!
    
    # Wait for agent to start
    sleep 2
    
    if ps -p $AGENT_PID > /dev/null; then
        echo -e "${GREEN}✓${NC} Fast agent started (PID: $AGENT_PID)"
        echo -e "${GREEN}✓${NC} Max iterations: $MAX_ITERATIONS"
    else
        echo -e "${RED}✗ Agent failed to start${NC}"
        exit 1
    fi
    
    echo ""
    echo -e "${YELLOW}Agent is running in background.${NC}"
    echo -e "${YELLOW}Monitor with:${NC} tail -f agent_daemon.log"
    echo -e "${YELLOW}Stop with:${NC}   pkill -f 'lightning_agent_fast.py'"
    echo ""
fi

# ============================================================================
# 7. Summary
# ============================================================================
echo -e "${BLUE}╔════════════════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                         Summary                                               ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════════════════════╝${NC}"
echo ""

echo -e "${CYAN}Configuration:${NC}"
echo -e "  CUDA:      ${QALLOW_ENABLE_CUDA}"
echo -e "  Cirq-Q:    $QALLOW_CIRQ"
echo -e "  Profiling: $QALLOW_PROFILE_SCOPE"
echo ""

if [[ "$RUN_PHASES" == true ]]; then
    echo -e "${CYAN}Phases:${NC}"
    if [[ "$INCLUDE_PHASE11" == true ]]; then
        echo -e "  Phases 11-15: ${GREEN}COMPLETED${NC}"
    else
        echo -e "  Phases 12-15: ${GREEN}COMPLETED${NC}"
    fi
    echo -e "  Logs: data/logs/"
    echo ""
fi

if [[ "$RUN_AGENT" == true ]]; then
    echo -e "${CYAN}Fast Agent:${NC}"
    echo -e "  Status: ${GREEN}RUNNING${NC}"
    echo -e "  PID: $AGENT_PID"
    echo -e "  Log: agent_daemon.log"
    echo ""
fi

echo -e "${GREEN}✓ Full build complete!${NC}"
echo ""

# ============================================================================
# 8. Next Steps
# ============================================================================
if [[ "$RUN_AGENT" == true ]]; then
    echo -e "${CYAN}Next Steps:${NC}"
    echo -e "  1. Monitor agent: ${YELLOW}tail -f agent_daemon.log${NC}"
    echo -e "  2. Check iterations: ${YELLOW}grep 'Iteration' agent_daemon.log | wc -l${NC}"
    echo -e "  3. View commits: ${YELLOW}git log --oneline --author='Lightning Agent' | head -5${NC}"
    echo -e "  4. Stop when done: ${YELLOW}pkill -f 'lightning_agent_fast.py'${NC}"
else
    echo -e "${CYAN}Next Steps:${NC}"
    echo -e "  1. Review results: ${YELLOW}ls -lh data/logs/${NC}"
    echo -e "  2. Run tests: ${YELLOW}cd build && ctest --output-on-failure${NC}"
    echo -e "  3. Start agent: ${YELLOW}./run_full_build.sh --agent-only${NC}"
fi

echo ""
