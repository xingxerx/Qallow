#!/bin/bash

###############################################################################
# Run Complete Quantum Workload with CUDA Acceleration and Error Correction
###############################################################################

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON_DIR="$PROJECT_ROOT/python"
LOGS_DIR="$PROJECT_ROOT/logs"
DATA_DIR="$PROJECT_ROOT/data"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Ensure directories exist
mkdir -p "$LOGS_DIR"
mkdir -p "$DATA_DIR/quantum_results"

# Activate virtual environment
if [ -f "$PROJECT_ROOT/qiskit-env/bin/activate" ]; then
    source "$PROJECT_ROOT/qiskit-env/bin/activate"
else
    echo -e "${RED}Virtual environment not found. Run setup_quantum_workload.sh first.${NC}"
    exit 1
fi

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Quantum Workload Execution${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check CUDA availability
echo -e "${YELLOW}[1/5] Checking CUDA availability...${NC}"
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep release | awk '{print $NF}')
    echo -e "${GREEN}✓ CUDA $CUDA_VERSION available${NC}"
    CUDA_ENABLED=true
else
    echo -e "${YELLOW}⚠ CUDA not available, using CPU${NC}"
    CUDA_ENABLED=false
fi

# Check Python dependencies
echo -e "${YELLOW}[2/5] Checking Python dependencies...${NC}"
python3 -c "import qiskit; print(f'Qiskit {qiskit.__version__}')" || {
    echo -e "${RED}Qiskit not installed. Run setup_quantum_workload.sh first.${NC}"
    exit 1
}

# Run CUDA benchmark (if CUDA available)
if [ "$CUDA_ENABLED" = true ]; then
    echo -e "${YELLOW}[3/5] Running CUDA quantum simulator benchmark...${NC}"
    python3 "$PYTHON_DIR/quantum_cuda_bridge.py" 2>&1 | tee "$LOGS_DIR/cuda_benchmark.log"
    echo -e "${GREEN}✓ CUDA benchmark complete${NC}"
else
    echo -e "${YELLOW}[3/5] Skipping CUDA benchmark (CUDA not available)${NC}"
fi

# Run main quantum workload
echo -e "${YELLOW}[4/5] Running IBM Quantum workload...${NC}"
python3 "$PYTHON_DIR/quantum_ibm_workload.py" 2>&1 | tee "$LOGS_DIR/quantum_workload.log"
echo -e "${GREEN}✓ Quantum workload complete${NC}"

# Run learning system analysis
echo -e "${YELLOW}[5/5] Running quantum learning system analysis...${NC}"
python3 "$PYTHON_DIR/quantum_learning_system.py" 2>&1 | tee "$LOGS_DIR/learning_system.log"
echo -e "${GREEN}✓ Learning system analysis complete${NC}"

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}✓ All quantum workloads completed successfully!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Display results summary
echo -e "${BLUE}Results Summary:${NC}"
echo "Logs:"
echo "  - CUDA Benchmark: $LOGS_DIR/cuda_benchmark.log"
echo "  - Quantum Workload: $LOGS_DIR/quantum_workload.log"
echo "  - Learning System: $LOGS_DIR/learning_system.log"
echo ""
echo "Data:"
echo "  - Quantum Results: $DATA_DIR/quantum_results/"
echo "  - CUDA Benchmark: $DATA_DIR/cuda_benchmark.json"
echo "  - Learning History: $DATA_DIR/quantum_learning_history_*.json"
echo ""
echo "Adaptive State:"
echo "  - $PROJECT_ROOT/adapt_state.json"
echo ""

# Show key metrics
if [ -f "$PROJECT_ROOT/adapt_state.json" ]; then
    echo -e "${BLUE}Current Adaptive State:${NC}"
    python3 -c "
import json
with open('$PROJECT_ROOT/adapt_state.json', 'r') as f:
    state = json.load(f)
    print(f\"  Learning Rate: {state.get('learning_rate', 'N/A')}\")
    print(f\"  Human Score: {state.get('human_score', 'N/A')}\")
    print(f\"  Iterations: {state.get('iterations', 'N/A')}\")
    print(f\"  Entanglement Score: {state.get('entanglement_score', 'N/A')}\")
    print(f\"  Error Correction: {state.get('error_correction_enabled', 'N/A')}\")
"
fi

echo ""
echo -e "${GREEN}Next steps:${NC}"
echo "1. Review logs in $LOGS_DIR/"
echo "2. Analyze results in $DATA_DIR/quantum_results/"
echo "3. Check adapt_state.json for learning progress"
echo "4. Run again to see adaptive improvements"

