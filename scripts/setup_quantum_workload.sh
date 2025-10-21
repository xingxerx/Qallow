#!/bin/bash

###############################################################################
# IBM Quantum Platform Workload Setup with CUDA Acceleration
# Comprehensive setup for quantum computing with error correction
###############################################################################

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON_DIR="$PROJECT_ROOT/python"
DATA_DIR="$PROJECT_ROOT/data"
LOGS_DIR="$PROJECT_ROOT/logs"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}IBM Quantum Workload Setup${NC}"
echo -e "${BLUE}========================================${NC}"

# Create necessary directories
echo -e "${YELLOW}[1/6] Creating directories...${NC}"
mkdir -p "$DATA_DIR/quantum_results"
mkdir -p "$LOGS_DIR"
mkdir -p "$PROJECT_ROOT/qiskit-env"

# Check Python version
echo -e "${YELLOW}[2/6] Checking Python environment...${NC}"
python3 --version

# Create virtual environment if it doesn't exist
if [ ! -d "$PROJECT_ROOT/qiskit-env" ] || [ ! -f "$PROJECT_ROOT/qiskit-env/bin/activate" ]; then
    echo -e "${YELLOW}Creating Python virtual environment...${NC}"
    python3 -m venv "$PROJECT_ROOT/qiskit-env"
fi

# Activate virtual environment
source "$PROJECT_ROOT/qiskit-env/bin/activate"

# Upgrade pip
echo -e "${YELLOW}[3/6] Upgrading pip...${NC}"
pip install --upgrade pip setuptools wheel

# Install Qiskit and dependencies
echo -e "${YELLOW}[4/6] Installing Qiskit and dependencies...${NC}"
pip install \
    qiskit==1.0.0 \
    qiskit-ibm-runtime==0.20.0 \
    'qiskit[visualization]' \
    matplotlib \
    numpy \
    scipy \
    pandas \
    jupyter \
    ipython

# Try to install qiskit-aer (optional, may fail on some systems)
echo -e "${YELLOW}Attempting to install qiskit-aer (optional)...${NC}"
pip install qiskit-aer==0.13.0 2>/dev/null || echo -e "${YELLOW}⚠ qiskit-aer installation skipped (using simulator instead)${NC}"

# Install CUDA-related packages (if CUDA is available)
echo -e "${YELLOW}[5/6] Checking CUDA availability...${NC}"
if command -v nvcc &> /dev/null; then
    echo -e "${GREEN}CUDA found: $(nvcc --version | grep release)${NC}"
    pip install cupy-cuda11x  # Adjust cuda version as needed
else
    echo -e "${YELLOW}CUDA not found, skipping cupy installation${NC}"
fi

# Create requirements file
echo -e "${YELLOW}[6/6] Creating requirements file...${NC}"
cat > "$PYTHON_DIR/requirements-quantum.txt" << 'EOF'
qiskit==1.0.0
qiskit-ibm-runtime==0.20.0
qiskit-aer==0.13.0
qiskit[visualization]
matplotlib>=3.5.0
numpy>=1.21.0
scipy>=1.7.0
pandas>=1.3.0
jupyter>=1.0.0
ipython>=7.0.0
EOF

echo -e "${GREEN}✓ Setup complete!${NC}"
echo ""
echo -e "${BLUE}Next steps:${NC}"
echo "1. Activate environment: source $PROJECT_ROOT/qiskit-env/bin/activate"
echo "2. Set IBM Quantum credentials (optional):"
echo "   python3 -c \"from qiskit_ibm_runtime import QiskitRuntimeService; QiskitRuntimeService.save_account(channel='ibm_cloud', token='YOUR_TOKEN', instance='YOUR_CRN')\""
echo "3. Run quantum workload: python3 $PYTHON_DIR/quantum_ibm_workload.py"
echo "4. Run CUDA benchmark: python3 $PYTHON_DIR/quantum_cuda_bridge.py"
echo ""
echo -e "${BLUE}Documentation:${NC}"
echo "- IBM Quantum: https://quantum.cloud.ibm.com"
echo "- Qiskit: https://qiskit.org"
echo "- CUDA: https://developer.nvidia.com/cuda-toolkit"

