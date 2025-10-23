#!/bin/bash
# Cirq Quantum Algorithm Development Setup for Qallow

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_PATH="${PROJECT_ROOT}/venv"

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║     Cirq Quantum Algorithm Development Environment Setup       ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo

# Check if venv exists
if [[ ! -d "$VENV_PATH" ]]; then
    echo "❌ Virtual environment not found at $VENV_PATH"
    echo "Please run: python3 -m venv $VENV_PATH"
    exit 1
fi

echo "✅ Virtual environment found at $VENV_PATH"
echo

# Activate venv
source "$VENV_PATH/bin/activate"
echo "✅ Virtual environment activated"
echo

# Verify Cirq installation
echo "Verifying Cirq installation..."
python3 -c "import cirq; print(f'✅ Cirq version: {cirq.__version__}')"
python3 -c "import cirq_google; print('✅ Cirq-Google installed')"
echo

# Create directories
mkdir -p "$SCRIPT_DIR/algorithms"
mkdir -p "$SCRIPT_DIR/notebooks"
mkdir -p "$SCRIPT_DIR/results"

echo "✅ Created quantum algorithm directories:"
echo "   - $SCRIPT_DIR/algorithms"
echo "   - $SCRIPT_DIR/notebooks"
echo "   - $SCRIPT_DIR/results"
echo

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                    Setup Complete!                            ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo
echo "To activate the environment, run:"
echo "  source $VENV_PATH/bin/activate"
echo
echo "To start developing quantum algorithms:"
echo "  cd $SCRIPT_DIR"
echo "  python3 algorithms/hello_quantum.py"
echo

