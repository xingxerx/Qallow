#!/bin/bash
# Install dependencies for Quantum Optimizer (AGI Evolution Feature 004)

set -e

echo "=========================================="
echo "Quantum Optimizer Dependency Installation"
echo "AGI Evolution Feature 004 - Task 1"
echo "=========================================="
echo ""

# Check if we're in a virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  Warning: Not in a virtual environment"
    echo "   Consider activating venv: source venv/bin/activate"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Installation cancelled."
        exit 1
    fi
fi

echo "Step 1: Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "   Python version: $PYTHON_VERSION"
echo ""

echo "Step 2: Upgrading pip..."
python3 -m pip install --upgrade pip
echo ""

echo "Step 3: Installing core dependencies..."
echo "   - numpy (scientific computing)"
python3 -m pip install "numpy>=1.24.0"
echo ""

echo "Step 4: Installing scikit-learn..."
echo "   - scikit-learn (Gaussian Process Regression)"
python3 -m pip install "scikit-learn>=1.3.0"
echo ""

echo "Step 5: Installing bayesian-optimization..."
echo "   - bayesian-optimization (Bayesian optimization engine)"
python3 -m pip install "bayesian-optimization>=1.4.0"
echo ""

echo "Step 6: Verifying installations..."
echo ""

# Verify numpy
if python3 -c "import numpy; print(f'✓ numpy {numpy.__version__}')" 2>/dev/null; then
    :
else
    echo "✗ numpy installation failed"
    exit 1
fi

# Verify scikit-learn
if python3 -c "import sklearn; print(f'✓ scikit-learn {sklearn.__version__}')" 2>/dev/null; then
    :
else
    echo "✗ scikit-learn installation failed"
    exit 1
fi

# Verify bayesian-optimization
if python3 -c "import bayes_opt; print(f'✓ bayesian-optimization installed')" 2>/dev/null; then
    :
else
    echo "✗ bayesian-optimization installation failed"
    exit 1
fi

echo ""
echo "=========================================="
echo "✓ All dependencies installed successfully!"
echo "=========================================="
echo ""
echo "You can now run the quantum optimizer demo:"
echo "   python3 examples/quantum_optimizer_demo.py"
echo ""
echo "Or import in your Python code:"
echo "   from python.quantum import QuantumCircuit, QuantumOptimizer"
echo ""

