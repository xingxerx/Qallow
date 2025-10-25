#!/bin/bash

# Qiskit C API + HPC Demo Runner
# This script runs the quantum chemistry demo on real IBM Quantum hardware

set -e

echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                                                                            ║"
echo "║         🚀 QISKIT C API + HPC DEMO - REAL QUANTUM HARDWARE 🚀             ║"
echo "║                                                                            ║"
echo "║              Sample-based Quantum Diagonalization (SQD)                    ║"
echo "║                                                                            ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Check if credentials are set
if [ -z "$QISKIT_IBM_TOKEN" ]; then
    echo "❌ ERROR: QISKIT_IBM_TOKEN not set"
    echo ""
    echo "Please set your IBM Quantum credentials:"
    echo "  export QISKIT_IBM_TOKEN='your_api_key'"
    echo "  export QISKIT_IBM_INSTANCE='your_crn'"
    echo ""
    echo "Get credentials at: https://quantum.ibm.com/"
    exit 1
fi

if [ -z "$QISKIT_IBM_INSTANCE" ]; then
    echo "❌ ERROR: QISKIT_IBM_INSTANCE not set"
    echo ""
    echo "Please set your IBM Quantum credentials:"
    echo "  export QISKIT_IBM_TOKEN='your_api_key'"
    echo "  export QISKIT_IBM_INSTANCE='your_crn'"
    echo ""
    echo "Get credentials at: https://quantum.ibm.com/"
    exit 1
fi

echo "✅ Credentials found"
echo "   Token: ${QISKIT_IBM_TOKEN:0:20}..."
echo "   Instance: $QISKIT_IBM_INSTANCE"
echo ""

# Check if executable exists
DEMO_PATH="/root/Qallow/qiskit-c-api-demo/build/c-api-demo"
if [ ! -f "$DEMO_PATH" ]; then
    echo "❌ ERROR: c-api-demo executable not found at $DEMO_PATH"
    echo ""
    echo "Please run the installation first:"
    echo "  bash /root/Qallow/QISKIT_C_API_HPC_SETUP.sh"
    exit 1
fi

echo "✅ Executable found: $DEMO_PATH"
echo ""

# Ask for execution mode
echo "═══════════════════════════════════════════════════════════════════════════"
echo "SELECT EXECUTION MODE:"
echo "═══════════════════════════════════════════════════════════════════════════"
echo ""
echo "1) Single Process (Local)"
echo "2) Distributed (MPI - 4 processes)"
echo "3) Distributed (MPI - 8 processes)"
echo "4) Distributed (MPI - 16 processes)"
echo "5) Distributed (MPI - 96 processes)"
echo ""
read -p "Enter choice (1-5): " choice

cd /root/Qallow/qiskit-c-api-demo/build

case $choice in
    1)
        echo ""
        echo "🚀 Running single process..."
        echo ""
        ./c-api-demo \
          --fcidump ../data/fcidump_Fe4S4_MO.txt \
          -v \
          --tolerance 1.0e-3 \
          --max_time 600 \
          --recovery 1 \
          --number_of_samples 300 \
          --num_shots 1000 \
          --backend_name ibm_kingston
        ;;
    2)
        echo ""
        echo "🚀 Running distributed (4 processes)..."
        echo ""
        mpirun -np 4 ./c-api-demo \
          --fcidump ../data/fcidump_Fe4S4_MO.txt \
          -v \
          --tolerance 1.0e-3 \
          --max_time 600 \
          --recovery 1 \
          --number_of_samples 600 \
          --num_shots 2000 \
          --backend_name ibm_kingston
        ;;
    3)
        echo ""
        echo "🚀 Running distributed (8 processes)..."
        echo ""
        mpirun -np 8 ./c-api-demo \
          --fcidump ../data/fcidump_Fe4S4_MO.txt \
          -v \
          --tolerance 1.0e-3 \
          --max_time 600 \
          --recovery 1 \
          --number_of_samples 1000 \
          --num_shots 4000 \
          --backend_name ibm_kingston
        ;;
    4)
        echo ""
        echo "🚀 Running distributed (16 processes)..."
        echo ""
        mpirun -np 16 ./c-api-demo \
          --fcidump ../data/fcidump_Fe4S4_MO.txt \
          -v \
          --tolerance 1.0e-3 \
          --max_time 600 \
          --recovery 1 \
          --number_of_samples 1500 \
          --num_shots 6000 \
          --backend_name ibm_kingston
        ;;
    5)
        echo ""
        echo "🚀 Running distributed (96 processes)..."
        echo ""
        mpirun -np 96 ./c-api-demo \
          --fcidump ../data/fcidump_Fe4S4_MO.txt \
          -v \
          --tolerance 1.0e-3 \
          --max_time 600 \
          --recovery 1 \
          --number_of_samples 2000 \
          --num_shots 10000 \
          --backend_name ibm_kingston
        ;;
    *)
        echo "❌ Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "✅ Demo completed!"
echo ""

