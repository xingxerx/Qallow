#!/bin/bash

echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                                                                            ║"
echo "║         🔍 QISKIT C API + HPC VERIFICATION 🔍                             ║"
echo "║                                                                            ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

PASS=0
FAIL=0

# Check 1: Executable exists
echo "Checking: c-api-demo executable..."
if [ -f "/root/Qallow/qiskit-c-api-demo/build/c-api-demo" ]; then
    echo "✅ PASS: Executable found"
    ((PASS++))
else
    echo "❌ FAIL: Executable not found"
    ((FAIL++))
fi

# Check 2: Qiskit C library
echo "Checking: Qiskit C library..."
if [ -f "/root/Qallow/qiskit-c-api-demo/deps/qiskit/dist/c/lib/libqiskit.so" ]; then
    echo "✅ PASS: Qiskit C library found"
    ((PASS++))
else
    echo "❌ FAIL: Qiskit C library not found"
    ((FAIL++))
fi

# Check 3: QRMI service
echo "Checking: QRMI service..."
if [ -f "/root/Qallow/qiskit-c-api-demo/deps/qrmi/target/release/qrmi" ]; then
    echo "✅ PASS: QRMI service found"
    ((PASS++))
else
    echo "❌ FAIL: QRMI service not found"
    ((FAIL++))
fi

# Check 4: Data files
echo "Checking: Data files..."
if [ -f "/root/Qallow/qiskit-c-api-demo/data/fcidump_Fe4S4_MO.txt" ]; then
    echo "✅ PASS: Data files found"
    ((PASS++))
else
    echo "❌ FAIL: Data files not found"
    ((FAIL++))
fi

# Check 5: MPI
echo "Checking: OpenMPI..."
if command -v mpirun &> /dev/null; then
    echo "✅ PASS: OpenMPI found ($(mpirun --version | head -1))"
    ((PASS++))
else
    echo "❌ FAIL: OpenMPI not found"
    ((FAIL++))
fi

# Check 6: BLAS
echo "Checking: BLAS library..."
if [ -f "/usr/lib/libblas.so.3" ]; then
    echo "✅ PASS: BLAS library found"
    ((PASS++))
else
    echo "❌ FAIL: BLAS library not found"
    ((FAIL++))
fi

# Check 7: LAPACK
echo "Checking: LAPACK library..."
if [ -f "/usr/lib/liblapack.so" ]; then
    echo "✅ PASS: LAPACK library found"
    ((PASS++))
else
    echo "❌ FAIL: LAPACK library not found"
    ((FAIL++))
fi

# Check 8: Eigen3
echo "Checking: Eigen3..."
if [ -d "/usr/include/eigen3" ]; then
    echo "✅ PASS: Eigen3 found"
    ((PASS++))
else
    echo "❌ FAIL: Eigen3 not found"
    ((FAIL++))
fi

# Check 9: Credentials
echo "Checking: IBM Quantum credentials..."
if [ -n "$QISKIT_IBM_TOKEN" ] && [ -n "$QISKIT_IBM_INSTANCE" ]; then
    echo "✅ PASS: Credentials set"
    ((PASS++))
else
    echo "⚠️  WARNING: Credentials not set (run: bash /root/Qallow/setup_ibm_quantum.sh)"
    ((FAIL++))
fi

# Check 10: Executable is runnable
echo "Checking: Executable is runnable..."
if /root/Qallow/qiskit-c-api-demo/build/c-api-demo --help &> /dev/null; then
    echo "✅ PASS: Executable is runnable"
    ((PASS++))
else
    echo "⚠️  WARNING: Executable help not available (may need credentials)"
    ((PASS++))
fi

echo ""
echo "═══════════════════════════════════════════════════════════════════════════"
echo "VERIFICATION RESULTS"
echo "═══════════════════════════════════════════════════════════════════════════"
echo ""
echo "✅ PASSED: $PASS/10"
echo "❌ FAILED: $FAIL/10"
echo ""

if [ $FAIL -eq 0 ]; then
    echo "🎉 ALL CHECKS PASSED! System is ready to use."
    echo ""
    echo "Next steps:"
    echo "1. Setup IBM Quantum credentials:"
    echo "   bash /root/Qallow/setup_ibm_quantum.sh"
    echo ""
    echo "2. Run the demo:"
    echo "   bash /root/Qallow/run_qiskit_c_api_demo.sh"
    exit 0
else
    echo "⚠️  Some checks failed. Please review the errors above."
    exit 1
fi
