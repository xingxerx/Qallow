#!/bin/bash
# Qallow Health Check Script
# Verifies all build targets and dependencies are working correctly

set +e  # Don't exit on errors, we want to check all items

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Counters
PASSED=0
FAILED=0
WARNINGS=0

# Helper functions
check_pass() {
    echo -e "${GREEN}✅${NC} $1"
    ((PASSED++))
}

check_fail() {
    echo -e "${RED}❌${NC} $1"
    ((FAILED++))
}

check_warn() {
    echo -e "${YELLOW}⚠️${NC} $1"
    ((WARNINGS++))
}

check_info() {
    echo -e "${BLUE}ℹ️${NC} $1"
}

echo "================================"
echo "Qallow Health Check"
echo "================================"
echo ""

# 1. System Dependencies
echo "[1/6] Checking system dependencies..."
echo "--------------------------------"

if command -v cmake &> /dev/null; then
    CMAKE_VERSION=$(cmake --version | head -n1)
    check_pass "CMake: $CMAKE_VERSION"
else
    check_fail "CMake not found"
fi

if command -v gcc &> /dev/null; then
    check_pass "GCC installed"
else
    check_fail "GCC not found"
fi

if command -v g++ &> /dev/null; then
    check_pass "G++ installed"
else
    check_fail "G++ not found"
fi

if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    check_pass "Python3: $PYTHON_VERSION"
else
    check_fail "Python3 not found"
fi

if command -v cargo &> /dev/null; then
    CARGO_VERSION=$(cargo --version)
    check_pass "Cargo: $CARGO_VERSION"
else
    check_warn "Cargo not found (Rust app won't build)"
fi

if command -v nvcc &> /dev/null; then
    NVCC_VERSION=$(nvcc --version | tail -n1)
    check_pass "NVCC: $NVCC_VERSION"
else
    check_warn "NVCC not found (CUDA disabled)"
fi

echo ""

# 2. Font Paths
echo "[2/6] Checking font paths..."
echo "--------------------------------"

FONT_PATH="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
if [ -f "$FONT_PATH" ]; then
    check_pass "DejaVuSans font found at $FONT_PATH"
else
    check_fail "DejaVuSans font not found at $FONT_PATH"
fi

echo ""

# 3. Python Dependencies
echo "[3/6] Checking Python dependencies..."
echo "--------------------------------"

python3 -c "import flask" 2>/dev/null && check_pass "Flask installed" || check_warn "Flask not installed"
python3 -c "import flask_cors" 2>/dev/null && check_pass "Flask-CORS installed" || check_warn "Flask-CORS not installed"
python3 -c "import cirq" 2>/dev/null && check_pass "Cirq installed" || check_warn "Cirq not installed"

echo ""

# 4. Python Dashboard
echo "[4/6] Checking Python dashboard..."
echo "--------------------------------"

if python3 -m py_compile "$PROJECT_ROOT/ui/dashboard.py" 2>/dev/null; then
    check_pass "dashboard.py syntax valid"
else
    check_fail "dashboard.py has syntax errors"
fi

echo ""

# 5. Build System
echo "[5/6] Checking build system..."
echo "--------------------------------"

if [ -f "$PROJECT_ROOT/CMakeLists.txt" ]; then
    check_pass "CMakeLists.txt found"
else
    check_fail "CMakeLists.txt not found"
fi

if [ -d "$PROJECT_ROOT/build" ]; then
    check_info "Build directory exists"
    if [ -f "$PROJECT_ROOT/build/CMakeCache.txt" ]; then
        check_pass "CMake cache found (build configured)"
    else
        check_warn "CMake cache not found (run: cmake -B build)"
    fi
else
    check_warn "Build directory not found (run: cmake -B build)"
fi

echo ""

# 6. Cirq Phase 11 Integration
echo "[6/6] Checking Cirq Phase 11 integration..."
echo "--------------------------------"

if [ -f "$PROJECT_ROOT/python/quantum/cirq_phase11.py" ]; then
    check_pass "cirq_phase11.py found"
    
    if python3 "$PROJECT_ROOT/python/quantum/cirq_phase11.py" --ticks=2 --simulator=ideal >/dev/null 2>&1; then
        check_pass "Cirq Phase 11 executes successfully"
    else
        check_warn "Cirq Phase 11 execution failed (may need dependencies)"
    fi
else
    check_fail "cirq_phase11.py not found"
fi

echo ""

# Summary
echo "================================"
echo "Health Check Summary"
echo "================================"
echo -e "${GREEN}Passed:${NC}   $PASSED"
echo -e "${RED}Failed:${NC}   $FAILED"
echo -e "${YELLOW}Warnings:${NC} $WARNINGS"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✅ All critical checks passed!${NC}"
    if [ $WARNINGS -gt 0 ]; then
        echo -e "${YELLOW}⚠️  $WARNINGS warnings - see above for details${NC}"
    fi
    exit 0
else
    echo -e "${RED}❌ $FAILED critical checks failed!${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Review failures above"
    echo "  2. Install missing dependencies"
    echo "  3. Run this script again"
    exit 1
fi

