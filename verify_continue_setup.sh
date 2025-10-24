#!/bin/bash

# ============================================================================
# Continue.dev Setup Verification Script
# ============================================================================
# Verifies that Continue.dev is properly configured for Qallow
# ============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Counters
CHECKS_PASSED=0
CHECKS_FAILED=0
CHECKS_WARNING=0

# ============================================================================
# Helper Functions
# ============================================================================

print_header() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}\n"
}

check_pass() {
    echo -e "${GREEN}✓ $1${NC}"
    ((CHECKS_PASSED++))
}

check_fail() {
    echo -e "${RED}✗ $1${NC}"
    ((CHECKS_FAILED++))
}

check_warn() {
    echo -e "${YELLOW}⚠ $1${NC}"
    ((CHECKS_WARNING++))
}

# ============================================================================
# Verification Checks
# ============================================================================

verify_files() {
    print_header "Checking Configuration Files"
    
    # Check .env exists
    if [ -f .env ]; then
        check_pass ".env file exists"
    else
        check_fail ".env file not found (run: cp .env.example .env)"
    fi
    
    # Check .continue/config.json exists
    if [ -f .continue/config.json ]; then
        check_pass ".continue/config.json exists"
    else
        check_fail ".continue/config.json not found"
    fi
    
    # Check MCP server config exists
    if [ -f .continue/mcpServers/qallow-memory.yaml ]; then
        check_pass ".continue/mcpServers/qallow-memory.yaml exists"
    else
        check_warn ".continue/mcpServers/qallow-memory.yaml not found (optional)"
    fi
    
    # Check setup script exists
    if [ -f setup_continue.sh ]; then
        check_pass "setup_continue.sh exists"
    else
        check_fail "setup_continue.sh not found"
    fi
}

verify_env_variables() {
    print_header "Checking Environment Variables"
    
    # Check if any API key is set
    if [ -n "$GEMINI_API_KEY" ]; then
        check_pass "GEMINI_API_KEY is set"
    elif [ -n "$ANTHROPIC_API_KEY" ]; then
        check_pass "ANTHROPIC_API_KEY is set"
    elif [ -n "$OPENAI_API_KEY" ]; then
        check_pass "OPENAI_API_KEY is set"
    elif [ -n "$OLLAMA_API_BASE" ]; then
        check_pass "OLLAMA_API_BASE is set"
    else
        check_warn "No API keys found in environment (run: source .env)"
    fi
    
    # Check Qallow variables
    if [ -n "$QALLOW_ENABLE_CUDA" ]; then
        check_pass "QALLOW_ENABLE_CUDA is set"
    else
        check_warn "QALLOW_ENABLE_CUDA not set"
    fi
}

verify_config_json() {
    print_header "Checking Continue Configuration"
    
    if [ ! -f .continue/config.json ]; then
        check_fail "config.json not found"
        return
    fi
    
    # Check if config is valid JSON
    if python3 -m json.tool .continue/config.json > /dev/null 2>&1; then
        check_pass "config.json is valid JSON"
    else
        check_fail "config.json is not valid JSON"
        return
    fi
    
    # Check for models
    if grep -q '"models"' .continue/config.json; then
        check_pass "Models are configured"
    else
        check_fail "No models found in config"
    fi
    
    # Check for Ollama
    if grep -q '"ollama"' .continue/config.json; then
        check_pass "Ollama model is configured"
    else
        check_warn "Ollama model not configured"
    fi
    
    # Check for Gemini
    if grep -q '"gemini"' .continue/config.json; then
        check_pass "Gemini model is configured"
    else
        check_warn "Gemini model not configured"
    fi
}

verify_dependencies() {
    print_header "Checking Dependencies"
    
    # Check Python
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
        check_pass "Python 3 is installed ($PYTHON_VERSION)"
    else
        check_fail "Python 3 is not installed"
    fi
    
    # Check Ollama (optional)
    if command -v ollama &> /dev/null; then
        check_pass "Ollama is installed"
        
        # Check if Ollama is running
        if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
            check_pass "Ollama is running"
        else
            check_warn "Ollama is installed but not running (run: ollama serve)"
        fi
    else
        check_warn "Ollama is not installed (optional, for local models)"
    fi
    
    # Check VS Code
    if command -v code &> /dev/null; then
        check_pass "VS Code is installed"
    else
        check_warn "VS Code is not installed"
    fi
}

verify_security() {
    print_header "Checking Security"
    
    # Check .env is in .gitignore
    if grep -q "^\.env$" .gitignore; then
        check_pass ".env is in .gitignore"
    else
        check_fail ".env is not in .gitignore"
    fi
    
    # Check .continue/config.json is in .gitignore
    if grep -q "\.continue/config\.json" .gitignore; then
        check_pass ".continue/config.json is in .gitignore"
    else
        check_fail ".continue/config.json is not in .gitignore"
    fi
    
    # Check .env doesn't have actual keys (basic check)
    if grep -E "GEMINI_API_KEY=[a-zA-Z0-9]" .env > /dev/null 2>&1; then
        check_warn ".env appears to have actual API keys (should be empty in repo)"
    else
        check_pass ".env doesn't have exposed API keys"
    fi
}

print_summary() {
    print_header "Verification Summary"
    
    echo -e "Passed:  ${GREEN}$CHECKS_PASSED${NC}"
    echo -e "Failed:  ${RED}$CHECKS_FAILED${NC}"
    echo -e "Warnings: ${YELLOW}$CHECKS_WARNING${NC}"
    echo ""
    
    if [ $CHECKS_FAILED -eq 0 ]; then
        echo -e "${GREEN}✓ All critical checks passed!${NC}"
        echo ""
        echo "Next steps:"
        echo "  1. Load environment: source .env"
        echo "  2. Open VS Code"
        echo "  3. Press Ctrl+L to start using Continue"
        return 0
    else
        echo -e "${RED}✗ Some checks failed. Please fix the issues above.${NC}"
        return 1
    fi
}

# ============================================================================
# Main
# ============================================================================

main() {
    print_header "Continue.dev Setup Verification"
    
    verify_files
    verify_env_variables
    verify_config_json
    verify_dependencies
    verify_security
    print_summary
}

if [ "${BASH_SOURCE[0]}" == "${0}" ]; then
    main "$@"
fi

