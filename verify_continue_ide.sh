#!/bin/bash

# Continue IDE Verification Script
# This script verifies that Continue IDE is properly configured

echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo "                    Continue IDE Configuration Verification"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

PASSED=0
FAILED=0

# Function to check if file exists
check_file() {
    local file=$1
    local description=$2
    
    if [ -f "$file" ]; then
        echo -e "${GREEN}✓${NC} $description"
        ((PASSED++))
        return 0
    else
        echo -e "${RED}✗${NC} $description"
        ((FAILED++))
        return 1
    fi
}

# Function to check if directory exists
check_dir() {
    local dir=$1
    local description=$2
    
    if [ -d "$dir" ]; then
        echo -e "${GREEN}✓${NC} $description"
        ((PASSED++))
        return 0
    else
        echo -e "${RED}✗${NC} $description"
        ((FAILED++))
        return 1
    fi
}

# Function to check if environment variable is set
check_env() {
    local var=$1
    local description=$2
    
    if [ -n "${!var}" ]; then
        echo -e "${GREEN}✓${NC} $description (${!var:0:10}...)"
        ((PASSED++))
        return 0
    else
        echo -e "${YELLOW}⚠${NC} $description (not set - optional)"
        return 0
    fi
}

# Function to check if command exists
check_command() {
    local cmd=$1
    local description=$2
    
    if command -v "$cmd" &> /dev/null; then
        echo -e "${GREEN}✓${NC} $description"
        ((PASSED++))
        return 0
    else
        echo -e "${YELLOW}⚠${NC} $description (not installed)"
        return 0
    fi
}

echo "1. Checking Configuration Files"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
check_file ~/.continue/config.json "Continue IDE config exists (~/.continue/config.json)"
check_file /root/Qallow/.env "Environment file exists (/root/Qallow/.env)"
check_file /root/Qallow/.env.example "Environment template exists (/root/Qallow/.env.example)"

echo ""
echo "2. Checking Environment Variables"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
check_env GEMINI_API_KEY "Gemini API key loaded"
check_env ANTHROPIC_API_KEY "Anthropic API key loaded"
check_env OPENAI_API_KEY "OpenAI API key loaded"

echo ""
echo "3. Checking Shell Configuration"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if grep -q "source.*\.env" ~/.bashrc 2>/dev/null; then
    echo -e "${GREEN}✓${NC} .env loading configured in ~/.bashrc"
    ((PASSED++))
else
    echo -e "${YELLOW}⚠${NC} .env loading not configured in ~/.bashrc"
fi

if [ -f ~/.zshrc ]; then
    if grep -q "source.*\.env" ~/.zshrc 2>/dev/null; then
        echo -e "${GREEN}✓${NC} .env loading configured in ~/.zshrc"
        ((PASSED++))
    else
        echo -e "${YELLOW}⚠${NC} .env loading not configured in ~/.zshrc"
    fi
fi

echo ""
echo "4. Checking Optional Tools"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
check_command ollama "Ollama installed (optional but recommended)"
check_command git "Git installed"

echo ""
echo "5. Checking Git Security"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if git check-ignore /root/Qallow/.env &>/dev/null; then
    echo -e "${GREEN}✓${NC} .env is protected by .gitignore"
    ((PASSED++))
else
    echo -e "${RED}✗${NC} .env is NOT protected by .gitignore"
    ((FAILED++))
fi

echo ""
echo "6. Checking Continue IDE Config Content"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if grep -q "ollama" ~/.continue/config.json 2>/dev/null; then
    echo -e "${GREEN}✓${NC} Ollama configured as default model"
    ((PASSED++))
else
    echo -e "${RED}✗${NC} Ollama not found in config"
    ((FAILED++))
fi

if grep -q "localhost:11434" ~/.continue/config.json 2>/dev/null; then
    echo -e "${GREEN}✓${NC} Ollama API base configured correctly"
    ((PASSED++))
else
    echo -e "${RED}✗${NC} Ollama API base not configured"
    ((FAILED++))
fi

echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo "                              Summary"
echo "════════════════════════════════════════════════════════════════════════════════"
echo -e "Passed: ${GREEN}$PASSED${NC}"
echo -e "Failed: ${RED}$FAILED${NC}"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ All checks passed! Continue IDE is properly configured.${NC}"
    echo ""
    echo "Next steps:"
    echo "1. Restart Continue IDE (close and reopen)"
    echo "2. If using Ollama, make sure it's running: ollama serve"
    echo "3. If using cloud models, verify API keys are set: echo \$GEMINI_API_KEY"
    exit 0
else
    echo -e "${RED}✗ Some checks failed. Please review the issues above.${NC}"
    echo ""
    echo "Troubleshooting:"
    echo "1. Reload shell: source ~/.bashrc"
    echo "2. Check .env file: cat /root/Qallow/.env"
    echo "3. Check config: cat ~/.continue/config.json"
    echo "4. Read guide: cat /root/Qallow/FIX_CONTINUE_IDE.md"
    exit 1
fi

