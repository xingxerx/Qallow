#!/bin/bash

################################################################################
# Qallow Reorganization Validation Script
# Purpose: Validate that codebase reorganization is complete and correct
# Author: GitHub Copilot
# Date: 2025-11-06
#
# Usage: bash scripts/validate.sh [--verbose]
#
# This script checks:
#   1. No loose files remain in root (except allowed exceptions)
#   2. All expected directories exist
#   3. Files have correct permissions
#   4. Git history is clean
#   5. No broken symlinks
################################################################################

set -u
set -o pipefail

# Configuration
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
readonly REPORT_FILE="${REPO_ROOT}/reorganization_report.txt"

VERBOSE=false

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

################################################################################
# Utility Functions
################################################################################

pass() {
    echo -e "${GREEN}✓${NC} $*"
}

fail() {
    echo -e "${RED}✗ FAIL${NC}: $*"
}

warn() {
    echo -e "${YELLOW}⚠${NC} $*"
}

info() {
    echo -e "${BLUE}ℹ${NC} $*"
}

print_header() {
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
}

print_section() {
    echo ""
    echo -e "${BLUE}▸ $1${NC}"
}

################################################################################
# Validation Checks
################################################################################

check_loose_files() {
    print_section "Checking for loose files in root"
    
    local allowed_in_root=(
        "CMakeLists.txt"
        "Cargo.toml"
        "Cargo.lock"
        "Makefile"
        "Dockerfile"
        "setup.bat"
        "bootstrap.sh"
        "LICENSE"
        "README.md"
        "Qallow.code-workspace"
        "reorg.log"
        ".gitignore"
        ".github"
        ".specify"
        "build"
        "venv"
        "target"
        "node_modules"
        "scripts"
        "docs"
        "config"
        "deploy"
        "public"
        "backend"
        "python"
        "src"
        "misc"
        "interface"
        "phases"
        "shared"
        "runtime"
        "alg"
        "ops"
        "telemetry"
        "io"
        "quantum_ml"
        "Testing"
        "c_ext"
        "server"
        "native_app"
        "proto"
        "include"
        ".git"
        ".vscode"
        ".github"
    )
    
    local loose_files=()
    
    for item in $(find "${REPO_ROOT}" -maxdepth 1 -not -name '.*' ! -path './.git/*'); do
        local name=$(basename "$item")
        local is_allowed=false
        
        for allowed in "${allowed_in_root[@]}"; do
            if [[ "$name" == "$allowed" ]]; then
                is_allowed=true
                break
            fi
        done
        
        if [[ "$is_allowed" == "false" ]] && [[ -f "$item" ]]; then
            loose_files+=("$name")
        fi
    done
    
    if [[ ${#loose_files[@]} -eq 0 ]]; then
        pass "No loose files found in root"
        return 0
    else
        fail "Found ${#loose_files[@]} loose file(s) in root:"
        for file in "${loose_files[@]}"; do
            echo "    - $file"
        done
        return 1
    fi
}

check_directories_exist() {
    print_section "Checking for required directories"
    
    local required_dirs=(
        "docs"
        "scripts"
        "config"
        "deploy"
        "public/assets"
        "backend/cpu/misc"
        "src"
    )
    
    local missing=0
    
    for dir in "${required_dirs[@]}"; do
        if [[ -d "${REPO_ROOT}/${dir}" ]]; then
            local file_count=$(find "${REPO_ROOT}/${dir}" -maxdepth 1 -type f 2>/dev/null | wc -l)
            info "Directory exists: ${dir}/ (${file_count} files)"
        else
            warn "Directory missing: ${dir}/"
            ((missing++))
        fi
    done
    
    if [[ $missing -eq 0 ]]; then
        pass "All required directories present"
        return 0
    else
        fail "${missing} required directory(ies) missing"
        return 1
    fi
}

check_file_organization() {
    print_section "Checking file organization by category"
    
    local issues=0
    
    # Check docs/
    if [[ -d "${REPO_ROOT}/docs" ]]; then
        local wrong_files=$(find "${REPO_ROOT}/docs" -maxdepth 1 -type f ! -name '*.md' ! -name '*.txt' 2>/dev/null)
        if [[ -n "$wrong_files" ]]; then
            fail "Found non-documentation files in docs/:"
            echo "$wrong_files" | sed 's|^|    - |'
            ((issues++))
        else
            pass "All files in docs/ are documentation (.md, .txt)"
        fi
    fi
    
    # Check scripts/
    if [[ -d "${REPO_ROOT}/scripts" ]]; then
        local py_count=$(find "${REPO_ROOT}/scripts" -maxdepth 1 -name '*.py' -type f 2>/dev/null | wc -l)
        local sh_count=$(find "${REPO_ROOT}/scripts" -maxdepth 1 -name '*.sh' -type f 2>/dev/null | wc -l)
        info "scripts/ contains ${py_count} Python and ${sh_count} shell scripts"
    fi
    
    # Check config/
    if [[ -d "${REPO_ROOT}/config" ]]; then
        local config_files=$(find "${REPO_ROOT}/config" -maxdepth 1 -type f 2>/dev/null | wc -l)
        info "config/ contains ${config_files} configuration file(s)"
    fi
    
    # Check deploy/
    if [[ -d "${REPO_ROOT}/deploy" ]]; then
        local deploy_files=$(find "${REPO_ROOT}/deploy" -maxdepth 1 -type f 2>/dev/null | wc -l)
        info "deploy/ contains ${deploy_files} deployment file(s)"
    fi
    
    return $issues
}

check_file_permissions() {
    print_section "Checking file permissions"
    
    local issues=0
    
    # Check that scripts are executable
    if [[ -d "${REPO_ROOT}/scripts" ]]; then
        while IFS= read -r script; do
            if [[ -f "$script" ]]; then
                if [[ -x "$script" ]]; then
                    if [[ "$VERBOSE" == "true" ]]; then
                        pass "Executable: $(basename $script)"
                    fi
                else
                    warn "Not executable: $(basename $script) (consider running 'chmod +x')"
                    ((issues++))
                fi
            fi
        done < <(find "${REPO_ROOT}/scripts" -maxdepth 1 -name '*.sh' -type f)
    fi
    
    pass "File permissions checked"
    return $issues
}

check_git_status() {
    print_section "Checking git status"
    
    cd "${REPO_ROOT}" || return 1
    
    if ! git rev-parse --git-dir > /dev/null 2>&1; then
        warn "Not a git repository"
        return 0
    fi
    
    # Check if working tree is clean
    if git diff-index --quiet HEAD -- 2>/dev/null; then
        pass "Git working tree is clean"
    else
        warn "Git working tree has uncommitted changes"
        git status --short | head -5
        if [[ $(git status --short | wc -l) -gt 5 ]]; then
            echo "  (and more...)"
        fi
    fi
    
    # Show recent commits related to reorganization
    local reorg_commits=$(git log --oneline --grep="reorganize\|chore" -n 5 2>/dev/null | wc -l)
    if [[ $reorg_commits -gt 0 ]]; then
        pass "Found ${reorg_commits} reorganization commit(s)"
        git log --oneline --grep="reorganize\|chore" -n 3 2>/dev/null | sed 's/^/    /'
    fi
    
    return 0
}

check_broken_symlinks() {
    print_section "Checking for broken symlinks"
    
    local broken=()
    
    while IFS= read -r link; do
        if [[ -L "$link" ]] && [[ ! -e "$link" ]]; then
            broken+=("$link")
        fi
    done < <(find "${REPO_ROOT}" -maxdepth 3 -type l 2>/dev/null)
    
    if [[ ${#broken[@]} -eq 0 ]]; then
        pass "No broken symlinks found"
        return 0
    else
        fail "Found ${#broken[@]} broken symlink(s):"
        for link in "${broken[@]}"; do
            echo "    - $link"
        done
        return 1
    fi
}

check_reorg_log() {
    print_section "Checking reorganization log"
    
    if [[ -f "${REPO_ROOT}/reorg.log" ]]; then
        pass "Reorganization log exists: reorg.log"
        local op_count=$(grep -c "\[SUCCESS\]" "${REPO_ROOT}/reorg.log" 2>/dev/null || echo 0)
        info "Log contains ${op_count} successful operation(s)"
        
        if [[ "$VERBOSE" == "true" ]]; then
            tail -5 "${REPO_ROOT}/reorg.log" | sed 's/^/    /'
        fi
    else
        warn "Reorganization log not found"
    fi
    
    return 0
}

################################################################################
# Report Generation
################################################################################

generate_report() {
    print_header "REORGANIZATION VALIDATION REPORT"
    
    # Initialize report
    {
        echo "# Qallow Codebase Reorganization Validation Report"
        echo "Generated: $(date)"
        echo "Repository: ${REPO_ROOT}"
        echo ""
        echo "## Executive Summary"
        echo ""
        echo "This report validates the reorganization of loose files into dedicated"
        echo "directories per Constitution § IV."
        echo ""
        echo "## Findings"
        echo ""
        echo "### Root Directory Contents"
        echo ""
        ls -1 "${REPO_ROOT}" | grep -v '^\.' | sed 's/^/- /'
        echo ""
        echo "### Organized Directories"
        echo ""
        for dir in docs scripts config deploy public backend src misc; do
            if [[ -d "${REPO_ROOT}/${dir}" ]]; then
                local count=$(find "${REPO_ROOT}/${dir}" -type f 2>/dev/null | wc -l)
                echo "- **${dir}/** (${count} files)"
            fi
        done
        echo ""
        echo "### Git Commits"
        echo ""
        git log --oneline --grep="reorganize\|chore" -n 10 2>/dev/null | sed 's/^/- /'
        echo ""
        echo "---"
        echo "Report generated by: validate.sh"
    } > "${REPORT_FILE}"
    
    echo "Report saved to: ${REPORT_FILE}"
}

################################################################################
# Main Execution
################################################################################

main() {
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --verbose)
                VERBOSE=true
                shift
                ;;
            *)
                echo "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    print_header "REORGANIZATION VALIDATION"
    
    local failures=0
    
    # Run all checks
    check_loose_files || ((failures++))
    check_directories_exist || ((failures++))
    check_file_organization || ((failures++))
    check_file_permissions || ((failures++))
    check_git_status || ((failures++))
    check_broken_symlinks || ((failures++))
    check_reorg_log || ((failures++))
    
    echo ""
    if [[ $failures -eq 0 ]]; then
        pass "All validation checks passed"
        print_header "REORGANIZATION SUCCESSFUL ✓"
        generate_report
        exit 0
    else
        fail "Validation found ${failures} issue(s)"
        print_header "REORGANIZATION NEEDS ATTENTION ⚠"
        generate_report
        exit 1
    fi
}

# Run main function
main "$@"
