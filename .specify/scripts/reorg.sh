#!/bin/bash

################################################################################
# Qallow Codebase Reorganization Script
# Purpose: Move loose root files into organized directories per Constitution § IV
# Created: 2025-11-06
# Log: reorg.log
################################################################################

set -u  # Exit on undefined variable
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Navigate up two levels from scripts/ to project root
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
LOG_FILE="${PROJECT_ROOT}/reorg.log"
ERROR_LOG="${PROJECT_ROOT}/reorg_errors.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Counters
FILES_MOVED=0
DIRS_CREATED=0
ERRORS=0

################################################################################
# Logging Functions
################################################################################

log_info() {
    local msg="$1"
    echo -e "${BLUE}[INFO]${NC} $msg" | tee -a "$LOG_FILE"
}

log_success() {
    local msg="$1"
    echo -e "${GREEN}[✓]${NC} $msg" | tee -a "$LOG_FILE"
}

log_warning() {
    local msg="$1"
    echo -e "${YELLOW}[⚠]${NC} $msg" | tee -a "$LOG_FILE"
}

log_error() {
    local msg="$1"
    echo -e "${RED}[✗]${NC} $msg" | tee -a "$LOG_FILE" "$ERROR_LOG"
    ((ERRORS++))
}

log_action() {
    local action="$1"
    local source="$2"
    local dest="$3"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $action: $source → $dest" >> "$LOG_FILE"
}

################################################################################
# Utility Functions
################################################################################

create_dir_if_missing() {
    local dir="$1"
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        log_success "Created directory: $dir"
        ((DIRS_CREATED++))
        log_action "CREATE_DIR" "" "$dir"
    else
        log_info "Directory already exists: $dir"
    fi
}

move_file() {
    local src="$1"
    local dest_dir="$2"
    
    if [ ! -f "$src" ]; then
        log_warning "Source file not found: $src"
        return 1
    fi
    
    local filename=$(basename "$src")
    local dest="${dest_dir}/${filename}"
    
    if mv "$src" "$dest" 2>/dev/null; then
        ((FILES_MOVED++))
        log_success "Moved: $filename → $dest_dir/"
        log_action "MOVE" "$src" "$dest"
    else
        log_error "Failed to move: $src → $dest"
        return 1
    fi
}

################################################################################
# Main Reorganization Logic
################################################################################

main() {
    echo ""
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║   Qallow Codebase Reorganization Script (Constitution § IV)    ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo ""
    
    # Initialize logs
    > "$LOG_FILE"
    > "$ERROR_LOG"
    
    log_info "Starting reorganization at: $PROJECT_ROOT"
    log_info "Timestamp: $(date)"
    log_info "=========================================="
    
    # Save current directory
    cd "$PROJECT_ROOT"
    
    ############################################################################
    # PHASE 1: CREATE TARGET DIRECTORIES
    ############################################################################
    
    echo -e "\n${BLUE}═══ PHASE 1: Creating Target Directories ═══${NC}\n"
    
    create_dir_if_missing "docs"
    create_dir_if_missing "scripts"
    create_dir_if_missing "config"
    create_dir_if_missing "config/docker"
    create_dir_if_missing "data/logs"
    create_dir_if_missing "data/metrics"
    
    ############################################################################
    # PHASE 2: MOVE DOCUMENTATION FILES (.md, .txt)
    ############################################################################
    
    echo -e "\n${BLUE}═══ PHASE 2: Moving Documentation Files ═══${NC}\n"
    
    log_info "Moving Markdown documentation files..."
    for file in *.md; do
        # Skip README.md (stays in root per convention)
        if [ "$file" = "README.md" ]; then
            log_info "Skipping (root convention): $file"
            continue
        fi
        if [ -f "$file" ]; then
            move_file "$file" "docs"
        fi
    done
    
    log_info "Moving text documentation files..."
    for file in *.txt; do
        # Skip if file doesn't exist (glob expanded to literal string)
        if [ ! -f "$file" ]; then
            continue
        fi
        # Only move known documentation files
        case "$file" in
            EXECUTION_SUMMARY.txt|RUN_QUICK_SUMMARY.txt|FINAL_SERVER_SUMMARY.txt|\
            IMPLEMENTATION_COMPLETE.txt|VERIFICATION_COMPLETE.txt|\
            SPECKIT_INSTALLATION_COMPLETE.txt|TEST_DRIVEN_FIX_IMPLEMENTATION_COMPLETE.txt|\
            SEQUENTIAL_THINKING_CHANGES.txt)
                move_file "$file" "docs"
                ;;
            *)
                log_info "Skipping non-doc text file: $file"
                ;;
        esac
    done
    
    ############################################################################
    # PHASE 3: MOVE PYTHON SCRIPTS
    ############################################################################
    
    echo -e "\n${BLUE}═══ PHASE 3: Moving Python Scripts ═══${NC}\n"
    
    log_info "Moving Python utility scripts..."
    for file in *.py; do
        if [ ! -f "$file" ]; then
            continue
        fi
        move_file "$file" "scripts"
    done
    
    # Move shell scripts (development/build utilities)
    if [ -f "run_full_build.sh" ]; then
        move_file "run_full_build.sh" "scripts"
    fi
    
    ############################################################################
    # PHASE 4: MOVE CONFIGURATION FILES
    ############################################################################
    
    echo -e "\n${BLUE}═══ PHASE 4: Moving Configuration Files ═══${NC}\n"
    
    log_info "Moving requirements files..."
    for req_file in requirements*.txt; do
        if [ -f "$req_file" ]; then
            move_file "$req_file" "config"
        fi
    done
    
    log_info "Moving Docker configuration..."
    if [ -f "docker-compose.yaml" ]; then
        move_file "docker-compose.yaml" "config/docker"
    fi
    
    ############################################################################
    # PHASE 5: MOVE DATA & LOG FILES
    ############################################################################
    
    echo -e "\n${BLUE}═══ PHASE 5: Moving Data & Log Files ═══${NC}\n"
    
    log_info "Moving CSV data files..."
    for csv_file in *.csv; do
        if [ ! -f "$csv_file" ]; then
            continue
        fi
        move_file "$csv_file" "data/logs"
    done
    
    log_info "Moving log files..."
    for log_file in *.log; do
        if [ ! -f "$log_file" ]; then
            continue
        fi
        move_file "$log_file" "data/logs"
    done
    
    log_info "Moving JSON metric/config files..."
    if [ -f "agent_changes.json" ]; then
        move_file "agent_changes.json" "data/metrics"
    fi
    
    ############################################################################
    # PHASE 6: VERIFY BUILD SYSTEM FILES REMAIN
    ############################################################################
    
    echo -e "\n${BLUE}═══ PHASE 6: Verifying Build System Files ═══${NC}\n"
    
    for build_file in CMakeLists.txt Cargo.toml bootstrap.sh; do
        if [ -f "$build_file" ]; then
            log_success "Build system file preserved: $build_file"
        else
            log_warning "Expected build file not found: $build_file"
        fi
    done
    
    ############################################################################
    # PHASE 7: FINAL VERIFICATION
    ############################################################################
    
    echo -e "\n${BLUE}═══ PHASE 7: Final Verification ═══${NC}\n"
    
    log_info "Scanning for remaining loose files..."
    
    # Check for any loose files that shouldn't be in root
    REMAINING=$(find . -maxdepth 1 -type f \
        \( -name "*.md" -o -name "*.py" -o -name "*.sh" -o -name "*.csv" -o -name "*.log" -o -name "*.json" -o -name "requirements*.txt" \) \
        ! -name "README.md" \
        ! -name "CMakeLists.txt" \
        ! -name "Cargo.toml" \
        ! -name "bootstrap.sh" \
        ! -name ".gitignore" \
        ! -name ".gitattributes" \
        2>/dev/null | wc -l)
    
    if [ "$REMAINING" -eq 0 ]; then
        log_success "✓ No loose files remaining in root (excluding build system + README)"
    else
        log_warning "Found $REMAINING loose files that may need attention:"
        find . -maxdepth 1 -type f \
            \( -name "*.md" -o -name "*.py" -o -name "*.sh" -o -name "*.csv" -o -name "*.log" -o -name "*.json" -o -name "requirements*.txt" \) \
            ! -name "README.md" \
            ! -name "CMakeLists.txt" \
            ! -name "Cargo.toml" \
            ! -name "bootstrap.sh" \
            ! -name ".gitignore" \
            ! -name ".gitattributes" \
            2>/dev/null | while read file; do
            log_warning "  - $file"
        done
    fi
    
    ############################################################################
    # SUMMARY REPORT
    ############################################################################
    
    echo ""
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║                    Reorganization Summary                      ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo ""
    echo -e "${GREEN}✓ Directories Created:${NC} $DIRS_CREATED"
    echo -e "${GREEN}✓ Files Moved:${NC} $FILES_MOVED"
    
    if [ $ERRORS -eq 0 ]; then
        echo -e "${GREEN}✓ Errors:${NC} $ERRORS"
        echo -e "\n${GREEN}✓ Reorganization completed successfully!${NC}"
    else
        echo -e "${RED}✗ Errors:${NC} $ERRORS"
        echo -e "\n${YELLOW}⚠ Some issues occurred. Check $ERROR_LOG for details.${NC}"
    fi
    
    echo ""
    echo "📊 Directory Structure After Reorganization:"
    echo "  docs/                 - Documentation, status reports, guides"
    echo "  scripts/              - Python utilities and shell scripts"
    echo "  config/               - Configuration files (requirements, docker)"
    echo "  data/logs/            - Log files and metric CSVs"
    echo "  data/metrics/         - JSON metrics and data"
    echo ""
    echo "📝 Full log saved to: $LOG_FILE"
    if [ $ERRORS -gt 0 ]; then
        echo "⚠️  Error log saved to: $ERROR_LOG"
    fi
    echo ""
    
    # Log final statistics
    log_info "=========================================="
    log_info "SUMMARY: Created $DIRS_CREATED dirs, moved $FILES_MOVED files, $ERRORS errors"
    log_info "Reorganization completed at: $(date)"
    
    return $ERRORS
}

################################################################################
# Execute Main
################################################################################

main
EXIT_CODE=$?

exit $EXIT_CODE
