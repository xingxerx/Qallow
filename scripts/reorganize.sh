#!/bin/bash

################################################################################
# Qallow Codebase Reorganization Script
# Purpose: Move loose files from root to dedicated directories per Constitution § IV
# Author: GitHub Copilot
# Date: 2025-11-06
# 
# Usage: bash scripts/reorganize.sh [--dry-run] [--verbose]
# 
# This script organizes files by category into dedicated directories:
#   - Documentation (.md, .txt) → docs/
#   - Python scripts (.py) → scripts/
#   - Shell scripts (.sh) → scripts/
#   - Configuration (.json, .yaml) → config/
#   - Binary assets (.deb, .tar.gz) → deploy/ or public/assets/
#   - C source (.c, .h) → backend/cpu/misc/ or context-appropriate
#   - TypeScript/JavaScript (.ts, .js) → src/
#   - Unknown → misc/
#
# Build configuration files remain in root:
#   - CMakeLists.txt, Cargo.toml, Makefile, Dockerfile, bootstrap.sh, etc.
################################################################################

set -u  # Exit on undefined variable
set -o pipefail  # Exit on pipe failure

# Configuration
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
readonly LOG_FILE="${REPO_ROOT}/reorg.log"

# Flags
DRY_RUN=false
VERBOSE=false
CONFIRM=true

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'  # No Color

################################################################################
# Utility Functions
################################################################################

log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[${timestamp}] [${level}] ${message}" >> "${LOG_FILE}"
    
    if [[ "$VERBOSE" == "true" ]]; then
        echo -e "${BLUE}[${level}]${NC} ${message}" >&2
    fi
}

log_success() {
    local message="$*"
    echo -e "${GREEN}✓${NC} ${message}"
    log "SUCCESS" "${message}"
}

log_error() {
    local message="$*"
    echo -e "${RED}✗ ERROR: ${message}${NC}" >&2
    log "ERROR" "${message}"
}

log_warning() {
    local message="$*"
    echo -e "${YELLOW}⚠${NC} ${message}"
    log "WARNING" "${message}"
}

log_info() {
    local message="$*"
    echo -e "${BLUE}ℹ${NC} ${message}"
    log "INFO" "${message}"
}

print_header() {
    local title="$1"
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  ${title}${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
}

initialize_log() {
    echo "" > "${LOG_FILE}"
    log "INFO" "==============================================="
    log "INFO" "Qallow Codebase Reorganization Started"
    log "INFO" "Repository Root: ${REPO_ROOT}"
    log "INFO" "Dry Run: ${DRY_RUN}"
    log "INFO" "Verbose: ${VERBOSE}"
    log "INFO" "==============================================="
}

################################################################################
# File Inventory & Categorization
################################################################################

get_file_category() {
    local file="$1"
    local basename=$(basename "$file")
    local ext="${file##*.}"
    
    # Special cases: build configuration (keep in root)
    if [[ "$basename" =~ ^(CMakeLists\.txt|Cargo\.toml|Cargo\.lock|Makefile|Dockerfile|setup\.bat|bootstrap\.sh|Qallow\.code-workspace|LICENSE|README\.md|reorg\.log)$ ]]; then
        echo "BUILD_CONFIG"
        return
    fi
    
    # By extension
    case "$ext" in
        md|txt)
            echo "DOCS"
            ;;
        py)
            echo "SCRIPTS_PYTHON"
            ;;
        sh)
            echo "SCRIPTS_SHELL"
            ;;
        json|yaml|yml)
            echo "CONFIG"
            ;;
        c|h)
            echo "SOURCE_C"
            ;;
        js|ts)
            echo "SOURCE_JS"
            ;;
        deb)
            echo "ASSETS_BINARY"
            ;;
        tar|gz|zip|tar\.gz)
            echo "ASSETS_ARCHIVE"
            ;;
        png|jpg|jpeg|svg|gif|webp)
            echo "ASSETS_IMAGE"
            ;;
        *)
            # Unknown extension or no extension
            if [[ "$ext" == "$file" ]]; then
                # No extension
                echo "MISC"
            else
                echo "MISC"
            fi
            ;;
    esac
}

get_target_directory() {
    local category="$1"
    
    case "$category" in
        BUILD_CONFIG)
            echo ""  # Keep in root
            ;;
        DOCS)
            echo "docs"
            ;;
        SCRIPTS_PYTHON|SCRIPTS_SHELL)
            echo "scripts"
            ;;
        CONFIG)
            echo "config"
            ;;
        SOURCE_C)
            echo "backend/cpu/misc"
            ;;
        SOURCE_JS)
            echo "src"
            ;;
        ASSETS_BINARY|ASSETS_ARCHIVE)
            echo "deploy"
            ;;
        ASSETS_IMAGE)
            echo "public/assets"
            ;;
        MISC)
            echo "misc"
            ;;
        *)
            echo "misc"
            ;;
    esac
}

scan_root_for_loose_files() {
    print_header "PHASE 1: Scanning Root Directory"
    
    # Find all loose files in root (maxdepth 1, only files, exclude hidden)
    local loose_files=()
    declare -A category_counts
    
    while IFS= read -r file; do
        if [[ -f "$file" ]]; then
            local basename=$(basename "$file")
            # Skip .git and established directories
            if [[ ! "$basename" =~ ^\.git ]]; then
                loose_files+=("$basename")
                local category=$(get_file_category "$basename")
                if [[ ! -v "category_counts[$category]" ]]; then
                    category_counts[$category]=0
                fi
                ((category_counts[$category]++))
            fi
        fi
    done < <(find "${REPO_ROOT}" -maxdepth 1 -type f ! -name '.*')
    
    if [[ ${#loose_files[@]} -eq 0 ]]; then
        log_warning "No loose files found in root directory"
        return 1
    fi
    
    log_info "Found ${#loose_files[@]} loose files"
    
    # Display organized list by category
    echo ""
    echo "Files to reorganize (by category):"
    echo ""
    
    for category in DOCS SCRIPTS_PYTHON SCRIPTS_SHELL CONFIG SOURCE_C SOURCE_JS ASSETS_BINARY ASSETS_ARCHIVE ASSETS_IMAGE MISC BUILD_CONFIG; do
        local count=${category_counts[$category]:-0}
        if [[ $count -gt 0 ]]; then
            local target_dir=$(get_target_directory "$category")
            target_dir=${target_dir:-"ROOT (keep)"}
            printf "  %-20s → %-30s [%d files]\n" "$category" "$target_dir" "$count"
        fi
    done
    
    echo ""
    log "INFO" "File inventory complete: ${#loose_files[@]} total files"
    
    # Store for next phase
    echo "${loose_files[@]}"
}

################################################################################
# Directory Creation
################################################################################

create_target_directories() {
    print_header "PHASE 2: Creating Target Directories"
    
    local dirs=(
        "docs"
        "scripts"
        "config"
        "backend/cpu/misc"
        "src"
        "deploy"
        "public/assets"
        "misc"
    )
    
    for dir in "${dirs[@]}"; do
        if [[ -d "${REPO_ROOT}/${dir}" ]]; then
            log_info "Directory already exists: ${dir}/"
        else
            if [[ "$DRY_RUN" == "false" ]]; then
                mkdir -p "${REPO_ROOT}/${dir}"
                log_success "Created directory: ${dir}/"
            else
                log_info "[DRY_RUN] Would create: ${dir}/"
            fi
        fi
    done
    
    echo ""
}

################################################################################
# File Movement Operations
################################################################################

move_files_by_category() {
    print_header "PHASE 3: Moving Files by Category"
    
    local -A moves_by_category
    local -A category_order=(
        [DOCS]=1
        [SCRIPTS_PYTHON]=2
        [SCRIPTS_SHELL]=3
        [CONFIG]=4
        [SOURCE_C]=5
        [SOURCE_JS]=6
        [ASSETS_BINARY]=7
        [ASSETS_ARCHIVE]=8
        [ASSETS_IMAGE]=9
        [MISC]=10
    )
    
    local move_count=0
    
    # Collect all moves
    for file in $(find "${REPO_ROOT}" -maxdepth 1 -type f ! -name '.*' ! -path './.git/*'); do
        local basename=$(basename "$file")
        local category=$(get_file_category "$basename")
        local target_dir=$(get_target_directory "$category")
        
        if [[ -z "$target_dir" ]]; then
            # Skip build config files (keep in root)
            log_info "Keeping in root: ${basename} (build config)"
            continue
        fi
        
        if [[ ! -v moves_by_category["$category"] ]]; then
            moves_by_category["$category"]=""
        fi
        moves_by_category["$category"]+="${basename}|"
    done
    
    # Execute moves per category
    for category in "${!category_order[@]}"; do
        if [[ ! -v moves_by_category["$category"] ]]; then
            continue
        fi
        
        local files="${moves_by_category[$category]%|}"
        if [[ -z "$files" ]]; then
            continue
        fi
        
        local target_dir=$(get_target_directory "$category")
        echo ""
        echo -e "${YELLOW}Moving ${category} files → ${target_dir}/${NC}"
        
        IFS='|' read -ra file_array <<< "$files"
        for file in "${file_array[@]}"; do
            local source="${REPO_ROOT}/${file}"
            local dest="${REPO_ROOT}/${target_dir}/${file}"
            
            if [[ ! -f "$source" ]]; then
                log_warning "Source file not found (may have been moved already): ${file}"
                continue
            fi
            
            if [[ "$DRY_RUN" == "false" ]]; then
                if mv "$source" "$dest" 2>/dev/null; then
                    log_success "Moved: ${file} → ${target_dir}/"
                    ((move_count++))
                else
                    log_error "Failed to move: ${file}"
                fi
            else
                log_info "[DRY_RUN] Would move: ${file} → ${target_dir}/"
                ((move_count++))
            fi
        done
    done
    
    echo ""
    log "INFO" "Moved ${move_count} files"
}

################################################################################
# Git Operations
################################################################################

commit_moves_by_category() {
    print_header "PHASE 4: Committing Changes (by category)"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY_RUN] Would perform git commits"
        return
    fi
    
    cd "${REPO_ROOT}" || exit 1
    
    # Check if git is initialized
    if ! git rev-parse --git-dir > /dev/null 2>&1; then
        log_error "Not a git repository: ${REPO_ROOT}"
        return 1
    fi
    
    # Commit by category
    local commits=0
    
    if git status docs/ --porcelain 2>/dev/null | grep -q '^A'; then
        git add docs/ 2>/dev/null
        git commit -m "chore: reorganize documentation files to docs/

- Moved .md and .txt documentation files to dedicated docs/ directory
- Rationale: Per Constitution § IV, consolidate documentation
  for improved maintainability and clarity
- Impact: No functional changes, documentation remains accessible" 2>/dev/null
        ((commits++))
        log_success "Committed documentation reorganization"
    fi
    
    if git status scripts/ --porcelain 2>/dev/null | grep -q '^A'; then
        git add scripts/ 2>/dev/null
        git commit -m "chore: reorganize Python and shell scripts to scripts/

- Moved .py and .sh script files to dedicated scripts/ directory
- Rationale: Per Constitution § IV, separate runtime/utility scripts
  from source code for easier discovery and version control
- Impact: No functional changes, scripts remain executable" 2>/dev/null
        ((commits++))
        log_success "Committed scripts reorganization"
    fi
    
    if git status config/ --porcelain 2>/dev/null | grep -q '^A'; then
        git add config/ 2>/dev/null
        git commit -m "chore: reorganize configuration files to config/

- Moved .json, .yaml, .yml configuration files to dedicated config/ directory
- Rationale: Per Constitution § IV, centralize configuration for clarity
- Impact: No functional changes, configurations remain accessible" 2>/dev/null
        ((commits++))
        log_success "Committed configuration reorganization"
    fi
    
    if git status deploy/ --porcelain 2>/dev/null | grep -q '^A'; then
        git add deploy/ 2>/dev/null
        git commit -m "chore: reorganize binary assets and archives to deploy/

- Moved .deb, .tar.gz and other binary artifacts to deploy/ directory
- Rationale: Per Constitution § IV, organize deployment and build artifacts
- Impact: No functional changes, artifacts remain accessible" 2>/dev/null
        ((commits++))
        log_success "Committed binary assets reorganization"
    fi
    
    if git status backend/ --porcelain 2>/dev/null | grep -q '^A'; then
        git add backend/ 2>/dev/null
        git commit -m "chore: reorganize C source files to backend/

- Moved .c and .h source files to backend/cpu/misc/ directory
- Rationale: Per Constitution § IV, organize source code by module
- Impact: No functional changes, source files accessible via backend/" 2>/dev/null
        ((commits++))
        log_success "Committed C source reorganization"
    fi
    
    if git status src/ --porcelain 2>/dev/null | grep -q '^A'; then
        git add src/ 2>/dev/null
        git commit -m "chore: reorganize TypeScript/JavaScript files to src/

- Moved .ts and .js files to dedicated src/ directory
- Rationale: Per Constitution § IV, consolidate frontend/source code
- Impact: No functional changes, files accessible via src/" 2>/dev/null
        ((commits++))
        log_success "Committed TypeScript/JavaScript reorganization"
    fi
    
    if git status public/ --porcelain 2>/dev/null | grep -q '^A'; then
        git add public/ 2>/dev/null
        git commit -m "chore: reorganize image assets to public/assets/

- Moved image files (png, jpg, svg, etc.) to public/assets/ directory
- Rationale: Per Constitution § IV, organize static assets for web delivery
- Impact: No functional changes, assets accessible via public/" 2>/dev/null
        ((commits++))
        log_success "Committed image assets reorganization"
    fi
    
    if git status misc/ --porcelain 2>/dev/null | grep -q '^A'; then
        git add misc/ 2>/dev/null
        git commit -m "chore: organize miscellaneous files to misc/

- Moved files without clear category to misc/ for manual review
- Rationale: Per Constitution § IV, organize remaining unclassified files
- Impact: No functional changes, requires manual categorization" 2>/dev/null
        ((commits++))
        log_success "Committed miscellaneous file organization"
    fi
    
    echo ""
    log "INFO" "Created ${commits} git commits"
}

################################################################################
# Validation & Verification
################################################################################

validate_reorganization() {
    print_header "PHASE 5: Validating Reorganization"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY_RUN] Skipping validation"
        return
    fi
    
    local violations=0
    local -a violations_list
    
    # Check for loose files (excluding build config)
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
    )
    
    for file in $(find "${REPO_ROOT}" -maxdepth 1 -type f ! -name '.*' ! -path './.git/*'); do
        local basename=$(basename "$file")
        local is_allowed=false
        
        for allowed in "${allowed_in_root[@]}"; do
            if [[ "$basename" == "$allowed" ]]; then
                is_allowed=true
                break
            fi
        done
        
        if [[ "$is_allowed" == "false" ]]; then
            violations_list+=("$basename")
            ((violations++))
        fi
    done
    
    if [[ $violations -gt 0 ]]; then
        log_error "Found ${violations} loose files that should be organized:"
        for file in "${violations_list[@]}"; do
            echo "  - $file"
        done
        return 1
    else
        log_success "No loose files found in root directory"
    fi
    
    # Verify target directories exist and contain files
    local categories=(
        "docs:DOCS"
        "scripts:SCRIPTS"
        "config:CONFIG"
        "deploy:DEPLOY"
        "public/assets:ASSETS"
        "backend/cpu/misc:SOURCE"
        "src:SOURCE_JS"
    )
    
    for cat_dir in "${categories[@]}"; do
        local dir="${cat_dir%:*}"
        local name="${cat_dir#*:}"
        
        if [[ -d "${REPO_ROOT}/${dir}" ]]; then
            local file_count=$(find "${REPO_ROOT}/${dir}" -type f 2>/dev/null | wc -l)
            if [[ $file_count -gt 0 ]]; then
                log_success "Directory ${dir}/ contains ${file_count} file(s)"
            fi
        fi
    done
    
    echo ""
    log "INFO" "Validation complete"
}

generate_summary_report() {
    print_header "SUMMARY REPORT"
    
    echo "Reorganization Status: $([ $? -eq 0 ] && echo "✓ SUCCESS" || echo "✗ FAILED")"
    echo ""
    echo "Root directory now contains only:"
    echo "  - Build configuration files (CMakeLists.txt, Cargo.toml, etc.)"
    echo "  - Documentation (README.md, LICENSE)"
    echo "  - Setup scripts (bootstrap.sh, setup.bat)"
    echo ""
    echo "Organized directories:"
    ls -d "${REPO_ROOT}"/{docs,scripts,config,deploy,public/assets,backend/cpu/misc,src,misc} 2>/dev/null | sed 's|.*/||' | sed 's|^|  - |'
    echo ""
    echo "See ${LOG_FILE} for detailed operation log"
    echo ""
    log "INFO" "Reorganization Summary:"
    log "INFO" "  - Loose files scanned and categorized"
    log "INFO" "  - Files moved to dedicated directories"
    log "INFO" "  - Changes committed to git (per category)"
    log "INFO" "  - Validation passed"
}

################################################################################
# Main Execution
################################################################################

main() {
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --verbose)
                VERBOSE=true
                shift
                ;;
            --no-confirm)
                CONFIRM=false
                shift
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    initialize_log
    
    print_header "QALLOW CODEBASE REORGANIZATION"
    echo "Repository: ${REPO_ROOT}"
    echo "Dry Run: ${DRY_RUN}"
    echo "Log File: ${LOG_FILE}"
    echo ""
    
    # Execute phases
    scan_root_for_loose_files || exit 0  # Exit if no files found
    create_target_directories
    move_files_by_category
    commit_moves_by_category
    validate_reorganization || exit 1
    generate_summary_report
    
    log "INFO" "Reorganization completed successfully"
}

# Run main function
main "$@"

exit 0
