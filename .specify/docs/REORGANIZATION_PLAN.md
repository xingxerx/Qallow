# Qallow Codebase Reorganization Plan

**Date**: November 6, 2025  
**Status**: Inventory Complete  
**Constitution Alignment**: § IV (Modular Directory Structure)

---

## Executive Summary

**Total loose files**: 102 files in root  
**Reorganization strategy**: Group by type into 6 destination directories  
**Execution method**: Bash shell script with logging  
**Expected outcome**: Clean, maintainable structure per Constitution § IV

---

## File Inventory by Category

### 📋 DOCUMENTATION (60 files) → `docs/`

**Status & Summary Files** (24):
- AGENT_CHECKLIST.md
- AGENT_FIXES_COMPLETE.md
- AGENT_FIX_ROOT_CAUSE.md
- AGENT_NOW_ACTUALLY_FIXING.md
- AGENT_QUICK_START.md
- AGENT_RUNNING_NOW.md
- AGENT_SAFE_MODE.md
- AGENT_SAFE_MODE_COMPLETE.md
- AGENTS_COMPLETE_STATUS.md
- AGENTS_HOW_TO_RUN.md
- AGENTS_QUICK_START.md
- AGENTLIGHTNING_QUICK_GUIDE.md
- BUILD_SUCCESS_REPORT.md
- COMPLETION_REPORT_MEDIUM_LOW_PRIORITY.md
- DAEMON_FINAL_STATUS.md
- EVERYTHING_FIXED.md
- EXECUTION_COMPLETE.md
- FINAL_IMPLEMENTATION_SUMMARY.md
- FINAL_STATUS.md
- PRODUCTION_VERIFICATION_FINAL.md
- PROJECT_STATUS.md
- REORGANIZATION_COMPLETE.md
- VERIFICATION_COMPLETE.txt

**Build & Setup Guides** (14):
- BOOTSTRAP_IMPLEMENTATION_SUMMARY.md
- BUILD_AND_RUN.md
- HOW_TO_RUN.md
- QUICK_START_FULL_BUILD.md
- RUN_EXECUTION_INDEX.md
- RUN_FULL_BUILD_GUIDE.md
- RUN_QUICK_SUMMARY.txt
- START_HERE.md
- CI_BOOTSTRAP_INTEGRATION.md
- CI_FIX_COMPLETE.md
- SPECKIT_INSTALLATION_COMPLETE.txt
- SPECKIT_INTEGRATION_VERIFICATION.md
- TEST_DRIVEN_FIX_IMPLEMENTATION_COMPLETE.txt
- SEQUENTIAL_THINKING_CHANGES.txt

**Architecture & CLI Documentation** (10):
- CLI_COMMAND_TREE.md
- CLI_CONSOLIDATION_SUMMARY.md
- CLI_DOCUMENTATION_INDEX.md
- CLI_IMPLEMENTATION_CHECKLIST.md
- CLI_QUICK_REFERENCE.md
- COMMAND_CHEAT_SHEET.md
- DAEMON_QUICK_REF.md
- NAVIGATION_SIMULATOR_V2.2_COMPLETE.md
- UNIFIED_CLI.md
- UNIFIED_CLI_SUMMARY.md

**Spec & Reference Guides** (8):
- SPECKIT_INDEX.md
- SPECKIT_QUICK_REFERENCE.md
- SPECKIT_SETUP_GUIDE.md
- SCALING_ROADMAP_SUMMARY.txt
- TEST_DRIVEN_FIXES_QUICK.md
- TEST_DRIVEN_FIX_CODE_STRUCTURE.md
- TEST_DRIVEN_FIX_SYSTEM.md
- RUN_QUICK_SUMMARY.txt

**Core Project Docs** (4):
- README.md (stays in root per convention)
- CONSTITUTION.md (→ docs/CONSTITUTION.md)
- CONTRIBUTING.md (→ docs/CONTRIBUTING.md)
- SECURITY.md (→ docs/SECURITY.md)

### 🐍 PYTHON SCRIPTS (7 files) → `scripts/`

- advanced_error_fixer.py
- agentlightning_runner_safe.py
- cleanup_corrupted_python.py
- generate_b64.py
- real_cuda_processor.py
- recursive_improvement_engine.py
- run_phase11.py
- run_qallow.py
- test_quantum_complete.py
- test_virtual_computer.py

**Note**: `run_phase11.py` and `run_qallow.py` are runtime utilities; could also go in `src/` but scripts/ is more appropriate for executable utilities.

### ⚙️ BUILD & BUILD CONFIG (4 files) → Keep in Root or Move to `config/`

- CMakeLists.txt (stays in root - build system standard)
- Cargo.toml (stays in root - Rust standard)
- bootstrap.sh (stays in root - setup script standard)
- docker-compose.yaml (→ `config/docker/` or `deploy/`)

**Decision**: CMakeLists.txt, Cargo.toml, bootstrap.sh stay in root (conventional locations). Move docker-compose.yaml to `deploy/docker-compose.yaml` per Constitution § IV.

### 📦 DEPENDENCY SPECIFICATIONS (5 files) → `config/`

- requirements.txt (→ `config/requirements.txt`)
- requirements-dev.txt (→ `config/requirements-dev.txt`)
- requirements-gpu.txt (→ `config/requirements-gpu.txt`)
- requirements-web.txt (→ `config/requirements-web.txt`)
- DEPENDENCY_MANIFEST.md (→ `docs/DEPENDENCY_MANIFEST.md` - already created as part of Constitution update)

### 📊 DATA & LOGS (6 files) → `data/`

- ethics_demo.csv
- log_phase12.csv
- log_phase13.csv
- qallow_stream.csv
- test_phase13.csv
- agent_changes.json
- agent_daemon.log
- qallow_bench.log

**Note**: These are generated artifacts; should be in `data/` which already exists per architecture. Consolidate into `data/logs/` and `data/metrics/`.

### ✅ KEEP IN ROOT (3 files)

- CMakeLists.txt (build system standard)
- Cargo.toml (Rust standard)
- bootstrap.sh (setup standard)
- LICENSE (convention)
- .gitignore (convention)
- .github/ (convention)

---

## Target Directory Structure (Post-Reorganization)

```
Qallow/
├── docs/                           # All documentation
│   ├── *.md                        # Spec guides, setup, status reports
│   ├── CONSTITUTION.md
│   ├── CONTRIBUTING.md
│   ├── SECURITY.md
│   ├── DEPENDENCY_MANIFEST.md
│   ├── ARCHITECTURE_SPEC.md        # (already exists)
│   ├── QUICKSTART.md               # (already exists)
│   └── specs/                      # (already exists)
│
├── scripts/                        # Python utilities & dev scripts
│   ├── *.py                        # All .py scripts from root
│   └── build_all.sh                # (if exists in root)
│
├── config/                         # Configuration files
│   ├── requirements.txt
│   ├── requirements-dev.txt
│   ├── requirements-gpu.txt
│   ├── requirements-web.txt
│   └── docker/
│       └── docker-compose.yaml
│
├── data/                           # Generated data & logs (already exists)
│   ├── logs/
│   │   ├── *.log                   # Log files
│   │   └── *.csv                   # Phase/metrics CSVs
│   ├── metrics/
│   │   └── *.json                  # Metrics data
│   └── examples/
│       └── *.csv                   # Demo/example data
│
├── CMakeLists.txt                  # Root (stays)
├── Cargo.toml                      # Root (stays)
├── bootstrap.sh                    # Root (stays)
├── LICENSE                         # Root (stays)
├── README.md                       # Root (stays - no move)
├── .github/                        # Root (stays)
├── .specify/                       # Root (stays)
├── backend/                        # (unchanged)
├── src/                            # (unchanged)
├── tests/                          # (unchanged)
└── ...other directories
```

---

## Migration Strategy

### Phase 1: Create Target Directories
```bash
mkdir -p docs/
mkdir -p scripts/
mkdir -p config/
mkdir -p config/docker/
mkdir -p data/logs/
mkdir -p data/metrics/
```

### Phase 2: Move Files by Category

**Step 1: Documentation Files → docs/**
```bash
# Move all *.md files (except README.md, Dockerfile files, and pre-exists like docs/QUICKSTART.md)
find . -maxdepth 1 -type f -name "*.md" ! -name "README.md" -exec mv {} docs/ \;

# Move *.txt documentation files
mv EXECUTION_SUMMARY.txt RUN_QUICK_SUMMARY.txt FINAL_SERVER_SUMMARY.txt \
   IMPLEMENTATION_COMPLETE.txt VERIFICATION_COMPLETE.txt \
   SPECKIT_INSTALLATION_COMPLETE.txt TEST_DRIVEN_FIX_IMPLEMENTATION_COMPLETE.txt \
   SEQUENTIAL_THINKING_CHANGES.txt docs/
```

**Step 2: Python Scripts → scripts/**
```bash
mv *.py scripts/
mv run_full_build.sh scripts/ 2>/dev/null || true
```

**Step 3: Config Files → config/**
```bash
mv requirements*.txt config/
mv docker-compose.yaml config/docker/
```

**Step 4: Data Files → data/**
```bash
mv *.csv data/logs/ 2>/dev/null || true
mv *.log data/logs/
mv agent_changes.json data/metrics/
```

### Phase 3: Verification

**Check for remaining loose files:**
```bash
find . -maxdepth 1 -type f \( -name "*.md" -o -name "*.py" -o -name "*.sh" -o -name "*.csv" -o -name "*.json" \) \
  ! -name "README.md" ! -name "CMakeLists.txt" ! -name "Cargo.toml" ! -name "bootstrap.sh"
```

---

## Impact Assessment

### ✅ Benefits
1. **Constitution § IV Compliance**: Directory structure now canonical and organized
2. **Improved Discoverability**: Files grouped by purpose; easier to locate
3. **Maintainability**: Clear separation of concerns (docs/, scripts/, config/)
4. **CI/CD Integration**: Configuration grouped in config/; easier to reference
5. **Version Control**: Cleaner root; reduced visual clutter in git status

### ⚠️ Considerations
1. **Script References**: Update any hardcoded paths in CMakeLists.txt, shell scripts, or CI workflows
2. **Python Imports**: Ensure Python path includes scripts/ if modules are imported
3. **Documentation Links**: Update internal cross-references (e.g., links in README.md → docs/)
4. **Build System**: Verify CMakeLists.txt finds requirements files in config/

---

## Files Summary

| Category | Source | Destination | Count |
|----------|--------|-------------|-------|
| Documentation | ./*.md | docs/ | 41 |
| Status/Reports | ./*.txt | docs/ | 8 |
| Python Scripts | ./*.py | scripts/ | 10 |
| Config Files | requirements*.txt | config/ | 4 |
| Docker Config | docker-compose.yaml | config/docker/ | 1 |
| Build System | CMakeLists.txt, Cargo.toml, bootstrap.sh | Root | 3 |
| Data/Logs | *.csv, *.log, *.json | data/{logs,metrics} | 8 |
| **TOTAL** | | | **75** |

**Remaining in root** (unchanged):
- README.md
- LICENSE
- CMakeLists.txt
- Cargo.toml
- bootstrap.sh
- .github/, .specify/, etc.

---

## Execution Plan

1. **Review this plan** ✓
2. **Create bash reorganization script** (next step)
3. **Execute script with logging** (tracks every move)
4. **Verify completion** (scan for remaining loose files)
5. **Update build system** (if needed: CMakeLists.txt references)
6. **Git commit** with message referencing Constitution § IV

---

## Next Steps

Execute `.specify/scripts/reorg.sh` (to be created) which will:
1. Create all target directories
2. Move files by category with logging
3. Report any errors or conflicts
4. Generate final `reorg.log` with full audit trail

Ready to proceed? ✅
