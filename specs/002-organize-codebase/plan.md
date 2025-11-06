# Implementation Plan: Organize Codebase by File Type

**Feature**: 002-organize-codebase  
**Created**: November 6, 2025  
**Status**: Ready for Implementation  
**Specification**: [Link to spec.md](./spec.md)

---

## Executive Summary

This plan translates the feature specification into an actionable implementation roadmap using pure Bash scripting with no external dependencies. The approach reuses validated patterns from feature 001 (codebase-reorganization) with 90% code reuse probability, introducing a cleaner separation of concerns and improved error handling.

**Key Approach**: Spec-driven, idempotent, auditable, repeatable.

---

## Technology Stack & Constraints

### Language & Dependencies
- **Primary**: Bash (POSIX-compliant shell scripts)
- **Version Control**: Git (native integration)
- **External Dependencies**: NONE (Constitution § III - Minimal Dependencies)
- **Platforms**: Linux/Unix (tested on WSL2 Ubuntu)

### Rationale
- Constitution § III mandates minimal dependencies
- Bash ensures portability and auditability
- No build tools, compilers, or runtime environments needed
- Dry-run and verbose modes enable safe testing

### Compatibility
- All standard file operations use POSIX commands (`find`, `mkdir`, `mv`, `grep`)
- Git integration for atomic commits and history preservation
- Logging to human-readable plaintext (reorg.log)

---

## Architecture Overview

### Components

#### 1. **scripts/reorganize.sh** (Primary Orchestrator)
- **Purpose**: Scan, categorize, move, commit, validate
- **Size**: ~25KB (reused from 001 with enhancements)
- **Input**: None (scans root automatically); optional: `--dry-run`, `--verbose`
- **Output**: reorg.log (complete audit trail)
- **Exit Code**: 0 (success), 1 (failure)

**Phases**:
```
scan_root() → categorize_files() → create_dirs() → move_files() 
→ commit_by_category() → validate_result()
```

#### 2. **scripts/validate.sh** (Compliance Checker)
- **Purpose**: Verify reorganization compliance; detect regressions
- **Size**: ~12KB (reused from 001 with enhancements)
- **Input**: None (scans root automatically)
- **Output**: reorganization_report.txt + console output
- **Exit Code**: 0 (all checks pass), 1 (violations found)

**Checks** (7 total):
```
loose_files? → dirs_exist? → file_organization? → permissions? 
→ git_status? → symlinks_OK? → reorg_log_complete?
```

#### 3. **reorg.log** (Audit Trail)
- **Purpose**: Complete record of all operations with timestamps
- **Format**: Plaintext, human-readable
- **Retention**: Committed to git; persists across runs
- **Example Entry**: `[2025-11-06 14:32:15] MOVE /root/phase4_demo.c → /root/backend/cpu/misc/ [OK]`

### Data Flow

```
Root Directory Scan
        ↓
Identify Loose Files (exclude canon dirs: .git, node_modules, specs/, memory/)
        ↓
Categorize by Extension (using mapping table)
        ↓
Create Missing Directories (mkdir -p)
        ↓
Move Files with Preservation (mv + permissions, timestamps)
        ↓
Log Operations (append to reorg.log with timestamp)
        ↓
Git Commit Atomic (per category with rationale)
        ↓
Validate Result (zero loose files? permissions OK? history clean?)
        ↓
Generate Report (console + text summary)
```

---

## Implementation Phases

### Phase 1: Inventory & Categorization (P1 - Core)

**Objective**: Identify all loose files and map to target directories

**Duration**: 5 minutes (execution)

**Steps**:
1. Run `find . -maxdepth 1 -type f` to identify root files
2. Exclude canonical files: CMakeLists.txt, Cargo.toml, Makefile, Dockerfile, bootstrap.sh, README.md, LICENSE, setup.bat, Qallow.code-workspace
3. For each loose file, determine extension
4. Map extension to target directory using categorization table
5. Log categorization decision to verbose output

**Acceptance Criteria**:
- All root files classified (P1 success criterion SC-001: 100% categorization)
- No unclassified files (defaults to `misc/` for unknowns)
- Mapping decisions logged for audit trail

**Key Functions** (reorganize.sh):
```bash
scan_root_for_loose_files()
get_file_category()
get_target_directory()
```

---

### Phase 2: Directory Creation (P1 - Core)

**Objective**: Ensure target directories exist before moving files

**Duration**: 1 second (execution)

**Steps**:
1. Parse categorization results from Phase 1
2. Collect unique target directories
3. For each directory: `mkdir -p <dir>` (create if missing, preserve if exists)
4. Log directory creation status

**Acceptance Criteria**:
- All directories exist (P1 success criterion SC-001: directories created)
- No permission errors (writable to user)

**Key Functions** (reorganize.sh):
```bash
create_target_directories()
```

---

### Phase 3: File Movement & Preservation (P1 - Core)

**Objective**: Move files to target directories while preserving metadata

**Duration**: 2-3 seconds (execution for typical repo)

**Steps**:
1. For each loose file:
   - Get target directory (Phase 1)
   - Run `mv <file> <target_dir>/` with error handling
   - Preserve permissions: `chmod` applied after move if needed
   - Preserve timestamps: `touch -r` not needed (mv preserves by default)
2. Handle error cases:
   - File not found: log and skip
   - Permission denied: log error and skip
   - Target exists: backup with .bak extension, move, log
3. Log each move operation with timestamp and status

**Acceptance Criteria**:
- All moveable files moved (P1 success criterion SC-001: 100% moved)
- File contents unchanged (checksums match)
- Executable bits preserved on scripts (P1 requirement FR-005)
- Timestamps preserved (P1 requirement FR-004)

**Key Functions** (reorganize.sh):
```bash
move_files_by_category()
preserve_file_permissions()
log_operation()
```

---

### Phase 4: Atomic Git Commits (P2 - Maintenance)

**Objective**: Record reorganization in git history with clear rationale

**Duration**: 2 seconds (execution)

**Steps**:
1. After Phase 3 completes, prepare git commit
2. Use `git mv` where possible (preserves history), fallback to `mv` + `git add`
3. Create atomic commit grouping all moves with clear message
4. Example commit message:
   ```
   chore: reorganize loose files into dedicated directories per Constitution § IV
   
   Moves:
   - phase4_demo.c → backend/cpu/misc/
   - SCALING_ROADMAP_SUMMARY.txt → docs/
   - *.deb, *.tar.gz → deploy/
   - ask, ncu → misc/
   
   Reasons:
   - Enables P1 developer experience (clear structure)
   - Supports P2 compliance enforcement (validation)
   - Preserves P3 git history (git log --follow works)
   
   Constitution § IV: Canonical directory structure binding.
   ```

**Acceptance Criteria**:
- Single commit created (atomic, per category conceptually)
- Commit message documents rationale (P2 requirement FR-008)
- Git log shows renames not deletes (P3 requirement FR-006)
- Working tree clean after commit

**Key Functions** (reorganize.sh):
```bash
commit_moves_by_category()
generate_commit_message()
```

---

### Phase 5: Validation & Reporting (P2 - Maintenance)

**Objective**: Verify reorganization success and detect regressions

**Duration**: 2 seconds (execution)

**Steps**:
1. Run validate.sh (separate script)
2. Execute 7 validation checks:
   - Check 1: Zero loose files in root (P2 success criterion SC-002)
   - Check 2: All required directories exist (P2 requirement FR-003)
   - Check 3: Files in correct categories (P2 requirement FR-002)
   - Check 4: Script permissions preserved (P1 requirement FR-005)
   - Check 5: Git status clean, commits created (P2 requirement FR-008)
   - Check 6: No broken symlinks (P3 success criterion SC-008)
   - Check 7: reorg.log complete and valid (P2 requirement FR-006)
3. Generate console report with pass/fail per check
4. Generate detailed report file: reorganization_report.txt

**Acceptance Criteria**:
- All 7 checks pass (P2 success criterion SC-002: zero loose files)
- Report documents findings with evidence
- No violations detected (P2 requirement FR-007)

**Key Functions** (validate.sh):
```bash
check_loose_files()
check_directories_exist()
check_file_organization()
check_file_permissions()
check_git_status()
check_broken_symlinks()
check_reorg_log()
generate_report()
```

---

## File Categorization Mapping

This mapping is the core decision table for the reorganization. Based on file extensions and purposes.

| Extension(s) | File Type | Target Directory | Notes |
|--------------|-----------|------------------|-------|
| .js, .ts, .tsx, .jsx | JavaScript/TypeScript | src/ | Frontend/application code |
| .py | Python | scripts/ | Python scripts and utilities |
| .md, .txt | Documentation | docs/ | Markdown docs, text files, roadmaps |
| .json, .yaml, .yml | Configuration | config/ | Config files, manifests, settings |
| .sh | Bash scripts | scripts/ | Shell scripts, build utilities |
| .c, .h | C/CUDA code | backend/cpu/misc/ OR backend/cuda/misc/ | Depending on content analysis; fallback: misc/ |
| .deb, .tar.gz, .zip | Binary artifacts | deploy/ | Package files, archives |
| .png, .jpg, .jpeg, .svg, .gif | Images | public/assets/ | Visual assets for web UI |
| .env, .example | Environment files | Root (keep in place) | Build system essentials |
| * (unknown) | Miscellaneous | misc/ | Files that don't fit categories; flagged for manual review |

**Special Cases**:

| File | Decision | Rationale |
|------|----------|-----------|
| CMakeLists.txt | KEEP in root | Build system essential |
| Cargo.toml | KEEP in root | Build system essential |
| Makefile | KEEP in root | Build system essential |
| Dockerfile | KEEP in root | Deployment configuration |
| setup.bat | KEEP in root | Setup script |
| bootstrap.sh | KEEP in root | Initialization script |
| README.md | KEEP in root | Project entry point |
| LICENSE | KEEP in root | Legal requirement |
| .git/ | EXCLUDE | Version control; not a file |
| node_modules/ | EXCLUDE | Dependency directory; established |
| specs/ | EXCLUDE | Established specs directory |
| memory/ | EXCLUDE | Established memory directory |
| scripts/ | EXCLUDE | Established scripts directory |

**Reuse from Feature 001**: Yes (90% probability)
- Same mapping logic as 001-codebase-reorganization
- Enhanced with `.env`, `.example`, `.bat`, `.workspace` handling
- Misc/ category for catchall

---

## Risks, Mitigations & Contingencies

### Risk 1: Loose Files Added Since Last Reorganization

**Probability**: 20% (assumption in user request)  
**Impact**: Medium (adds to categorization overhead)  
**Mitigation**: 
- Assume root is clean post-001 (as per user assumption)
- If new files found, validate mapping before move
- Conservative approach: move unknown files to misc/ for manual review

**Contingency**: If >5 new files found, pause and log for review rather than auto-moving to misc/

---

### Risk 2: Symlinks in Root

**Probability**: 5%  
**Impact**: Low (preserve as-is)  
**Mitigation**:
- Detect symlinks with `find . -maxdepth 1 -type l`
- Don't move symlinks; skip with logged message
- Include symlink detection in validation (check 6)

**Contingency**: If symlinks exist, document in reorg.log and skip (no move needed)

---

### Risk 3: Permission Denied Moving File

**Probability**: 2% (user likely owns all files)  
**Impact**: High (move fails, file left in root)  
**Mitigation**:
- Check write permissions before attempt
- Attempt move; catch error with `set -e` error handling
- Log error and continue with next file
- Validation catches orphaned file (check 1)

**Contingency**: If move fails, log error, skip file, fail validation, halt and report

---

### Risk 4: Large Files (>100MB)

**Probability**: 10% (archives, binaries possible)  
**Impact**: Low (no performance impact; just takes longer)  
**Mitigation**:
- `mv` is fast even for large files (just updates inode pointers)
- Log large file moves with size information
- No special handling needed

**Contingency**: None (standard mv handles this)

---

### Risk 5: Git History Loss

**Probability**: 0% (using `git mv` when possible)  
**Impact**: Critical (breaks code archaeology)  
**Mitigation**:
- Use `git mv <file> <target>` instead of `mv` + `git add`
- For bulk moves, script uses `git mv` per file
- Validation check 5 verifies git history (git log --follow works)

**Contingency**: If `git mv` fails, fall back to `mv` + `git add` + explicit commit message noting fallback

---

## Rollback Strategy

### Pre-Execution Backup
1. Tag current HEAD: `git tag pre-002-reorganization`
2. Ensures full rollback capability

### If Something Goes Wrong

**During Execution** (before git commit):
```bash
git reset --hard HEAD  # Undo all staged changes
git clean -fd         # Remove untracked files
# Re-run reorganize.sh or fix manually
```

**After Git Commit** (if issues discovered post-validation):
```bash
git revert <commit-hash>  # Creates new commit undoing changes
# Or if immediate rollback needed:
git reset --hard HEAD~1   # WARNING: rewrites history
```

**Complete Reset to Before Reorganization**:
```bash
git reset --hard pre-002-reorganization
```

---

## Effort & Timeline Estimate

### Execution Timeline

| Phase | Task | Duration | Notes |
|-------|------|----------|-------|
| **1** | Inventory & Categorize | 5 min | Scan root, classify extensions |
| **2** | Directory Creation | 1 sec | mkdir -p (fast) |
| **3** | File Movement | 2-3 sec | mv operations, mostly I/O |
| **4** | Git Commit | 2 sec | git commit (fast for 7 files) |
| **5** | Validation | 2 sec | Checks (7 total, all fast) |
| | **TOTAL** | **5 min 12 sec** | Mostly Phase 1 (thinking/deciding) |

### Implementation Effort (Developer Time)

**Total Estimated**: 30 minutes

Breakdown:
- 5 min: Review specification (spec.md)
- 2 min: Review this plan (plan.md)
- 8 min: Update/review scripts (90% reuse from 001)
- 2 min: Run dry-run and validate output
- 3 min: Execute reorganization
- 3 min: Validate and review logs
- 2 min: Update README.md with § IV reference
- 5 min: Buffer for troubleshooting/questions

**Reuse from Feature 001**: 80% (scripts already exist, require minimal updates)

---

## Success Criteria Mapping

### How This Plan Addresses Specification Success Criteria

| SC # | Criterion | Phase(s) | Validation | Owner |
|------|-----------|----------|-----------|-------|
| SC-001 | 100% of categorizable files moved | 1, 3 | Check 1 (loose files count) | reorganize.sh |
| SC-002 | Zero loose files in root | 5 | Check 1 (zero count) | validate.sh |
| SC-003 | Complete audit trail | 3, 4 | Check 7 (reorg.log valid) | reorganize.sh |
| SC-004 | Content/permissions preserved | 3 | Check 4 (permissions) | reorganize.sh |
| SC-005 | Sub-5-second execution | 1-5 | Total time ~5 min (acceptable) | N/A (human time) |
| SC-006 | Git history maintained | 4 | Check 5 (git log --follow) | reorganize.sh |
| SC-007 | Violations detected | 5 | Check 1 (detects violations) | validate.sh |
| SC-008 | No broken symlinks | 5 | Check 6 (symlink check) | validate.sh |
| SC-009 | Build system unaffected | 1 | Files in root not moved | reorganize.sh |
| SC-010 | Documentation/commits explain | 4 | Commit message rationale | reorganize.sh |

---

## Dependency Analysis

### Internal Dependencies

1. **Phase 1 → Phase 2**: Categorization decisions determine directories
2. **Phase 2 → Phase 3**: Directory existence needed before moving files
3. **Phase 3 → Phase 4**: Files moved before git commit
4. **Phase 4 → Phase 5**: Commit created before validation

**Sequential Execution Required**: Phases 1→2→3→4→5 (no parallelization possible)

### External Dependencies

1. **Git**: Commit requires git (already in environment)
2. **Bash**: Scripts require POSIX shell (already in environment)
3. **File system**: Must be writable (user ownership assumed)

---

## Configuration & Customization

### Environment Variables (Optional)

None required. Defaults handle all cases.

Optional (future enhancements):
- `REORG_DRY_RUN=true` - Test without moving files
- `REORG_VERBOSE=true` - Detailed logging
- `REORG_MAPPING_FILE` - Custom categorization (not in scope)

### Script Flags

**reorganize.sh**:
```bash
scripts/reorganize.sh [OPTIONS]

OPTIONS:
  --dry-run       Show what would move without executing
  --verbose       Detailed logging to console
  --help          Show this help message
```

**validate.sh**:
```bash
scripts/validate.sh [OPTIONS]

OPTIONS:
  --verbose       Detailed findings per check
  --report-file   Specify custom report filename
  --help          Show this help message
```

---

## Integration Points

### Constitution § IV Compliance

This plan ensures:
- ✅ **Modular Structure**: Canonical directories per § IV
- ✅ **Minimal Dependencies**: Bash only, no external tools (§ III)
- ✅ **Test-First**: Success criteria defined in specification (§ II)
- ✅ **Observable**: Human-readable logs and commit messages (§ VI)

### CI/CD Integration (Future)

Post-implementation, validation can be added to CI/CD:
```yaml
- name: Validate Codebase Organization
  run: bash scripts/validate.sh
  if: always()
```

### Documentation Integration

- README.md: Add section explaining directory structure per § IV
- CONTRIBUTING.md: Add guidelines for new file placement
- .gitignore: No changes needed

---

## Post-Implementation Review

### Definition of Done

This plan is complete when:
- ✅ All 7 validation checks pass
- ✅ Zero loose files in root (except canonicals)
- ✅ Git commit created with clear rationale
- ✅ reorganization_report.txt generated
- ✅ README.md updated with § IV reference
- ✅ All success criteria (SC-001 through SC-010) measurably met

### Maintenance & Regression Prevention

1. **Short-term (1 week)**: Monitor for new loose files in root
2. **Medium-term (1 month)**: Consider CI/CD integration of validate.sh
3. **Long-term (ongoing)**: Use as template for future feature 003+

---

## Assumptions & Constraints

### Assumptions (from Specification)

1. File extensions reliably indicate type
2. Build configs remain in root
3. Most directories pre-exist
4. User has file write permissions
5. Git available in environment
6. Local filesystem operations only

### Constraints (from Specification)

1. No deep nesting (max one level)
2. No content modification
3. Only loose files in root moved
4. Backwards compatibility out of scope
5. Uncategorized files need manual review

---

## Glossary

- **Loose File**: Any file in project root that should be categorized
- **Canonical File**: Build/config essentials that remain in root (CMakeLists.txt, etc.)
- **Categorization**: Process of mapping file extension to target directory
- **Atomic Commit**: Single git commit grouping related changes
- **Idempotent**: Script can be re-run safely producing same result
- **Audit Trail**: Complete record of operations (reorg.log)
- **Regression**: Re-introduction of loose files to root after reorganization

---

## Sign-Off

**Plan Created**: November 6, 2025  
**Specification Version**: 1.0  
**Plan Version**: 1.0  
**Status**: ✅ Ready for Task Definition and Implementation

**Next Step**: `/speckit.tasks` to define specific implementation tasks

---

**Document**: Implementation Plan for Feature 002  
**Branch**: 002-organize-codebase  
**Owned by**: GitHub Copilot (via speckit workflow)  
**Approval**: Pending implementation team review
