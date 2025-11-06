# Implementation Plan: Codebase Reorganization

**Branch**: `001-codebase-reorganization` | **Date**: 2025-11-06 | **Spec**: `docs/specs/001-codebase-reorganization/spec.md`  
**Input**: Feature specification defining loose file reorganization per Constitution § IV

---

## Summary

Qallow repository root contains ~17 loose files that should be organized into dedicated directories per Constitution § IV canonical structure. This plan defines technical approach to move files while preserving content, git history, and audit trail.

---

## Technical Context

**Language/Version**: Bash (shell scripting) | Pure POSIX-compatible shell, no external dependencies  
**Primary Dependencies**: None (bash, find, mv, git built-in)  
**Storage**: Local filesystem operations; no databases or network I/O  
**Testing**: Shell script validation + git log inspection + checksum verification  
**Target Platform**: Linux (primary development environment)  
**Project Type**: Single repository reorganization (monorepo structure)  
**Performance Goals**: < 5 seconds total execution time (small files)  
**Constraints**: No concurrent operations; atomic commits per category  
**Scale/Scope**: 17 loose files → 5-7 target directories

---

## Constitution Check (§ I-VIII)

- [x] **§ I (Library-First)**: Not directly applicable (no new library code)
- [x] **§ II (Test-First)**: Validation script tests reorganization; tests defined before implementation
- [x] **§ III (Minimal Dependencies)**: Pure bash; no external dependencies required
- [x] **§ IV (Modular Structure)**: **CORE GOAL** – Implement canonical structure per Constitution § IV
- [x] **§ V (Spec-Driven)**: This spec drives the implementation
- [x] **§ VI (Text I/O Observability)**: All operations logged to `reorg.log` (text format)
- [x] **§ VII (Versioning)**: Git commits per category with clear messages
- [x] **§ VIII (Simplicity)**: MVP-focused; pure shell without frameworks

**Gate Status**: ✅ PASS – All principles aligned

---

## Project Structure

### Documentation (this feature)

```text
docs/specs/001-codebase-reorganization/
├── spec.md                 # This spec
├── plan.md                 # This plan
├── research.md             # Background (if needed)
├── contracts/              # File move contracts (if needed)
└── tasks.md                # Task breakdown
```

### Source Code (implementation)

```text
scripts/
├── reorganize.sh           # Main reorganization script (NEW)
├── validate.sh             # Validation script (NEW)
└── [other scripts]         # Existing scripts

docs/
├── specs/                  # Specs directory
├── [reorganized files]     # Documentation files moved here

config/
├── [reorganized files]     # Config files moved here (NEW dir if needed)

public/assets/
├── [reorganized files]     # Binary assets moved here (NEW dir if needed)

reorg.log                   # Audit trail of all operations
```

---

## Detailed Implementation Strategy

### Phase 1: Inventory & Planning (Pre-execution)

**Script**: `scripts/reorganize.sh` (Phase 1)

1. Scan root directory for loose files
2. Classify files by extension and content
3. Generate mapping of file → target directory
4. Create categorized list of moves (by category)
5. Generate pre-execution report (show what will move where)

**Output**: Visible categorized list of pending moves

### Phase 2: Create Target Directories

**Script**: `scripts/reorganize.sh` (Phase 2)

1. Create `docs/` if missing
2. Create `scripts/` if missing
3. Create `config/` if missing
4. Create `public/assets/` if missing
5. Create `misc/` if missing
6. Log all directory creation actions

**Output**: All target directories exist with correct permissions

### Phase 3: Move Files by Category

**Script**: `scripts/reorganize.sh` (Phase 3 - multiple passes)

Execute separate pass for each category:

1. **Pass 1: Documentation files** (*.md, *.txt)
   - Move to `docs/`
   - Commit: "chore: reorganize documentation files to docs/"

2. **Pass 2: Python scripts** (*.py)
   - Move to `scripts/`
   - Commit: "chore: reorganize Python scripts to scripts/"

3. **Pass 3: Configuration files** (.json, .yaml, .yml)
   - Move to `config/`
   - Commit: "chore: reorganize configuration files to config/"

4. **Pass 4: Binary & asset files** (.deb, .tar.gz)
   - Move to `public/assets/` or `deploy/`
   - Commit: "chore: reorganize binary assets"

5. **Pass 5: Shell scripts** (*.sh, but preserve bootstrap.sh in root as build artifact)
   - Move to `scripts/` (if not in root already)
   - Commit: "chore: reorganize shell scripts to scripts/"

6. **Pass 6: Source files** (*.c, *.h not in backend/interface)
   - Move to `backend/cpu/misc/` or evaluate context
   - Commit: "chore: reorganize source files"

7. **Pass 7: Unknown/miscellaneous** (files without clear category)
   - Move to `misc/` with warning
   - Commit: "chore: organize miscellaneous files to misc/"

**Output**: All files moved, logged, and committed per category

### Phase 4: Validation & Verification

**Script**: `scripts/validate.sh` (Phase 4)

1. Scan root directory for remaining loose files
2. Verify target directories contain expected files
3. Generate checksum validation report
4. Verify git commits created successfully
5. Check for any broken symlinks

**Output**: Validation report confirming reorganization complete

---

## File Categorization Rules

| Pattern | Category | Target Dir | Notes |
|---------|----------|------------|-------|
| `*.md`, `*.txt` | Documentation | `docs/` | All documentation consolidated |
| `*.py` | Python scripts | `scripts/` | Utility and development scripts |
| `*.sh` | Shell scripts | `scripts/` | Except bootstrap.sh (build artifact) |
| `*.json`, `*.yaml`, `*.yml` | Configuration | `config/` | Config files for tools and services |
| `*.c`, `*.h` | C/CUDA source | `backend/cpu/misc/` | C source files (context-dependent) |
| `*.js`, `*.ts` | TypeScript/JavaScript | `src/` | Frontend/browser code |
| `*.deb`, archives | Binary assets | `deploy/` or `public/assets/` | Large binaries, package files |
| `image/*` | Image assets | `public/assets/` | PNG, JPG, SVG, etc. |
| Build config | Build artifacts | Root | CMakeLists.txt, Cargo.toml, Makefile, Dockerfile, setup.bat |
| Unknown extension | Miscellaneous | `misc/` | Fallback; manual review recommended |

---

## Special Cases & Exceptions

| File | Handling | Reason |
|------|----------|--------|
| `CMakeLists.txt` | Keep in root | Build orchestration |
| `Cargo.toml`, `Cargo.lock` | Keep in root | Rust project manifest |
| `Makefile` | Keep in root | Build target definition |
| `Dockerfile` | Keep in root | Container definition |
| `bootstrap.sh` | Keep in root | Build initialization script |
| `setup.bat` | Keep in root | Windows setup script |
| `LICENSE` | Keep in root | Legal requirement (convention) |
| `README.md` | Keep in root | Project overview (top-level) |
| `Qallow.code-workspace` | Keep in root | VS Code workspace config |
| `ask` | Move to `misc/` | Executable with no extension (unclear purpose) |
| `ncu` | Move to `misc/` | Executable with no extension (unclear purpose) |
| `phase4_demo.c` | Move to `backend/cpu/demos/` | Demonstration/example code |
| `qallow.tar.gz` | Move to `deploy/` or `public/assets/` | Archive artifact |
| `cuda-keyring_1.1-1_all.deb` | Move to `deploy/` | System package |
| `reorg.log` | Keep in root or move to `docs/` | Audit trail; keep accessible |

---

## Git Commit Strategy

Each file category is committed separately to create clear, reviewable history:

```bash
# Example commits
git add docs/
git commit -m "chore: reorganize documentation files to docs/

- SCALING_ROADMAP_SUMMARY.txt → docs/
- [Other .md, .txt files] → docs/

Rationale: Per Constitution § IV, consolidate documentation
in dedicated directory for maintainability and clarity."

git add scripts/
git commit -m "chore: reorganize Python scripts to scripts/

- Various .py files → scripts/
- Shell utilities → scripts/

Rationale: Separate runtime scripts from source code,
enabling easier discovery and version control."

# ... similar commits for each category
```

---

## Rollback & Contingency

If reorganization causes issues:

```bash
# Revert all commits
git revert --no-edit <first-reorg-commit>..<last-reorg-commit>

# Or reset to pre-reorganization state
git reset --hard <commit-before-reorg>
```

Audit trail in `reorg.log` enables manual reconstruction if needed.

---

## Implementation Artifacts

### Files to Create

1. **`scripts/reorganize.sh`** (350 lines)
   - Main reorganization script with logging
   - Phase-based execution (inventory → create dirs → move → validate)
   - Preserves file metadata (permissions, timestamps)
   - Logs all operations to `reorg.log`

2. **`scripts/validate.sh`** (200 lines)
   - Post-reorganization validation script
   - Scans for loose files
   - Verifies target directories
   - Generates validation report

3. **`docs/REORGANIZATION_LOG.md`** (auto-generated)
   - Summary of reorganization
   - List of files moved with new locations
   - Commit hash references

### Files to Update

1. **`CMakeLists.txt`** (if needed)
   - Update any hardcoded file references to new locations
   - Example: `add_subdirectory(scripts)` if scripts become part of build

2. **`.gitignore`** (if needed)
   - Update patterns if new directories created
   - Add `reorg.log` if not already ignored

### Files to Create (Directories)

- `docs/` (may already exist; create if missing)
- `scripts/` (may already exist; create if missing)
- `config/` (likely missing; create)
- `public/assets/` (likely missing; create)
- `misc/` (create as catch-all)

---

## Testing & Validation Strategy

### Pre-Execution Tests

- [ ] Verify all files in inventory list
- [ ] Validate git is initialized and clean
- [ ] Confirm sufficient disk space
- [ ] Check file permissions are readable

### Post-Execution Tests

- [ ] Verify 0 loose files remain in root (except allowed exceptions)
- [ ] Confirm all moves completed successfully
- [ ] Check `reorg.log` contains all operations
- [ ] Verify checksums match (content preservation)
- [ ] Validate git commits created
- [ ] Run `git log` to confirm commit messages

### Regression Tests

- [ ] Build succeeds (`cmake --build build`)
- [ ] Tests pass (`ctest`)
- [ ] No broken imports or references
- [ ] All scripts remain executable

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| File permission issues | Low | Medium | Pre-check permissions; use `chmod` if needed |
| Disk space insufficient | Very Low | High | Check `df` before execution |
| Git conflicts | Very Low | Medium | Ensure clean working tree before running |
| Build system breaks | Low | High | Test build post-reorganization; have rollback ready |
| Scripts lose executability | Low | Low | Verify `chmod +x` on moved scripts |
| Symlink breakage | Low | Medium | Pre-scan for symlinks; handle specially if found |

---

## Success Criteria Mapping to Implementation

| Success Criterion | Implementation Method |
|---|---|
| SC-001: 100% files moved | Script returns success only when all categorized files moved |
| SC-002: 0 loose files remain | Validation script scans root and reports findings |
| SC-003: reorg.log complete | Every operation logged with timestamp and status |
| SC-004: Content preserved | Checksum comparison before/after (optional post-move) |
| SC-005: Permissions preserved | Use `cp -p` or `rsync -a` semantics (or native `mv`) |
| SC-006: Logical git commits | One commit per file category with descriptive messages |
| SC-007: Validation passes | `validate.sh` exits with code 0 if all checks pass |
| SC-008: No broken symlinks | Validation script checks for symlink integrity |
| SC-009: Fast execution | Script optimized; all operations local filesystem |
| SC-010: Atomic per category | Each category committed before next begins |

---

## Timeline & Milestones

| Phase | Duration | Milestone |
|-------|----------|-----------|
| 1. Review & approve spec | 1 day | Spec ✅ |
| 2. Implement scripts | 1 day | Scripts ready for testing |
| 3. Test on staging (if applicable) | 0.5 day | All tests pass |
| 4. Execute reorganization | < 5 min | Script runs; `reorg.log` generated |
| 5. Validate & commit | 0.5 day | All commits pushed; CI passes |
| **Total** | **~3 days** | Reorganization complete |

---

**Status**: ✅ READY FOR TASK BREAKDOWN PHASE

**Next**: Follow with `/tasks` command to generate detailed task breakdown for implementation execution.
