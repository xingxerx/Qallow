# Feature Specification: Codebase Reorganization

**Feature Branch**: `001-codebase-reorganization`  
**Created**: 2025-11-06  
**Status**: Draft  
**Input**: Organize existing codebase by moving loose files into dedicated directories per Constitution § IV

---

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Developer discovers clear project structure (Priority: P1)

A developer clones Qallow and opens the repository root. They immediately understand the project organization:
- Build configuration files (CMake, Cargo) at root (expected)
- All documentation consolidated in `docs/`
- All scripts in `scripts/`
- Source code organized in module directories (`backend/`, `python/`, `src/`)
- Assets organized in `public/assets/` or `deploy/`

**Why this priority**: First impression of project maintainability; enables faster onboarding and reduces confusion about file locations.

**Independent Test**: 
- Verify no loose `.md`, `.txt`, `.py`, `.c`, `.deb`, or asset files in root
- Confirm `docs/`, `scripts/`, `config/` directories contain appropriate files
- Validate each directory contains only expected file types (no nesting confusion)

**Acceptance Scenarios**:

1. **Given** developer opens repository, **When** they examine root directory, **Then** they see only CMakeLists.txt, Cargo.toml, Dockerfile, LICENSE, README.md, bootstrap.sh, Makefile, and build artifacts (no loose scripts or documentation)

2. **Given** developer needs a utility script, **When** they search for Python files, **Then** they find them organized in `scripts/` directory (not scattered in root)

3. **Given** developer needs project documentation, **When** they navigate to `docs/`, **Then** they find all markdown/text documentation files organized logically

4. **Given** developer needs configuration files, **When** they navigate to `config/`, **Then** they find all config files (.json, .yaml) in one place

---

### User Story 2 - Maintainer verifies file organization compliance (Priority: P2)

A maintainer runs a validation check to ensure no loose files violate the new directory structure:
- All scripts in `scripts/`
- All documentation in `docs/`
- All configs in `config/`
- All C source files in `backend/` or `interface/`
- Project structure matches Constitution § IV

**Why this priority**: Enables automated compliance checks in CI/CD; prevents future regressions.

**Independent Test**:
- Run validation script that scans root for non-conforming files
- Generate report of any violations found
- Provide clear instructions for remediation

**Acceptance Scenarios**:

1. **Given** reorganization complete, **When** validation script runs, **Then** it reports "No loose files found" + list of all organized files with their new locations

2. **Given** a developer accidentally adds a new loose file to root, **When** validation runs, **Then** it clearly identifies the violation and suggests correct location

3. **Given** validation passes, **When** CI/CD runs, **Then** the check succeeds and allows merge (no blocking violations)

---

### User Story 3 - Git history preserved with clear commit record (Priority: P3)

A developer examining git history sees organized commits:
- Each file category moved in separate, logical commits
- Commit messages clearly indicate what moved and why
- `reorg.log` provides detailed audit trail of all operations

**Why this priority**: Enables future developers to understand reorganization rationale; useful for debugging or reverting if needed.

**Independent Test**:
- Verify git log shows logical commits per file category
- Confirm `reorg.log` contains all move operations with timestamps
- Validate commit messages reference Constitution § IV

**Acceptance Scenarios**:

1. **Given** reorganization complete, **When** developer runs `git log`, **Then** they see commits like "chore: move documentation files to docs/" + "chore: move Python scripts to scripts/"

2. **Given** developer needs to understand why a file moved, **When** they check `reorg.log`, **Then** they see timestamp, source, destination, and category

3. **Given** reorganization causes issue, **When** developer examines commit, **Then** they can easily identify all files moved in that batch

---

### Edge Cases

- **Empty directories**: If `scripts/`, `docs/`, etc. already exist but contain no files, the script creates them only if files need to be moved there
- **Hidden files**: Skip files starting with `.` (already git-ignored)
- **Large binary files**: `qallow.tar.gz`, `.deb` files move to appropriate category; ensure no truncation
- **Symbolic links**: Treat symlinks as regular files; preserve link target
- **Files without extensions**: Place in `misc/` directory if no mapping rule applies
- **Case sensitivity**: Preserve original case; mapping is case-insensitive for extension matching

---

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST scan root directory and identify all loose files (excluding .git, build/, venv/, node_modules/, .specify/, and established subdirectories)
- **FR-002**: System MUST apply mapping rules to categorize files:
  - `.md`, `.txt` → `docs/`
  - `.py` → `scripts/`
  - `.c`, `.h` → `backend/` or `interface/` (determine based on content/context)
  - `.js`, `.ts` → `src/`
  - `.json`, `.yaml`, `.yml` → `config/`
  - `.sh` → `scripts/`
  - `.deb`, image files, assets → `public/assets/` or `deploy/`
  - Build config (CMakeLists.txt, Cargo.toml, Makefile, Dockerfile) → remain in root
  - Unknown extensions → `misc/`
- **FR-003**: System MUST create target directories if they do not exist
- **FR-004**: System MUST move files using standard shell operations (`mv` command)
- **FR-005**: System MUST log every operation to `reorg.log` with timestamp, source file, destination directory, and status
- **FR-006**: System MUST preserve all file content, permissions, and metadata (no corruption)
- **FR-007**: System MUST group moves by file category and commit separately (e.g., one commit for docs, one for scripts)
- **FR-008**: System MUST generate validation report confirming no loose files remain in root (except allowed exceptions)
- **FR-009**: System MUST be executable via bash script with no external dependencies (pure shell operations)
- **FR-010**: System MUST provide clear error messages if move operations fail (e.g., permission denied, disk full)

### Key Entities *(if data involved)*

- **File**: Object with name, extension, current path (root), target directory, size, last modified date
- **Category**: File type classification (docs, scripts, config, assets, build, source, misc) with associated target directory
- **MoveOperation**: Record of file move including timestamp, source, destination, status (success/failed), error message
- **ValidationReport**: Audit trail showing all loose files before reorganization, all moves performed, final state

---

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% of loose files successfully categorized and moved to appropriate directories
- **SC-002**: 0 files remain in root directory (except CMakeLists.txt, Cargo.toml, README.md, LICENSE, Dockerfile, Makefile, bootstrap.sh, setup.bat, Qallow.code-workspace, and hidden files)
- **SC-003**: `reorg.log` contains complete audit trail with timestamp and status for every file move operation
- **SC-004**: All file content preserved (verified via checksum or file size comparison before/after)
- **SC-005**: All permissions preserved (files maintain executable bit if present)
- **SC-006**: Git history preserved with logical commits per file category (max 3-5 commits for full reorganization)
- **SC-007**: Validation script runs successfully and confirms "All files organized correctly"
- **SC-008**: Zero broken symlinks or missing files after reorganization
- **SC-009**: Reorganization script executes in < 5 seconds (files are small/medium; network not involved)
- **SC-010**: 100% completion rate—no partial failures or rolled-back operations (atomic per category)

---

## Assumptions

- **Root directory state**: Assumed to contain loose files that should be organized per Constitution § IV
- **File types**: Extensions used for categorization (e.g., `.py` → Python script); edge cases handled manually if needed
- **No conflicts**: Assumed no existing files in target directories with same names as files being moved (move operation would fail if conflict exists)
- **Git available**: Assumed git is initialized and available for commit operations
- **Bash shell**: Assumed bash is available as default shell (no PowerShell/Zsh-specific syntax)
- **Permissions**: Assumed user has permission to read, move, and commit files
- **Disk space**: Assumed sufficient disk space available (moves are local filesystem operations)
- **Immutability**: Assumed reorganization is one-time operation; build artifacts (build/, venv/) regenerated post-reorganization
- **No automation conflicts**: Assumed no concurrent file system operations or automated tools running during reorganization

---

## Notes & Constraints

- **Constitution § IV Compliance**: All moves must align with canonical directory structure defined in Constitution
- **Non-destructive**: All file content must be preserved; verify via checksums if possible
- **Reversible**: Git commits should allow easy revert if needed
- **Scalability**: Script should work for current ~17 loose files; handle edge cases for future growth
- **Documentation**: Each move category should be documented in commit message with rationale

---

## Specification Quality Checklist

- [x] User scenarios prioritized (P1/P2/P3)
- [x] Independent testing defined for each scenario
- [x] Acceptance scenarios use Given-When-Then format
- [x] Edge cases identified (empty dirs, hidden files, large binaries, symlinks)
- [x] Functional requirements testable and specific (FR-001 through FR-010)
- [x] Key entities defined (File, Category, MoveOperation, ValidationReport)
- [x] Success criteria measurable and technology-agnostic (SC-001 through SC-010)
- [x] Assumptions documented (9 key assumptions listed)
- [x] Constraints and compliance requirements clear
- [x] Spec ready for planning phase

---

**Status**: ✅ READY FOR PLANNING PHASE
