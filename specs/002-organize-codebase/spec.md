# Feature Specification: Organize Codebase by File Type

**Feature Branch**: `002-organize-codebase`  
**Created**: November 6, 2025  
**Status**: Draft  
**Input**: Organize existing codebase by moving loose files into dedicated directories: group by type (.js/.ts → src/, .py → scripts/, .md/.txt → docs/, configs → config/, assets → public/assets/). No nesting beyond one level. Enable Spec-Driven iterations, improve maintainability, version control. Preserve all content unchanged.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Developer Discovers Clear Project Structure (Priority: P1)

A new developer clones the repository and immediately understands where different types of files are located. Instead of seeing a cluttered root directory with mixed file types, they find organized subdirectories where code, configuration, documentation, and assets are logically grouped.

**Why this priority**: This is the foundation for improving developer experience. Without a clear structure, new team members waste time searching for files and developers make mistakes placing files in wrong locations. This is blocking for all other improvements.

**Independent Test**: Can be fully tested by having a new developer (or test user) review the root directory structure and locate specific file types (documentation, configuration, source code) without external guidance, then verify they can understand the project layout in under 2 minutes.

**Acceptance Scenarios**:

1. **Given** a developer opens the repository root, **When** they look at the directory structure, **Then** they immediately see logical categories (backend/, src/, docs/, scripts/, config/, deploy/, etc.) without feeling confused about file placement
2. **Given** a developer wants to find Python scripts, **When** they navigate to a standard location, **Then** they find all .py files organized in dedicated directories (scripts/, python/)
3. **Given** a developer wants to find configuration files, **When** they check the config/ directory, **Then** they find all .json, .yaml, .env related configuration files organized together
4. **Given** a developer wants to find documentation, **When** they look at docs/, **Then** they find all .md and .txt documentation files in one accessible location

---

### User Story 2 - Maintainer Verifies Codebase Compliance (Priority: P2)

A project maintainer runs a validation check to ensure all loose files have been properly organized according to the project's canonical structure. This enables automated regression prevention and prevents future developers from adding unorganized files to the root directory.

**Why this priority**: This enables repeatable, automated verification. Once loose files are organized, we need assurance that they stay organized. This prevents entropy from accumulating in the root directory over time as more contributors add files.

**Independent Test**: Can be fully tested by running a validation script that scans the root directory for files that don't belong there (according to categorization rules) and reports which files are misplaced, enabling the feature to work independently.

**Acceptance Scenarios**:

1. **Given** all loose files have been moved to proper directories, **When** a maintainer runs validation, **Then** the validation reports zero loose files in the root directory
2. **Given** a new loose file is added to the root, **When** validation is run, **Then** it detects and reports the violation, enabling catch-and-fix before merge
3. **Given** the validation script executes, **When** all directories are properly structured, **Then** the script reports success with a summary of file counts per directory

---

### User Story 3 - Git History Preserved Through Reorganization (Priority: P3)

When files are moved to new locations, the git history for those files is preserved, allowing developers to use `git blame`, `git log`, and other history tools without losing context about past changes to moved files.

**Why this priority**: Preserving git history is valuable for code archaeology and understanding the evolution of specific files. However, the primary value comes from having organized structure (P1) and enforcing it (P2). History preservation is a "nice to have" that ensures we don't lose valuable context.

**Independent Test**: Can be fully tested by verifying that after reorganization, `git log --follow` and `git blame` still show the complete history of previously moved files, confirming that git operations reflect continuous file evolution rather than deletion/recreation.

**Acceptance Scenarios**:

1. **Given** a file has been moved from root to a new directory, **When** running `git log --follow filename`, **Then** it shows the complete history before and after the move
2. **Given** developers need to understand when a file was last modified, **When** they use `git blame` on a moved file, **Then** it shows accurate historical annotations
3. **Given** the reorganization is complete, **When** git status is checked, **Then** it shows renames rather than delete + add operations where possible

---

### User Story 3 - [Brief Title] (Priority: P3)

[Describe this user journey in plain language]

**Why this priority**: [Explain the value and why it has this priority level]

**Independent Test**: [Describe how this can be tested independently]

**Acceptance Scenarios**:

1. **Given** [initial state], **When** [action], **Then** [expected outcome]

---

[Add more user stories as needed, each with an assigned priority]

### Edge Cases

<!--
  ACTION REQUIRED: The content in this section represents placeholders.
  Fill them out with the right edge cases.
-->

- What happens when [boundary condition]?
- How does system handle [error scenario]?

## Requirements *(mandatory)*

<!--
  ACTION REQUIRED: The content in this section represents placeholders.
  Fill them out with the right functional requirements.
---

### Edge Cases

- **What happens when some loose files can't be categorized?** Files that don't fit standard categories (ask, ncu, etc.) are placed in `misc/` directory with clear documentation explaining their purpose
- **How does the system handle files with multiple valid categories?** Use primary purpose (e.g., if a file contains both code and documentation, categorize by primary content type; document the decision)
- **What if root contains hidden files or symlinks?** Skip hidden files (`.gitignore`, `.env` stay in root if intentional), handle symlinks with special care (preserve as symlinks, don't break references)
- **What about directory permissions and executable bits?** Preserve file permissions and executable bits when moving (use `cp -p` or equivalent; verify with shell scripts)
- **What if loose files exceed 100MB?** Use standard file operations; large files should still be reorganized (they may be binary artifacts, logs, or archives that belong in deploy/ or misc/)

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST scan the project root directory and identify all loose files (excluding .git/, node_modules/, specs/, memory/, scripts/ established directories)
- **FR-002**: System MUST categorize identified loose files by type using a mapping table: file extension → target directory
- **FR-003**: System MUST create target directories if they don't already exist (backend/cpu/misc/, backend/cuda/misc/, python/, src/, scripts/, docs/, config/, deploy/, public/assets/, misc/)
- **FR-004**: System MUST move files to their target directories using standard file operations, preserving file contents, timestamps, and permissions
- **FR-005**: System MUST preserve executable bits on script files during move operations
- **FR-006**: System MUST generate an audit log (reorg.log) recording every file operation with timestamp, source path, target path, and status
- **FR-007**: System MUST validate reorganization by verifying zero loose files remain in root after operation completes
- **FR-008**: System MUST support atomic commits per file category, with clear commit messages explaining rationale
- **FR-009**: System MUST handle edge cases gracefully: missing source files, permission errors, pre-existing target files
- **FR-010**: System MUST enable repeatable execution through dry-run mode and verbose logging

### Key Entities

- **Loose File**: Any file in project root that should be categorized (identified by extension, name, or type)
- **File Category**: Classification grouping files by type (code, documentation, configuration, assets, scripts, deployment artifacts, miscellaneous)
- **Target Directory**: Canonical location for each file category per project structure
- **File Move Operation**: Atomic operation transferring file from root to target directory while preserving metadata
- **Audit Log Entry**: Record of a file operation including timestamp, source, destination, and result status

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% of categorizable loose files in root are moved to proper directories (measure: `find root -maxdepth 1 -type f | wc -l` returns only essential build/config files)
- **SC-002**: Zero loose files remain in root after reorganization completes (measure: validation script confirms no violations)
- **SC-003**: Complete audit trail exists for all file operations (measure: reorg.log contains entry for each moved file with timestamp and result)
- **SC-004**: All file content, permissions, and timestamps are preserved (measure: checksums match before/after; executable bits remain on scripts)
- **SC-005**: Reorganization completes in under 5 seconds for typical repository (measure: execution time logged; <5 sec for ~20 files)
- **SC-006**: Git history is maintained for moved files (measure: `git log --follow` shows continuous history; no delete+add operations where move was possible)
- **SC-007**: Validation script successfully identifies any newly added loose files (measure: adding test file to root and running validation detects it)
- **SC-008**: Zero broken symlinks or dangling references after move (measure: `find . -type l -! -exec test -e {} \;` returns no results)
- **SC-009**: Build system and runtime operations are unaffected (measure: existing build/run tests pass after reorganization)
- **SC-010**: Documentation and commit messages clearly explain file organization rationale (measure: EXECUTION_REPORT.md exists with full explanation)

## Assumptions

- **File Extension Mapping**: Assumes extensions reliably indicate file type (e.g., .py files are Python scripts, .md files are Markdown documentation). Special cases documented in categorization table.
- **Root Directory Stability**: Assumes build configuration files (CMakeLists.txt, Cargo.toml, Makefile, Dockerfile, setup.bat) remain in root and are not moved (these are build system essentials).
- **Directory Pre-existence**: Most target directories (docs/, scripts/, config/) already exist; script creates missing ones rather than assuming all must be created from scratch.
- **Permission Preservation**: Assumes shell environment has sufficient permissions to move all files and preserve attributes (may fail if running as restricted user).
- **Git History Priority**: Uses `git mv` where possible to preserve history; falls back to standard `mv` if git operations fail, with explicit logging.
- **No Network Operations**: Reorganization uses only local filesystem operations; no remote repository pushes during initial move (pushed separately after validation).
- **Idempotency**: Script can be re-run safely (moves already-moved files to same location with logged message; doesn't duplicate).
- **File Naming Conventions**: Assumes ASCII-compatible filenames; special character handling is not in scope (documented as limitation).

## Constraints & Limitations

- **No Deep Nesting**: Files placed at most one level deep (e.g., `backend/cpu/misc/` is deepest; no `backend/cpu/phase-implementations/x86/misc/`)
- **No Content Modification**: Files are moved as-is; no transformation, conversion, or content analysis performed
- **Scope**: Only loose files in project root are moved; pre-organized subdirectories (backend/, src/, etc.) are not touched or reorganized
- **Backwards Compatibility**: Any scripts or tools that hard-code file paths must be updated separately (out of scope)
- **Manual Decisions**: Files that genuinely don't fit categories (ask, ncu) go to `misc/` but require manual review for proper categorization (documented in EXECUTION_REPORT)

## Integration Points

- **Version Control (Git)**: Reorganization integrates with git history preservation; commits are created per category
- **Build System (CMake, Cargo)**: CMakeLists.txt and Cargo.toml remain in root; no build system changes required
- **CI/CD Pipeline**: Validation script can be integrated into CI to prevent regression (future enhancement)
- **Editor/IDE**: File paths change; any editor configurations with hard-coded paths require updates (documented in EXECUTION_REPORT)
- **Documentation**: README.md updated to reflect new directory structure (out of current scope; future task)

## Success Definition

✅ **Feature is successful when**:
1. All 10 success criteria are measurably met
2. Validation script confirms zero loose files in root
3. Git history is clean with 2 commits (one per file category) and clear messages
4. Execution report documents all decisions, edge cases, and future maintenance steps
5. Reusable scripts (reorganize.sh, validate.sh) are available for future use
6. Constitution § IV modular structure is fully implemented and verified
