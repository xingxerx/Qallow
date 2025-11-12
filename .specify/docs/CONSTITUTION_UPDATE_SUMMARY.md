# Constitution Update Complete ✅

## What Was Updated

The Qallow Project Constitution has been amended from **v2.2 → v3.0.0**, establishing binding governance for modular, test-first development with explicit observability and dependency requirements.

**Files Modified**:
- `.specify/memory/constitution.md` – Updated with 8 core principles, canonical directory structure, and binding compliance rules
- `.specify/CONSTITUTION_SYNC_REPORT.md` – Generated detailed sync report with template alignment analysis and action items

---

## 8 Core Binding Principles

### I. **Library-First Modular Architecture**
Every feature is a standalone, independently testable library:
- Placed in `backend/{cpu|cuda}/{name}/`, `python/{name}/`, or `src/{service}/`
- Exposes clean contracts via headers (.h) or public APIs
- Each library has clear, concrete purpose (no organizational-only utilities)

### II. **Test-First Development (NON-NEGOTIABLE)**
Red-Green-Refactor cycle is mandatory; tests written first, must fail initially:
- Unit tests: `tests/{module_name}/` mirrored from source
- Integration tests: `tests/integration/`
- CI gates: 0 failures + 100% pass rate; **exceptions require written rationale**

### III. **Minimal Dependencies & Explicit Coupling**
- Baseline: CMake, gcc ≥11, Python ≥3.10 only
- Every external dependency must be documented in `DEPENDENCY_MANIFEST.md` with rationale
- No implicit transitive dependencies; all imports explicitly listed
- Python versions locked in `requirements-{base,dev,gpu,web}.txt`

### IV. **Modular Directory Structure (CANONICAL)**
Project layout now binding:

```
backend/{cpu,cuda}/{module}/     # Phase implementations (ethics, quantum, etc.)
core/include/                     # Shared contracts & types
include/qallow/                   # Public API
interface/                        # Orchestration + UI
python/{module}/                  # Python bridges & utilities
src/{cli,runtime,telemetry}/      # Services & CLI
tests/{module_name}/              # Tests mirroring source
docs/specs/{number}-{name}/       # Spec-driven artifacts
scripts/                          # Build & dev utilities (NOT deployment)
deploy/                           # Kubernetes, Docker (separate from scripts/)
```

**Key Rule**: No loose files in root except CMakeLists.txt, package configs, LICENSE, README, bootstrap.sh

### V. **Spec-Driven Development with Observability**
- Every major feature: `docs/specs/{number}-{name}/spec.md` + `plan.md`
- Specs define user stories & acceptance criteria (not implementation)
- Telemetry integrated: `QALLOW_PROFILE_SCOPE` macros, `qallow_log_*` functions
- All metrics → `src/runtime/telemetry_outputs.c` → `data/logs/` (CSV, JSON)

### VI. **Observability & Testability Through Text I/O**
- External APIs MUST use text protocols (JSON, CSV, CLI args/stdout)
- **Zero opaque binary formats** for external interfaces
- Every execution produces human-readable logs in `data/logs/`
- Structured logging: `qallow_log_info`, `qallow_log_error` with context tags

### VII. **Versioning & Breaking Changes**
- Semantic versioning: MAJOR.MINOR.PATCH
  - **MAJOR**: API/contract changes, phase removals, architecture shifts
  - **MINOR**: New phases, new observability, non-breaking extensions
  - **PATCH**: Bugfixes, performance, docs, internal refactors
- Reflected in: `CMakeLists.txt`, `pyproject.toml`, phase bridge contracts
- Breaking changes require: deprecation (1 minor release), migration guide

### VIII. **Simplicity & YAGNI Principle**
- MVP first: deliver core value with minimal scaffolding
- Measure before optimizing; no pre-optimization
- Defer speculative features to separate specs
- Each phase: clear entry point, deterministic logic, isolated state

---

## What Changed from v2.2

| Aspect | v2.2 | v3.0.0 |
|--------|------|--------|
| Test-First Status | Advisory best-practice | **NON-NEGOTIABLE** binding gate |
| Dependencies | Not explicitly managed | **NEW**: § III explicit coupling rules + DEPENDENCY_MANIFEST.md |
| Directory Structure | Descriptive (existing layout) | **CANONICAL** (binding rules for all new code) |
| Observability | Mentioned in phase docs | **NEW § VI**: Binding text I/O requirement; zero opaque binary formats |
| Versioning | Implicit (v2.2 informal) | **NEW § VII**: MAJOR.MINOR.PATCH with propagation rules |
| Simplicity | Not addressed | **NEW § VIII**: YAGNI principle binding |
| Code Quality | 7 principles | **8 principles** (added I, III, VI, VII, VIII; reorganized) |
| Versioning Decision | PATCH updates (v2.0 → v2.2) | **MAJOR** (v2.2 → v3.0.0 due to backward-incompatible governance shift) |

---

## Immediate Actions Required

### 🔴 CRITICAL (This Week)

1. **Create `DEPENDENCY_MANIFEST.md`**
   - Inventory all external dependencies (CMake, gcc, CUDA, Python packages, cirq)
   - For each: document § III rationale (why needed, transitive depth check)
   - Example:
     ```
     ## External Dependencies
     - **CMake ≥3.20**: Build orchestration (no alternative; standard practice)
     - **gcc ≥11 | clang ≥15**: C compilation (no third-party alternative viable)
     - **CUDA 12.0+**: GPU acceleration (optional; feature flag for CPU-only)
     - **cirq**: Quantum circuit execution (optional; only loaded when QALLOW_cirq=1)
     ```

2. **Audit Root Directory**
   - Identify loose Python scripts: `run_phase11.py`, `run_qallow.py`, etc.
   - Decision: Move to `scripts/` (development utilities) or `src/` (runtime services)
   - Update `CMakeLists.txt` install targets accordingly

3. **Update `README.md`** → Add "Project Structure" section
   - Reference § IV canonical layout
   - Link to `docs/ARCHITECTURE_SPEC.md`
   - Clarify: backend/{cpu|cuda} vs. python/ vs. src/ purpose

### 🟡 HIGH (Next Week)

1. **Update `docs/QUICKSTART.md`**
   - Add § V (Spec-Driven) workflow
   - Link to `.specify/templates/spec-template.md`
   - Example: "New feature? Start with `docs/specs/{number}-{name}/spec.md`"

2. **CMakeLists.txt Audit**
   - Verify `project(Qallow VERSION 3.0.0)` reflects new semantic versioning
   - Remove hardcoded paths; use `${CMAKE_BINARY_DIR}`, `${PROJECT_SOURCE_DIR}`
   - Document VERSION propagation rule (§ VII)

3. **Add Constitution Compliance Checklist**
   - Prepend to `.specify/templates/checklist-template.md` (if exists)
   - Or create `.specify/COMPLIANCE_CHECKLIST.md`:
     ```
     - [ ] § I: Library placed in canonical directory
     - [ ] § II: Tests written first; red-green-refactor documented
     - [ ] § III: New dependencies in DEPENDENCY_MANIFEST.md with rationale
     - [ ] § IV: No loose module files in root
     - [ ] § V: Spec referenced (if feature); § VI observability integrated
     - [ ] § VI: External APIs use text protocols (JSON, CSV, CLI)
     - [ ] § VII: Version bumps follow MAJOR.MINOR.PATCH; breaking changes have migration guide
     - [ ] § VIII: MVP focused; no speculative code
     ```

4. **Review Build Scripts**
   - `scripts/build_all.sh`: Verify no references to deprecated paths
   - Add pre-build validation: check for loose module files in root (§ IV enforcement)

### 🟢 MEDIUM (Ongoing)

1. **Monitor PR Reviews** – Use § I-VIII compliance checklist
2. **Track Directory Migration** – Document progress on loose files cleanup
3. **Prepare Q1 2026 Tooling** – Plan automated linting for § IV enforcement (no cross-module includes except via core/)

---

## Template Alignment Status

### ✅ Already Aligned (No Changes Required)
- `.specify/templates/plan-template.md` – Constitution Check gate already present; § I-IV reference naturally
- `.specify/templates/spec-template.md` – User stories + acceptance criteria support § II test-first; add optional telemetry reminder
- `.specify/templates/tasks-template.md` – Phase structure + independent testing aligns with § I-II; implicit guidance sufficient

### ⚠️ Pending Review
- `.specify/templates/checklist-template.md` – Add § I-VIII compliance checkboxes
- `.specify/templates/agent-file-template.md` – Cross-check for § III dependency documentation requirements

---

## Compliance Checklist for All PRs

Starting today, all pull requests must include:

```markdown
## Constitution Compliance (§ I-VIII)

- [ ] **§ I** (Library-First): New code placed in canonical dirs (backend/{cpu|cuda}, python/, src/, scripts/)
- [ ] **§ II** (Test-First): Tests written first; red-green-refactor cycle followed; 0 failures
- [ ] **§ III** (Minimal Deps): New dependencies documented in DEPENDENCY_MANIFEST.md with rationale
- [ ] **§ IV** (Modular Structure): No loose module files in root; CMakeLists.txt reflects structure
- [ ] **§ V** (Spec-Driven): Spec referenced if feature; observability integrated
- [ ] **§ VI** (Text I/O): External APIs use JSON/CSV/CLI; no opaque binary formats
- [ ] **§ VII** (Versioning): Version bumps follow MAJOR.MINOR.PATCH; breaking changes include migration guide
- [ ] **§ VIII** (Simplicity): MVP focused; speculative code deferred to separate specs

**Constitution Version**: 3.0.0 | **Verified By**: [Author/Reviewer]
```

---

## Roadmap

### **Immediate** (This week)
✅ Constitution updated to v3.0.0  
⏳ DEPENDENCY_MANIFEST.md creation  
⏳ README.md structure section  
⏳ Root directory loose file audit  

### **Q1 2026**
- Automated linting enforces § IV module isolation
- Legacy path references deprecated in build scripts
- All features follow new spec structure in docs/specs/

### **Q2 2026**
- Unified test framework consolidates redundant runners under centralized `tests/` entry point
- Quarterly compliance report on § I-VIII adherence

---

## Reference Documents

- **Constitution**: `.specify/memory/constitution.md`
- **Sync Report**: `.specify/CONSTITUTION_SYNC_REPORT.md` (detailed impact analysis)
- **Phase Architecture**: `docs/ARCHITECTURE_SPEC.md` (phase reference table)
- **Quickstart**: `docs/QUICKSTART.md` (workflow guidance)
- **Spec Template**: `.specify/templates/spec-template.md` (spec-driven workflow)

---

## Suggested Git Commit

```
docs: amend constitution to v3.0.0 (modular governance + observability)

Major version bump establishes binding governance for modular, test-first development:

- § I (Library-First): Bind canonical directory structure (backend/{cpu|cuda}, python/, src/)
- § II (Test-First): Elevate to NON-NEGOTIABLE gating criterion; red-green-refactor mandatory
- § III (Minimal Deps): NEW principle; explicit coupling rules + DEPENDENCY_MANIFEST.md requirement
- § IV (Modular Structure): NEW principle; canonicalize project layout with binding rules
- § V-VIII: Refined spec-driven, observability, versioning, and simplicity principles

All future PRs require § I-VIII compliance checklist. Directory structure now canonical;
loose module files must migrate to designated directories. Breaking: Dependencies now
explicitly managed; all transitive chains must be justified.

BREAKING: Tests required before code ships; no exceptions without written rationale.
Directory structure is canonical; loose module code must migrate.

Closes: #[ISSUE_NUMBER] (if applicable)
```

---

## Questions?

Refer to:
1. **Constitution principles** (§ I-VIII) → `.specify/memory/constitution.md`
2. **Implementation roadmap** → `.specify/CONSTITUTION_SYNC_REPORT.md` (immediate actions section)
3. **Development workflow** → `docs/QUICKSTART.md` + `.specify/templates/spec-template.md`
4. **Directory structure** → Constitution § IV + `docs/ARCHITECTURE_SPEC.md`

**Status**: ✅ READY FOR ADOPTION
