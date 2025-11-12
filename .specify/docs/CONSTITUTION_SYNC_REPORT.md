# Constitution Sync Impact Report
**Generated**: 2025-11-06  
**Constitution Version**: 3.0.0 (Previous: 2.2)

---

## Version Change & Rationale

### Version Bump: 2.2 → 3.0.0 (MAJOR)

**Rationale**: Backward-incompatible restructuring of project governance:
- Principles reorganized from advisory guidelines to binding canonical structure
- New modular directory structure (Section IV) becomes **mandatory** (not aspirational)
- Test-first elevated from best-practice to non-negotiable gating criterion
- Explicit dependencies policy introduced as new principle (Section III)
- Three new principles added addressing critical gaps: Observability (VI), Versioning (VII), Simplicity (VIII)

This constitutes a major governance shift affecting all future development and code review practices.

---

## Modified Principles

| Principle | Change | Impact |
|-----------|--------|--------|
| **I. Library-First** | Expanded to bind directory structure convention (backend/{cpu\|cuda}, python/, src/cli/) | All new features must follow placement rules; existing scattered code requires migration plan |
| **II. Test-First** | Elevated from guidance to **NON-NEGOTIABLE**; red-green-refactor cycle now binding | No exceptions; exceptions require written rationale in commit; CI gates enforce 100% pass rate |
| **III. Minimal Dependencies** | NEW principle; explicit coupling rules, DEPENDENCY_MANIFEST.md requirement | Review all transitive dependencies; document rationale for each external library |
| **IV. Modular Directory Structure** | NEW principle; canonicalizes Qallow's actual structure with binding rules | Immediately applicable; loose files in root must migrate; update CMakeLists.txt references |
| **V. Spec-Driven** | Refined; now explicitly tied to telemetry and CSV/JSON outputs | Existing specs must add observability requirements; telemetry schema must be documented |
| **VI. Observability & Text I/O** | NEW principle; zero-opaque-binary-formats rule, structured logging | Audit all external APIs; migrate binary protocols where applicable; add CSV/JSON exporters |
| **VII. Versioning** | NEW principle; MAJOR.MINOR.PATCH rules for library contracts and breaking changes | Update VERSION in CMakeLists.txt, pyproject.toml; define contract versions for phase modules |
| **VIII. Simplicity & YAGNI** | NEW principle; defers speculative code, emphasizes MVP value delivery | Audit TODO/FIXME comments; move speculative features into separate specs |

---

## Added Sections

1. **Section IV: Modular Directory Structure** – Canonicalizes project layout with explicit rules for module placement, tests organization, and root-level cleanup
2. **Directory Reorganization Roadmap** – Phases immediate actions (establish structure, move loose files) with Q1/Q2 2026 tooling enforcement milestones
3. **Versioning Guidance (Section VII)** – Defines MAJOR/MINOR/PATCH semantics and propagation paths (CMakeLists.txt, pyproject.toml, phase contracts)

---

## Removed Sections

None. Previous constitution (v2.2) content preserved; new principles added orthogonally.

---

## Template Alignment Status

### `.specify/templates/plan-template.md` ✅ ALIGNED
- **Constitution Check** section (line 25) already gates plan execution
- New principles I-VIII integrate seamlessly; plan templates can reference § I (modular architecture), § IV (directory structure), § III (dependencies)
- **Action**: Add note in template: "Verify dependencies against § III (Minimal Dependencies); confirm module placement per § IV"
- **Status**: No template changes required; guidance applies as-is

### `.specify/templates/spec-template.md` ✅ ALIGNED
- User stories template already supports independent testing (aligns with § II test-first, § VIII MVP focus)
- Edge cases and acceptance scenarios support observability requirements (§ VI)
- **Action**: Ensure spec writers reference § V (Spec-Driven) + § VI (observability requirements) for telemetry inclusion
- **Status**: No template changes required; add optional reminder: "Include telemetry schema in data model section"

### `.specify/templates/tasks-template.md` ✅ ALIGNED
- Phase structure (Shared Infrastructure → User Stories → Polish) aligns with § I (library-first), § II (test-first Red-Green-Refactor)
- Test tasks and independent story implementation match § II requirements
- **Action**: Clarify in template that Phase 1 infrastructure must establish directory structure per § IV
- **Status**: No template changes required; guidance applies implicitly

### Additional Templates to Review

**`.specify/templates/checklist-template.md`** (not yet reviewed)  
- Action: Verify compliance checklist includes § I-VIII checks
- Status: ⚠ **PENDING** – recommend adding "Constitution Compliance" section with § I-VIII checkboxes

**`.specify/templates/agent-file-template.md`** (not yet reviewed)  
- Action: Verify agent guidance doesn't conflict with new principles
- Status: ⚠ **PENDING** – recommend cross-check for § III dependency documentation requirements

---

## Runtime Guidance Documents Requiring Updates

| Document | Current Status | Recommended Action | Priority |
|----------|---|---|---|
| `README.md` | References phases 1-13; structure guidance outdated | Update "Quick Start" section to reference directory structure (§ IV) and module organization | HIGH |
| `docs/QUICKSTART.md` | Likely missing spec-first guidance | Add reference to § V (Spec-Driven); link to `docs/specs/` template location | HIGH |
| `docs/ARCHITECTURE_SPEC.md` | Phase reference table current but no modularity guidance | Add subsection: "Module Placement Rules per Constitution § IV" with backend/, python/, src/ examples | MEDIUM |
| `DEPENDENCY_MANIFEST.md` | May not exist or be incomplete | **Action Required**: Create/audit; list all external deps with § III rationale; review transitive chains | CRITICAL |
| `scripts/build_all.sh` | Build logic correct but no validation of directory structure | Add pre-build check: verify no loose module files in root (§ IV rule enforcement) | MEDIUM |
| `CMakeLists.txt` | Likely contains legacy path references | Audit for hardcoded paths; ensure VERSION reflects semantic versioning (§ VII) | MEDIUM |

---

## Compliance Checklist Additions

All future PRs must verify:

- [ ] **§ I**: New libraries placed in designated dirs (backend/{cpu\|cuda}, python/, src/); standalone header + implementation
- [ ] **§ II**: Tests written first; red-green-refactor cycle documented in commit message
- [ ] **§ III**: New dependencies documented in DEPENDENCY_MANIFEST.md with rationale
- [ ] **§ IV**: Module layout follows canonical structure; loose files eliminated or moved to scripts/
- [ ] **§ V**: Specs reference (if feature); observability integrated (§ VI)
- [ ] **§ VI**: External APIs use text protocols (JSON, CSV); no opaque binary formats
- [ ] **§ VII**: Version bumps reflect semantic versioning; breaking changes include deprecation + migration guide
- [ ] **§ VIII**: MVP focused; speculative features deferred to separate specs; no YAGNI violations

---

## Deferred TODOs

1. **Automated Linting** (Q1 2026): Enforce § IV module isolation—no cross-module includes except via core/
2. **Legacy Path Deprecation** (Q1 2026): Audit build scripts for hardcoded old paths; migrate references
3. **Unified Testing Framework** (Q2 2026): Consolidate redundant test runners under centralized `tests/` entry point
4. **DEPENDENCY_MANIFEST.md Audit** (Immediate): Comprehensive inventory of all external libraries with § III rationale

---

## Recommended Immediate Actions (Week of Nov 6)

### Priority 1 (This Week)
1. **Create/Audit DEPENDENCY_MANIFEST.md** – List all external deps (CMake, gcc, CUDA, Python packages, cirq) with § III rationale
2. **Update README.md** – Add "Project Structure" section referencing § IV and canonical layout
3. **Audit Root Directory** – Identify loose Python scripts (run_phase11.py, run_qallow.py, etc.) and plan migration to scripts/ or src/

### Priority 2 (Next Week)
1. **Update docs/QUICKSTART.md** – Add § V (Spec-Driven) workflow; link to spec template
2. **CMakeLists.txt Audit** – Verify VERSION semantic versioning format; remove hardcoded paths
3. **Add Compliance Checklist** – Prepend to `.specify/templates/checklist-template.md` with § I-VIII gates

### Priority 3 (Ongoing)
1. Monitor all PRs for Constitution compliance; use § I-VIII checklist
2. Track directory structure migration progress; report status quarterly
3. Prepare Q1 2026 automated linting enforcement roadmap

---

## Summary for Project Leads

**Qallow Constitution v3.0.0 establishes binding governance for modular, test-first development with explicit dependency and observability requirements.** This upgrade from advisory guidelines (v2.2) to enforceable principles affects:

- **All new features** must start as independently testable libraries following § IV canonical structure
- **All code changes** require passing test gates (§ II) and dependency reviews (§ III)
- **All external APIs** must expose text protocols (JSON, CSV, CLI); no opaque binary formats (§ VI)
- **All library contracts** must follow semantic versioning (§ VII) with migration guides for breaking changes

**No code review exceptions** without written rationale linked to Constitution sections. This ensures reproducible, maintainable, observable evolution of the Qallow platform.

---

**Sync Report Status**: ✅ COMPLETE  
**Constitution Ready for Adoption**: YES  
**Commit Message Template**:
```
docs: amend constitution to v3.0.0 (modular governance + observability)

- Introduce binding § IV (modular directory structure) replacing advisory guidance
- Elevate § II (test-first) to non-negotiable gating criterion
- Add § III (minimal dependencies), § VI (text I/O observability), § VII (versioning), § VIII (YAGNI)
- Establish immediate roadmap for directory cleanup + Q1/Q2 2026 tooling enforcement
- All PRs now require § I-VIII compliance checklist verification

Breaking: Directory structure now canonical; loose module files must migrate to backend/{cpu|cuda}, python/, src/, or scripts/
```
