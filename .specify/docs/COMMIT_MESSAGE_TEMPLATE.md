# Suggested Commit Message

```git
docs: amend constitution to v3.0.0 (modular governance + observability)

## Summary
Major version bump establishes binding governance for modular, test-first 
development with explicit observability and dependency requirements.

## Changes

### Core Governance (MAJOR - backward-incompatible)
- **§ I (Library-First)**: Bind canonical directory structure
  * backend/{cpu|cuda}/{module}/ for phase implementations
  * python/{module}/ for Python bridges
  * src/{service}/ for runtime/telemetry/CLI
  * No organizational-only libraries; clear concrete purpose required
  
- **§ II (Test-First)**: Elevate to NON-NEGOTIABLE binding gate
  * Red-Green-Refactor cycle mandatory
  * 100% test pass rate or written exception required in commit message
  * Unit tests: tests/{module}/, Integration: tests/integration/
  * CI gates: ctest (C/CUDA) + pytest (Python) with 0 failures
  
- **§ III (Minimal Dependencies)** [NEW]
  * Baseline: CMake, gcc ≥11, Python ≥3.10 only
  * Every external dependency documented in DEPENDENCY_MANIFEST.md with rationale
  * Avoid deep transitive chains; use feature flags for optional deps
  * Python versions locked in requirements-{base,dev,gpu,web}.txt
  
- **§ IV (Modular Structure)** [NEW - CANONICAL]
  * Bind project layout: backend/, core/include/, python/, src/, tests/, docs/specs/, scripts/, deploy/
  * Each module owns its tests (tests/{module_name}/)
  * No loose files in root (except CMakeLists.txt, package configs, LICENSE, README, bootstrap)
  * Move loose scripts (run_*.py) to scripts/ (development) or src/ (runtime)

### Extended Principles
- **§ V (Spec-Driven)**: Refined with observability integration
  * Specs → docs/specs/{number}-{name}/spec.md + plan.md
  * Telemetry: QALLOW_PROFILE_SCOPE macros, qallow_log_* functions
  * All metrics → src/runtime/telemetry_outputs.c → data/logs/ (CSV, JSON)
  
- **§ VI (Text I/O Observability)** [NEW]
  * ALL external interfaces MUST use text protocols (JSON, CSV, CLI args/stdout)
  * ZERO opaque binary formats for external APIs
  * Every execution produces logs in data/logs/
  * Structured logging with context tags (module, phase, iteration)
  
- **§ VII (Versioning)** [NEW]
  * MAJOR.MINOR.PATCH semantic versioning
  * Reflected in: CMakeLists.txt, pyproject.toml, phase bridge contracts
  * Breaking changes: deprecation (1 minor), migration guide, changelog
  
- **§ VIII (Simplicity & YAGNI)** [NEW]
  * MVP first: deliver core value with minimal scaffolding
  * No pre-optimization; measure before tuning
  * Defer speculative features to separate specs
  * Each phase: clear entry point, deterministic logic, isolated state

### Documentation & Compliance
- Establish PR compliance checklist for all future changes (§ I-VIII verification)
- Create DEPENDENCY_MANIFEST.md (immediate action - this week)
- Update README.md with Project Structure section (reference § IV)
- Update CMakeLists.txt semantic versioning (§ VII compliance)
- Plan Q1 2026 automated linting for § IV enforcement

## Impact
- **Modular Structure**: Now canonical; existing code migrates on next touch
- **Test-First**: Binding gate; no exceptions without written rationale
- **Dependencies**: Explicit coupling required; DEPENDENCY_MANIFEST.md mandatory
- **Observability**: Text I/O binding; zero opaque binary external formats
- **Versioning**: Semantic versioning enforced; breaking changes require migration guide

## Breaking Changes
- Directory structure is canonical; loose module code must migrate to backend/{cpu|cuda}/, python/, src/, or scripts/
- Tests required before code ships; no exceptions without documentation
- All external dependencies must be justified in DEPENDENCY_MANIFEST.md
- Opaque binary external APIs must be converted to JSON/CSV/CLI text protocols

## Files Modified
- `.specify/memory/constitution.md` – Updated to v3.0.0 with 8 binding principles
- `.specify/CONSTITUTION_SYNC_REPORT.md` – Generated impact analysis + roadmap

## Roadmap
- **Immediate** (this week): Create DEPENDENCY_MANIFEST.md, audit root files, update README
- **Q1 2026**: Automated linting enforces § IV module isolation
- **Q2 2026**: Unified test framework consolidation

## Verification
- [x] Constitution template filled with Qallow-specific values
- [x] All principles § I-VIII defined with binding enforcement rules
- [x] Directory structure canonicalized with migration plan
- [x] Template alignment verified (plan/spec/tasks templates ✅)
- [x] Governance checklist defined
- [x] Roadmap clarified (immediate vs Q1/Q2 2026)

## Related Issues
Closes: [ISSUE_NUMBER] (if applicable)

---

**Constitution Version**: 3.0.0  
**Ratified**: 2025-06-13 (original)  
**Last Amended**: 2025-11-06 (this commit)
```

## How to Use This Commit

1. **Copy the commit template** above into your local git message editor
2. **Update** `[ISSUE_NUMBER]` if this closes a GitHub issue
3. **Review** the Breaking Changes section with your team
4. **Commit** with: `git commit -F commit_message.txt`
5. **Push** and open PR with compliance checklist in PR description

## PR Description Template

Once committed, use this for your PR:

```markdown
## Constitution Amendment: v2.2 → v3.0.0

This PR updates the Qallow Project Constitution to establish binding governance for modular, test-first development with explicit observability and dependency requirements.

### Why This Change?
- **Modularity**: Project structure was descriptive; now canonical with binding rules
- **Test-First**: Elevated from guidance to non-negotiable gating criterion
- **Dependencies**: Explicit coupling policy and DEPENDENCY_MANIFEST.md requirement
- **Observability**: Text I/O binding; zero opaque binary formats for external APIs
- **Versioning**: Semantic versioning with migration guides for breaking changes

### What's New (§ I-VIII)
- ✅ § I (Library-First): Canonical directory structure binding
- ✅ § II (Test-First): NON-NEGOTIABLE gate; 100% pass rate
- ✅ § III (Minimal Deps): NEW - explicit coupling + DEPENDENCY_MANIFEST.md
- ✅ § IV (Modular Structure): NEW - canonical layout with binding rules
- ✅ § V (Spec-Driven): Refined with observability
- ✅ § VI (Text I/O): NEW - zero opaque binary formats
- ✅ § VII (Versioning): NEW - MAJOR.MINOR.PATCH enforcement
- ✅ § VIII (Simplicity): NEW - YAGNI principle

### Immediate Actions (Assigned)
- [ ] Create DEPENDENCY_MANIFEST.md (due Nov 8)
- [ ] Audit root loose files (due Nov 8)
- [ ] Update README.md § IV (due Nov 10)
- [ ] Update CMakeLists.txt versioning (due Nov 10)
- [ ] Add PR compliance checklist (due Nov 12)

### Files Modified
- `.specify/memory/constitution.md` (v3.0.0 - 191 lines)
- `.specify/CONSTITUTION_SYNC_REPORT.md` (NEW - impact analysis)
- `.specify/CONSTITUTION_UPDATE_SUMMARY.md` (NEW - implementation guide)
- `.specify/CONSTITUTION_QUICK_REFERENCE.md` (NEW - cheat sheet)

### Breaking Changes
- ✓ Directory structure now canonical; existing code migrates on touch
- ✓ Tests required before ship; exceptions require rationale
- ✓ All deps must be justified
- ✓ External APIs must use text protocols (JSON/CSV/CLI)

### Review Checklist
- [ ] Constitution principles § I-VIII clear and enforceable?
- [ ] Directory structure migration plan feasible for existing code?
- [ ] Templates (plan/spec/tasks) remain aligned?
- [ ] DEPENDENCY_MANIFEST.md creation tracked?
- [ ] Q1/Q2 2026 roadmap realistic?

### Related Documentation
- Constitution: `.specify/memory/constitution.md`
- Sync Report: `.specify/CONSTITUTION_SYNC_REPORT.md`
- Summary: `.specify/CONSTITUTION_UPDATE_SUMMARY.md`
- Quick Ref: `.specify/CONSTITUTION_QUICK_REFERENCE.md`

### Questions?
See `.specify/CONSTITUTION_QUICK_REFERENCE.md` FAQ section.
```

---

**Ready to commit!** Copy the commit message and PR template above.
