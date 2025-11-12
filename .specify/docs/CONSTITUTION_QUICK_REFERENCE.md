# 🎯 Qallow Constitution v3.0.0 - Quick Reference

## Updated: November 6, 2025
**Version Bump**: 2.2 → 3.0.0 (MAJOR - backward-incompatible governance shift)

---

## 8 Core Binding Principles at a Glance

| § | Principle | Key Rule | Binding Since |
|---|-----------|----------|---|
| **I** | Library-First Modular | Place code in `backend/{cpu\|cuda}/`, `python/`, `src/`, `interface/` | v3.0.0 |
| **II** | Test-First (NON-NEGOTIABLE) | Red-Green-Refactor mandatory; 100% pass rate or written exception | v3.0.0 |
| **III** | Minimal Dependencies | Document every external dep in DEPENDENCY_MANIFEST.md | **NEW** v3.0.0 |
| **IV** | Modular Structure | Canonical layout (backend/, python/, src/, tests/, docs/specs/) | **NEW** v3.0.0 |
| **V** | Spec-Driven | Every major feature: docs/specs/{number}-{name}/spec.md + plan.md | v2.2+ refined |
| **VI** | Text I/O Observability | All external APIs: JSON/CSV/CLI; ZERO opaque binary formats | **NEW** v3.0.0 |
| **VII** | Versioning | MAJOR.MINOR.PATCH with migration guides for breaking changes | **NEW** v3.0.0 |
| **VIII** | Simplicity & YAGNI | MVP first; defer speculative features to separate specs | **NEW** v3.0.0 |

---

## Canonical Directory Structure (§ IV - BINDING)

```
backend/{cpu,cuda}/{phase_module}/    → Phase implementations (ethics, quantum, elasticity, etc.)
core/include/                          → Shared types & macros (qallow_types.h, contracts)
include/qallow/                        → Public API headers
interface/                             → Orchestration (launcher.c, main.c) + SDL UI
python/{module}/                       → Python quantum bridge, utilities
src/{cli,runtime,telemetry}/           → Services, CLI, telemetry
tests/{module_name}/                   → Unit/integration tests mirroring source
docs/specs/{number}-{name}/            → Spec-driven artifacts (spec.md, plan.md, research.md)
scripts/                               → Build utilities, dev scripts (NOT deployment)
deploy/                                → Kubernetes, Docker manifests (separate from scripts/)
```

**ROOT RULE**: No loose module files in root except: CMakeLists.txt, README.md, LICENSE, bootstrap.sh, package.json, Cargo.toml

---

## PR Compliance Checklist (Required for All PRs)

```markdown
## Constitution Compliance § I-VIII ✅

- [ ] **§ I**: Code placed in `backend/{cpu|cuda}/`, `python/`, `src/`, `interface/`, or `scripts/`
- [ ] **§ II**: Tests written FIRST; red-green-refactor cycle; `ctest` or `pytest` passes 100%
- [ ] **§ III**: New dependencies in `DEPENDENCY_MANIFEST.md` with rationale + transitive analysis
- [ ] **§ IV**: No loose files in root; CMakeLists.txt reflects canonical structure
- [ ] **§ V**: Spec linked (if feature); `docs/specs/{number}-{name}/` exists with spec.md + plan.md
- [ ] **§ VI**: External APIs use JSON, CSV, or CLI; verified ZERO opaque binary formats
- [ ] **§ VII**: Version bumps follow MAJOR.MINOR.PATCH; breaking changes include deprecation + migration
- [ ] **§ VIII**: MVP-focused; speculative code → separate spec; no YAGNI violations

**Constitution Version**: v3.0.0 | **Verified**: [Date/Author]
```

---

## Critical Immediate Tasks (This Week)

| Task | Owner | Deadline | File(s) |
|------|-------|----------|---------|
| **Create DEPENDENCY_MANIFEST.md** | Team Lead | Nov 8 | `DEPENDENCY_MANIFEST.md` (new) |
| **Audit root loose files** | Arch Team | Nov 8 | Identify: `run_*.py`, etc. |
| **Update README.md § IV ref** | Docs | Nov 10 | `README.md` → add structure section |
| **CMakeLists.txt audit** | Build Team | Nov 10 | `CMakeLists.txt` → verify versioning |
| **Add compliance checklist** | QA | Nov 12 | `.specify/templates/checklist-template.md` or new file |

---

## What Changed for Developers

### ❌ NOW FORBIDDEN
- ✗ Loose Python scripts in root → Move to `scripts/` or `src/`
- ✗ Code without tests → Test-first or written exception required
- ✗ Undocumented dependencies → DEPENDENCY_MANIFEST.md required
- ✗ Opaque binary external APIs → Text protocols (JSON, CSV, CLI) required
- ✗ Breaking changes without migration guide → Deprecation + migration path required
- ✗ Speculative features → Defer to separate specs in `docs/specs/`

### ✅ NOW REQUIRED
- ✓ Modular library placement (§ I, § IV)
- ✓ Red-Green-Refactor test cycle (§ II)
- ✓ Dependency rationale in DEPENDENCY_MANIFEST.md (§ III)
- ✓ Text I/O for all external APIs (§ VI)
- ✓ Semantic versioning (§ VII)
- ✓ Spec references in `docs/specs/` (§ V)

---

## Reference Files (Read These First)

1. **Constitution**: `.specify/memory/constitution.md` (191 lines, ~8KB)
2. **Sync Report**: `.specify/CONSTITUTION_SYNC_REPORT.md` (detailed impact analysis)
3. **This Summary**: `.specify/CONSTITUTION_UPDATE_SUMMARY.md` (full implementation guide)
4. **Phase Architecture**: `docs/ARCHITECTURE_SPEC.md` (reference existing phases)
5. **Build Guidance**: `README.md` + `docs/QUICKSTART.md`

---

## FAQ

### Q: Do I need to rewrite existing code?
**A**: No, but new code MUST follow v3.0.0. Existing code migrates on next touch (refactor, bugfix). Schedule migration of loose files over Q1 2026.

### Q: What if I need to deviate from § I-IV structure?
**A**: Document exception in PR description with Constitution section reference. Example: "§ IV exception: Device-specific kernel requires build/cuda_ops/ due to [reason]. Migration plan: [timeline]."

### Q: Tests are too slow—can I skip some?
**A**: § II allows pragmatic testing; document trade-offs in commit message. Minimum: contract tests for new API. Performance tests on request in code review.

### Q: What's in DEPENDENCY_MANIFEST.md?
**A**: Example:
```markdown
## External Dependencies (Constitution § III)

- **CMake ≥3.20**: Build orchestration (standard, no alternative)
- **gcc ≥11 | clang ≥15**: C compilation (required for performance)
- **CUDA 12.0+**: GPU acceleration (OPTIONAL; feature flag; CPU fallback available)
- **cirq**: Quantum circuit execution (OPTIONAL; only loaded with QALLOW_cirq=1)
- **Python 3.10+**: Runtime (requirement justified by phase modules)

Transitive Analysis:
- cirq → depends on: cirq-aer, numpy, scipy
  * Depth: 2 | Justification: Must accept transitive for quantum simulation
  * Risk: Lock versions in requirements-gpu.txt to prevent surprise breaks
```

### Q: When does the roadmap phase start (Q1 2026)?
**A**: After directory structure stabilizes and DEPENDENCY_MANIFEST.md is complete. Automated linting will then enforce § IV module isolation.

---

## Key Contacts & Escalation

- **Constitution Questions**: Review `.specify/memory/constitution.md` § [number]
- **Directory Structure**: Reference Constitution § IV + `docs/ARCHITECTURE_SPEC.md`
- **Compliance Checklist**: See "PR Compliance Checklist" section above
- **Roadmap Milestones**: Check `.specify/CONSTITUTION_SYNC_REPORT.md` → "Roadmap" section

---

## Version History

| Version | Date | Change | Status |
|---------|------|--------|--------|
| 3.0.0 | 2025-11-06 | Modular governance + observability binding; 8 principles | **ACTIVE** |
| 2.2 | 2025-11-04 | Phase architecture + workflow guidance (advisory) | Superseded |
| 2.1 | 2025-07-16 | Spec-first workflow foundation | Legacy |

---

**Print This Checklist. Post on Sprint Board. Reference in Every PR.**

Constitution v3.0.0 is **binding effective immediately**. All new code must comply with § I-VIII.
