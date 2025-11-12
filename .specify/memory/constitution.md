# Qallow Project Constitution

## Core Principles

### I. Library-First Modular Architecture
Every feature starts as a standalone, independently testable library with clear purpose. Each library:
- **Must be self-contained** with explicit dependencies declared in CMakeLists.txt or package.json
- **Must be independently testable** before integration
- **Must expose a clean contract** via header files (.h), interfaces, or exported symbols
- **Must reside in a dedicated directory** following the directory structure convention:
  - C/CUDA libraries: `backend/{cpu|cuda}/{module_name}/` with `{module_name}.h` + `{module_name}.c`
  - Python libraries: `python/{module_name}/` with `__init__.py` + exports
  - CLI/Interface: `interface/` or `src/cli/`
- **No organizational-only libraries**—every library must provide concrete value or be merged into a parent module

### II. Test-First Development (NON-NEGOTIABLE)
Test-Driven Development is mandatory following the Red-Green-Refactor cycle:
1. **Red**: Write test cases based on spec requirements; tests MUST fail initially
2. **Green**: Implement minimal code to pass tests
3. **Refactor**: Optimize and improve without breaking tests

- All unit tests reside in `tests/` with mirrored structure from source code
- Integration tests grouped in `tests/integration/` by feature/contract
- CI gates enforce: 0 test failures + 100% pass rate across all configurations
- For C/CUDA: `ctest --test-dir build` must succeed; for Python: `pytest tests/` must succeed
- CRITICAL: No code ships without passing tests; exceptions require written rationale in commit message

### III. Minimal Dependencies & Explicit Coupling
- **Baseline only**: CMake, C compiler (gcc ≥11 or clang ≥15), Python ≥3.10
- **Optional for GPU**: CUDA 12.0+, cirq (Python quantum bridge)
- **Avoid deep dependency chains**: Use vendored code or feature flags for optional dependencies
- **Document every external dependency** with rationale in `DEPENDENCY_MANIFEST.md`
- **Prefer stable, narrow APIs**: No monolithic frameworks without justification
- Module imports/includes must be explicitly listed; no implicit transitive dependencies
- For Python: lock versions in `requirements-*.txt` (base, dev, gpu, web variants)

### IV. Modular Directory Structure
Project organization enforces loose coupling and clear module boundaries:

```
Qallow/
├── backend/
│   ├── cpu/             # CPU-optimized phase implementations
│   │   ├── ethics/      # Phase 8-10 ethics computation
│   │   ├── quantum/     # Phase 11 quantum bridge simulation
│   │   ├── elasticity/  # Phase 12 elasticity mechanics
│   │   ├── harmonics/   # Phase 13 resonance harmonics
│   │   └── lattice/     # Phase 14-15 lattice convergence
│   └── cuda/            # CUDA mirrors of above phases
├── core/                # Shared contracts, types, macros
│   └── include/         # Central header repository
├── include/qallow/      # Public API headers
├── interface/           # Orchestration (launcher.c, main.c) + SDL UI
├── python/              # Python quantum bridge + utilities
│   ├── quantum/
│   └── [other modules]
├── src/                 # CLI, runtime, distributed, ethics, telemetry
│   ├── cli/
│   ├── runtime/
│   ├── telemetry/
│   └── [other services]
├── tests/               # Unit & integration test suites
├── docs/specs/          # Specification-driven design artifacts
├── scripts/             # Build & utility scripts (not deployment artifacts)
└── deploy/              # Kubernetes manifests, Docker configs (separate from scripts/)
```

**Key rules**:
- One logical module = one directory under a parent (backend, src, python)
- Each module owns its tests under `tests/{module_name}/`
- Shared utilities in `core/include/` or `include/qallow/` only
- No loose files in root except: CMakeLists.txt, package configs, license, README, bootstrap
- Move loose Python scripts (e.g., `run_phase11.py`, `run_qallow.py`) → `scripts/` (for development) or `src/` (for runtime)
- Build outputs: `build/` (generated, transient); deployable artifacts only in `deploy/`

### V. Spec-Driven Development with Observability
- Every major feature begins with a specification in `docs/specs/{number}-{name}/spec.md`
- Specifications define user stories, acceptance criteria, and edge cases (NOT implementation)
- Technical planning documented in `docs/specs/{number}-{name}/plan.md` with architecture choices
- Telemetry on all hot paths: `QALLOW_PROFILE_SCOPE` macros in C, `qallow_log_*` in Python
- All metrics funnel through centralized telemetry in `src/runtime/telemetry_outputs.c` → `data/logs/`
- Structured output (CSV, JSON) enables reproducible analysis and dashboard integration
- Phase execution state tracked and persisted (status file sync to Windows shares where applicable)

### VI. Observability & Testability Through Text I/O
- **All external interfaces use text protocols** (JSON over HTTP, CSV for metrics, CLI args/stdin/stdout)
- **Debugging guarantee**: Every execution produces human-readable logs in `data/logs/`
- **Zero opaque binary formats** for external APIs; internal optimizations acceptable with documented contracts
- **Structured logging**: Use `qallow_log_info`, `qallow_log_error` with context tags (module, phase, iteration)
- **Metrics exporters**: CSV and JSON summarization of telemetry for dashboards and reproducibility

### VII. Versioning & Breaking Changes
- Follow **MAJOR.MINOR.PATCH** semantic versioning
  - **MAJOR**: Library API or phase contract changes, removal of phases, fundamental architecture shifts
  - **MINOR**: New phases added, new observability features, non-breaking protocol extensions
  - **PATCH**: Bug fixes, performance improvements, documentation, internal refactors
- Version bumps reflected in: `CMakeLists.txt` (project VERSION), Python `pyproject.toml`, and phase bridge contracts
- Breaking changes require: deprecation notice (1 minor release), migration guide, and full changelog entry

### VIII. Simplicity & YAGNI Principle
- Start simple: MVP must deliver core value with minimal scaffolding
- No pre-optimization; measure before tuning
- Avoid boilerplate frameworks; use language-native constructs
- Defer features marked TODO/FIXME into separate specs; do not ship speculative code
- Each phase implementation: clear entry point, deterministic logic, isolated state

## Code Quality Standards

### C/CUDA (`backend/{cpu|cuda}/`)
- Struct naming: `{module}_state_t` for state, `{module}_config_t` for config
- Phase files: `phase_NN_name.c` (one phase per file for clarity)
- Defensive: bounds checking on arrays, null pointer guards
- Telemetry: `QALLOW_PROFILE_SCOPE` around loops/hot functions
- Shared types: `core/include/qallow_types.h`; internal types in module headers

### Python (`python/`, `src/`)
- Type hints required for all function signatures (use typing module, torch/numpy annotations)
- Docstrings: module-level, class-level, method-level with Examples section
- Error handling: raise descriptive exceptions, never silent failures
- Logging: use `qallow_log_*` or structured logging framework; no bare `print` outside CLI

### Build System
- **CMake as primary**: `CMakeLists.txt` defines all targets, tests, dependencies
- **Scripts for convenience**: `scripts/build_all.sh [--cpu|--cuda]` orchestrates full builds
- **No hardcoded paths**: Use CMake variables (CMAKE_BINARY_DIR, PROJECT_SOURCE_DIR)
- **Parallel by default**: `cmake --build build --parallel` in scripts and CI

## Testing Requirements & Gates

All code changes must pass:

1. **Syntax validation**: No compiler warnings (or documented exceptions)
2. **Unit tests**: `ctest --test-dir build` (C/CUDA) or `pytest tests/` (Python); 100% pass rate
3. **Integration tests**: Multi-scenario framework covering known configuration combinations
4. **Performance benchmarks**: Run `tests/sequential_phase_benchmark.sh` for phase comparisons
5. **Documentation**: Spec linked in commit, README updated if CLI/config changed

**Success Criteria**:
- All test scenarios pass with 100% success rate
- Zero crashes or undefined behavior
- Performance within 5% of previous baseline (or new baseline justified in PR)
- Coherence maintained (if applicable to phase)
- New telemetry fields documented in schema

## Directory Reorganization Roadmap

**Immediate** (this update):
- Establish above structure as canonical
- Move loose Python scripts from root → `scripts/` or `src/` as appropriate
- Consolidate utility docs (standalone .md files) into `docs/specs/` or `docs/guides/`

**Q1 2026**:
- Automated linting to enforce module isolation (no cross-module includes except through core/)
- Deprecate legacy paths in build scripts; all references use new structure

**Q2 2026**:
- Consolidate redundant testing frameworks under `tests/` with unified entry point

## Governance

### Amendment Process
1. Changes to principles require written rationale in PR description
2. Constitution updates trigger automatic template consistency checks
3. Version bump documented in git commit message with affected sections
4. All PRs must reference compliance checks: "Constitution § I-VIII verified"

### Compliance Checklist
- [ ] No new dependencies without Governance review
- [ ] All modules reside in designated directories per structure above
- [ ] Tests written before/alongside implementation
- [ ] Public APIs documented and versioned
- [ ] Telemetry integrated for observability
- [ ] CI gates enforce test passage and no new compiler warnings

### Guiding Documents
- **Development Workflow**: See `docs/QUICKSTART.md`, `.specify/templates/spec-template.md`
- **Phase Architecture**: `docs/ARCHITECTURE_SPEC.md`
- **Dependency Policy**: `DEPENDENCY_MANIFEST.md`
- **Runtime Telemetry**: `src/runtime/telemetry_outputs.c` and schema documentation
- **MCP Memory Service**: Available for context persistence via VS Code; configure in `.vscode/mcp.json`

---

**Version**: 3.0.0 | **Ratified**: 2025-06-13 | **Last Amended**: 2025-11-06

**Amendment Summary**:
- **MAJOR version bump** (2.2 → 3.0.0): Restructured principles around modular architecture enforcement, modular directory structure now canonical, test-first made primary gate (not advisory), explicit dependencies policy introduced, YAGNI principle elevated.
- **New principles**: VI (Observability & Text I/O), VII (Versioning), VIII (Simplicity) added.
- **Structure formalized**: `backend/{cpu|cuda}/`, `core/include/`, `python/`, `src/`, `tests/`, `docs/specs/`, `scripts/`, `deploy/` now binding canonical layout.
- **Roadmap clarified**: Immediate implementation vs. Q1/Q2 2026 milestones for tooling enforcement.
