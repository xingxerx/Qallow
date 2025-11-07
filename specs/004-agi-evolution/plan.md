# Feature 004: AGI Evolution Planning & Implementation Roadmap

**Feature ID**: 004-agi-evolution  
**Title**: Artificial General Intelligence Evolution (Phase 1: Meta-Learning)  
**Version**: 1.0.0  
**Date**: 2025-11-07  

---

## 1. Implementation Strategy

This plan decomposes Feature 004 into **8 actionable implementation tasks**, sequenced to deliver Phase 1 (Meta-Learning) with full Constitution compliance and quantum backend support.

### Execution Phases

| Phase | Duration | Deliverables | Validation |
|-------|----------|--------------|-----------|
| **Phase A: Foundation** | 2-3 hours | Cognitive state structure, build system updates | Unit tests pass, compilation succeeds |
| **Phase B: Core Engine** | 3-4 hours | Bayesian optimization framework, acquisition functions | Convergence benchmarks, telemetry generation |
| **Phase C: Quantum Integration** | 2-3 hours | CUDA-Q/Cirq backends, auto-detection | Multi-backend tests pass, speedup verified |
| **Phase D: Integration & Polish** | 2-3 hours | CLI integration, documentation, compliance audit | Full workflow test, Constitution pass |

**Total Estimated Effort**: 9-13 hours (can parallelize Phases A & B foundations)

---

## 2. Risk Analysis & Mitigation

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|-----------|
| R1: Quantum backend unavailable | High | Low | Implement robust CPU fallback; test without QISKIT |
| R2: Meta-learning diverges on pathological functions | Medium | Medium | Implement convergence detection, iteration limits (max 1000) |
| R3: Memory bloat with large parameter spaces | Medium | Medium | Profile memory usage; implement sparse tensor support in Phase 2 |
| R4: Integration conflicts with existing phases | High | Low | Keep Phase 1 isolated in `src/mind/`, no modifications to Phase 12-15 |
| R5: Ethics audit reveals conflicts | High | Low | Design ethics constraints into Bayesian objective from start (R5 → Task 1) |

### Risk Owners
- **R1, R3, R5**: @Agent (technical architecture)
- **R2, R4**: @User (acceptance/validation)

---

## 3. Task Breakdown (8 Core Tasks)

### TASK 1: Cognitive State & Ethics Foundation [PRE-REQ: None]

**Objective**: Establish unified cognitive state structure integrating self-model, ethics, and goals.

**Work Items**:
1. Create `core/include/cognitive.h` with `cognitive_state_t` struct:
   - `ethics_score` (E = S + C + H)
   - `self_model` (learner's internal representation)
   - `goal_vector` (task objectives)
   - `learning_rate` (adaptive parameter)
   - `backend` (active execution engine)

2. Implement `src/constitution.c` initialization:
   - `cognitive_state_init()` function
   - Load ethics scoring from existing `algorithms/ethics_*.c`
   - JSON serialization: `cognitive_state_to_json()`
   - Validation: `cognitive_state_validate()` (ethics ∈ [0,1], checks Constitution)

3. Create unit tests in `tests/test_cognitive_state.c`:
   - Test initialization with default values
   - Test ethics scoring integration
   - Test JSON round-trip serialization
   - Test Constitution § compliance

**Deliverables**:
- ✅ `core/include/cognitive.h` (50+ lines)
- ✅ `src/constitution.c` function stubs + initialization (100+ lines)
- ✅ `tests/test_cognitive_state.c` (60+ lines, 4+ test cases)
- ✅ Integration into CMakeLists.txt build targets

**Acceptance Criteria**:
- All 4 unit tests pass
- JSON serialization produces valid JSON readable by Python
- Constitution § audit recognizes cognitive state as § IV compliant
- Build system compiles without errors

**Dependencies**: None (standalone)  
**Blocks**: All subsequent meta-learning tasks  
**Estimated Effort**: 2 hours

---

### TASK 2: Bayesian Optimization Core Engine [PRE-REQ: Task 1]

**Objective**: Implement classical Bayesian optimization framework as foundation for meta-learning.

**Work Items**:
1. Create `src/mind/quantum_learn.c` with core structures:
   - `bayesian_optimizer_t` struct (GP surrogate, acquisition function)
   - Gaussian Process implementation (mean, covariance matrix)
   - Expected Improvement (EI) acquisition function

2. Implement main optimization loop:
   - `bayesian_optimize()` function (takes loss function, parameter bounds)
   - Iterative sampling: select next point via EI maximization
   - Evaluate loss, update GP model, iterate
   - Return best parameters found + convergence history

3. Support parameter representation:
   - Dense vector representation (for 1-100 parameters)
   - Bounds checking (parameter constraints)
   - Normalization to [-1, 1] for numerical stability

4. Telemetry logging:
   - Log iteration count, current best loss, GP uncertainty
   - Export to CSV: `data/logs/metalearn_convergence.csv`
   - Track backend used, execution time

**Deliverables**:
- ✅ `src/mind/quantum_learn.c` (250+ lines)
- ✅ `core/include/quantum_learn.h` (100+ lines, API definitions)
- ✅ `src/mind/gp_surrogate.c` (150+ lines, Gaussian Process implementation)
- ✅ Unit tests: `tests/test_bayesian_opt.c` (100+ lines, 5+ test cases)

**Acceptance Criteria**:
- Convergence test: minimize `f(x) = (x-3)^2` → finds x≈3 within 30 iterations
- Convergence test: Rastrigin function → finds minimum within iteration budget
- Telemetry CSV contains correct column headers and data types
- No memory leaks (valgrind/sanitizer check)
- Execution time <500ms for 100 iterations on single core

**Dependencies**: Task 1 (cognitive state)  
**Blocks**: Tasks 3, 5, 6  
**Estimated Effort**: 3-4 hours

---

### TASK 3: Quantum-Enhanced Sampling (CUDA-Q Backend) [PRE-REQ: Task 2]

**Objective**: Integrate CUDA-Q for hybrid quantum sampling (optional, with CPU fallback).

**Work Items**:
1. Detect CUDA-Q availability at compile-time and runtime:
   - Check for `libcudaqrt.so` in system paths
   - Set CMake flag `QALLOW_ENABLE_CUDAQ` (auto-detect)
   - Provide CLI override: `--ml-backend cuda-q`

2. Implement `src/mind/quantum_sampling.c`:
   - `quantum_sample_parameters()` function
   - Use CUDA-Q to create superposition of candidate parameters
   - Measure to get probabilistically biased samples
   - Fallback to classical Sobol sequence if CUDA-Q unavailable

3. Importance weighting:
   - Adjust acquisition function to account for quantum sampling probabilities
   - Maintain unbiased optimizer despite quantum measurement collapse

4. Integration with Bayesian optimizer:
   - Call `quantum_sample_parameters()` in acquisition function maximization
   - Log quantum samples used, rejection rate, speedup factor

**Deliverables**:
- ✅ `src/mind/quantum_sampling.c` (150+ lines)
- ✅ `core/include/quantum_sampling.h` (50+ lines)
- ✅ CMakeLists.txt updates (CUDA-Q detection + conditional compilation)
- ✅ Unit tests: `tests/test_quantum_sampling.c` (80+ lines, skip if CUDA-Q unavailable)

**Acceptance Criteria**:
- CPU fallback works 100% of the time (even without CUDA-Q)
- CUDA-Q backend when available produces ≥2x speedup on convergence
- Quantum sampling produces unbiased parameter suggestions
- Telemetry correctly identifies quantum samples vs classical samples
- Snyk security: No buffer overflows, proper quantum result handling

**Dependencies**: Task 2 (Bayesian core)  
**Blocks**: Task 6  
**Estimated Effort**: 2-3 hours

---

### TASK 4: Cirq Python Backend Integration [PRE-REQ: Task 2]

**Objective**: Support Cirq as alternative quantum sampling backend (Python subprocess model).

**Work Items**:
1. Create `python/quantum/metalearn_quantum_cirq.py`:
   - Accept parameters via stdin (JSON)
   - Use Cirq to create parameterized circuits
   - Execute quantum sampling on simulator/hardware
   - Return results via stdout (JSON)

2. Integrate with C layer (`src/mind/quantum_learn.c`):
   - Spawn `python quantum/metalearn_quantum_cirq.py` as subprocess
   - Pass parameter bounds, iteration number as arguments
   - Parse JSON output, incorporate into acquisition function

3. Auto-detection at runtime:
   - Test for Cirq availability via `python -c "import cirq"`
   - Set backend to Cirq if CUDA-Q unavailable but Cirq present

4. Error handling:
   - Graceful degradation to CPU sampling if subprocess fails
   - Log all Cirq executions and timing

**Deliverables**:
- ✅ `python/quantum/metalearn_quantum_cirq.py` (100+ lines)
- ✅ C interface in `src/mind/quantum_sampling.c` (subprocess spawning, 50+ lines)
- ✅ CMakeLists.txt Python subprocess integration
- ✅ Tests: `tests/test_cirq_backend.c` (60+ lines, optional if Cirq unavailable)

**Acceptance Criteria**:
- Cirq backend detects available systems automatically
- Subprocess communication works without deadlocks
- Parameter passing/return via JSON is correct
- Fallback to CPU if Cirq subprocess fails
- Execution overhead <50ms per call

**Dependencies**: Task 2 (Bayesian core)  
**Blocks**: Task 6  
**Estimated Effort**: 2-3 hours

---

### TASK 5: Multi-Backend Orchestration & CLI [PRE-REQ: Task 2, Task 3, Task 4]

**Objective**: Implement runtime backend selection and CLI integration.

**Work Items**:
1. Create `src/mind/metalearn_executor.c`:
   - `metalearn_execute()` function (wraps Bayesian optimizer)
   - Auto-detect available backends (CUDA, CUDA-Q, Cirq, CPU)
   - Precedence: CUDA-Q > CUDA > Cirq > CPU
   - User override via `--ml-backend` flag

2. Extend CLI in `interface/launcher.c`:
   - Add command: `./qallow run meta-learning --iterations=N --backend=auto --objective=sphere|rastrigin|user_func`
   - Add flags: `--ml-backend [auto|cuda|cuda-q|cirq|cpu]`, `--ml-max-iter N`, `--ml-learning-rate X`

3. CLI argument parsing:
   - Parse objective function name (built-in or external)
   - Pass parameters to Bayesian optimizer
   - Export telemetry to `data/logs/metalearn_*.csv`

4. Help documentation:
   - Update `./qallow --help` to list meta-learning commands
   - Add examples in help text

**Deliverables**:
- ✅ `src/mind/metalearn_executor.c` (100+ lines)
- ✅ CLI integration in `interface/launcher.c` (50+ lines)
- ✅ Argument parser updates (20+ lines)
- ✅ CMakeLists.txt targets for metalearn CLI

**Acceptance Criteria**:
- `./qallow run meta-learning --backend=auto` selects best available backend
- `./qallow run meta-learning --ml-backend=cpu --iterations=50` forces CPU
- `./qallow --help` displays meta-learning command docs
- All backend selections log to telemetry

**Dependencies**: Tasks 2, 3, 4 (all backends)  
**Blocks**: Task 7  
**Estimated Effort**: 2 hours

---

### TASK 6: Build System Integration & CUDA Targets [PRE-REQ: Tasks 2, 3, 4]

**Objective**: Integrate meta-learning into CMake build with optional CUDA/CUDA-Q support.

**Work Items**:
1. Update `CMakeLists.txt`:
   - Add meta-learning source files to `QALLOW_SOURCES`
   - Create `qallow_metalearn` executable target (standalone)
   - Create `qallow_metalearn_cuda` target (CUDA-enabled)
   - Add CUDA-Q detection and conditional compilation

2. Implement CUDA kernels for GP operations:
   - `backend/cuda/metalearn_kernel.cu` (optional, for ≥2x speedup)
   - Parallel covariance matrix computations
   - Batch parameter evaluation

3. Build testing:
   - Run `ctest -R metalearn` after build
   - Generate performance benchmark: meta-learning 100 iterations, compare backends

4. Documentation:
   - Update `BUILD_RUN_GUIDE.md` with meta-learning build steps
   - Add troubleshooting section (CUDA-Q not found → fallback to CPU)

**Deliverables**:
- ✅ `CMakeLists.txt` updates (30+ lines)
- ✅ `backend/cuda/metalearn_kernel.cu` (optional, 100+ lines if included)
- ✅ `tests/CMakeLists.txt` meta-learning test targets
- ✅ Build documentation updates (50+ lines)

**Acceptance Criteria**:
- `./build_unified.sh` compiles all meta-learning targets without errors
- `ctest -R metalearn` passes all meta-learning tests
- Parallel CUDA build is ≥2x faster than CPU build (if CUDA available)
- `build/qallow_metalearn_cuda --help` displays correct options

**Dependencies**: Tasks 2, 3, 4  
**Blocks**: Task 7, Task 8  
**Estimated Effort**: 2 hours

---

### TASK 7: Constitution Audit & Ethics Compliance [PRE-REQ: All Core Tasks 1-6]

**Objective**: Verify 100% Constitution alignment and ethics constraints.

**Work Items**:
1. Create `specs/004-agi-evolution/CONSTITUTION_AUDIT.md`:
   - Audit each § of Constitution against meta-learning implementation
   - §1.2 (Self-Improvement): Verify meta-learning doesn't escape ethics bounds
   - §2.1 (Ethics-First): Verify all loss functions respect E = S + C + H
   - §3.1 (Transparency): Verify telemetry captures all decisions
   - §4.0 (Canonical Structure): Verify files in correct locations
   - §5.0 (Minimal Dependencies): Verify no new external packages

2. Create automated audit tool:
   - `scripts/audit_metalearn_ethics.sh` (runs Constitution checks)
   - Parses source files for ethics constraints
   - Verifies telemetry includes ethics scores
   - Reports pass/fail for each §

3. Manual audit:
   - Review `src/mind/quantum_learn.c` for potential escape paths
   - Verify loss function bounds prevent objective hacking
   - Confirm all backend selection is logged

**Deliverables**:
- ✅ `specs/004-agi-evolution/CONSTITUTION_AUDIT.md` (100+ lines)
- ✅ `scripts/audit_metalearn_ethics.sh` (80+ lines)
- ✅ Audit report with pass/fail for each § (CSV or markdown)

**Acceptance Criteria**:
- 100% of Constitution § checks pass
- Ethics constraints integrated into Bayesian loss function
- Telemetry proves all meta-learning decisions bounded by ethics
- Audit tool produces reproducible results

**Dependencies**: Tasks 1-6 (complete implementation)  
**Blocks**: Task 8 (final validation)  
**Estimated Effort**: 1-2 hours

---

### TASK 8: Documentation, Examples & Validation [PRE-REQ: All Tasks 1-7]

**Objective**: Create comprehensive documentation, examples, and final validation.

**Work Items**:
1. Create `docs/METALEARN_GUIDE.md` (100+ lines):
   - Quick-start: `./qallow run meta-learning --backend=auto --iterations=100`
   - Explanation of Bayesian optimization mechanics
   - Quantum backend options (CUDA-Q vs Cirq vs CPU)
   - Custom loss function examples (Python, C)
   - Telemetry interpretation guide
   - Troubleshooting (backend not found → fallback steps)

2. Create worked examples:
   - `examples/metalearn_sphere_function.c` (minimize sphere function)
   - `examples/metalearn_rastrigin.py` (minimize Rastrigin function with Python)
   - `examples/metalearn_custom_loss.c` (user-defined loss template)

3. Create integration test:
   - `tests/test_metalearn_integration.c` (end-to-end workflow)
   - Tests all backends in sequence (CUDA-Q → CUDA → Cirq → CPU)
   - Verifies telemetry output
   - Benchmarks convergence speed

4. Final validation:
   - Run `./qallow run meta-learning --iterations=100 --backend=auto`
   - Verify CSV output: `data/logs/metalearn_*.csv`
   - Verify no regressions: Phase 12-15 still execute correctly
   - Run Constitution audit: all § pass

**Deliverables**:
- ✅ `docs/METALEARN_GUIDE.md` (150+ lines)
- ✅ 3+ worked examples in `examples/` (150+ lines total)
- ✅ Integration test: `tests/test_metalearn_integration.c` (100+ lines)
- ✅ Final validation report (pass/fail checklist)

**Acceptance Criteria**:
- Documentation clear enough for user to run meta-learning without external help
- All 3+ examples run successfully and produce expected output
- Integration test passes (all backends functional)
- No regressions in existing Phases 12-15
- Constitution audit fully passes

**Dependencies**: All Tasks 1-7 complete  
**Blocks**: FEATURE COMPLETE  
**Estimated Effort**: 2-3 hours

---

## 4. Task Dependency Graph

```
Task 1 (Cognitive State)
   ├─→ Task 2 (Bayesian Core)
   │      ├─→ Task 3 (CUDA-Q Backend)
   │      │      ├─→ Task 5 (Multi-Backend Orchestration)
   │      │      │      └─→ Task 6 (Build System)
   │      │      │             └─→ Task 7 (Constitution Audit)
   │      │      │                    └─→ Task 8 (Documentation & Validation)
   │      │      └─→ Task 6 (Build System) ──┘
   │      │
   │      ├─→ Task 4 (Cirq Backend)
   │      │      └─→ Task 5 (Multi-Backend Orchestration) ──┘
   │      │
   │      └─→ Task 5 (Multi-Backend Orchestration) ──────┘
```

**Critical Path**: Task 1 → Task 2 → Task 5 → Task 6 → Task 7 → Task 8 (9-13 hours)

---

## 5. Execution Timeline

### Phase A: Foundation (Hours 0-3)
- **Task 1**: Cognitive state & ethics (2 hrs)
- **Task 1 Parallel**: Build system setup for meta-learning (1 hr)

### Phase B: Core Engine (Hours 3-7)
- **Task 2**: Bayesian optimization (3-4 hrs)
- **Task 2 Parallel**: Initial unit tests (1 hr)

### Phase C: Quantum Integration (Hours 7-10)
- **Task 3**: CUDA-Q backend (2 hrs)
- **Task 4**: Cirq backend (2 hrs)
- **Task 4 Parallel**: CLI framework (1 hr)

### Phase D: Integration & Polish (Hours 10-13)
- **Task 5**: Multi-backend orchestration (2 hrs)
- **Task 6**: Final build integration (1 hr)
- **Task 7**: Constitution audit (1-2 hrs)
- **Task 8**: Documentation & validation (2-3 hrs)

---

## 6. Success Metrics & Validation Checkpoints

| Checkpoint | Metric | Pass Condition |
|------------|--------|----------------|
| CP1: Task 1 Complete | Cognitive state unit tests | 4/4 tests pass, JSON serialization works |
| CP2: Task 2 Complete | Bayesian optimizer convergence | Sphere function: x→3 within 30 iterations |
| CP3: Task 3 Complete | CUDA-Q integration | CPU fallback 100%, CUDA-Q 2x speedup if available |
| CP4: Task 5 Complete | CLI interface | `./qallow run meta-learning` executes, produces telemetry |
| CP5: Task 7 Complete | Constitution compliance | 100% § audit pass, ethics bounded |
| CP6: Task 8 Complete | Integration test | All backends work, no regressions, docs clear |

---

## 7. Approval Gate Checkpoints

| Gate | Condition | Owner |
|------|-----------|-------|
| G1 | Task 1-2 code review | @User |
| G2 | Task 3-4 quantum backend approval | @User |
| G3 | Task 5-6 CLI/build integration | @User |
| G4 | Task 7 Constitution audit pass | @User |
| G5 | Task 8 final documentation | @User (before GitHub push) |

---

## 8. Assumptions & Constraints

### Assumptions
- A1: User will review and approve each task checkpoint before proceeding to next task
- A2: No breaking changes to existing Phase 12-15 workflow
- A3: Meta-learning performance targets (60% convergence speedup) are achievable with classical + quantum

### Constraints
- C1: No new external package dependencies (CUDA/CUDA-Q are optional runtime)
- C2: CPU fallback mandatory (even if CUDA-Q unavailable)
- C3: Ethics constraints cannot be disabled or bypassed
- C4: Total effort <15 hours (to fit within sprint budget)

---

## 9. Post-Implementation (Phase 005+)

Once Feature 004 Phase 1 complete, Feature 005 will address:
- Phase 2 (Cognitive Architecture): Unified reasoning framework
- Phase 3 (Self-Improvement): Recursive meta-learning (depth >3)
- Phase 4 (Generalization): Domain-agnostic problem solving
- Phase 5 (Consciousness): Recursive self-awareness

---

**Plan Version: 1.0.0**  
**Last Updated: 2025-11-07**  
**Status: ✅ READY FOR TASK BREAKDOWN**
