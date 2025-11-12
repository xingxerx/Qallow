# Feature 004: AGI Evolution - Implementation Tasks

**Feature**: 004-agi-evolution (Phase 1: Meta-Learning)  
**Version**: 1.0.0  
**Status**: Ready for Implementation  
**Total Tasks**: 8  
**Completed**: 0/8  
**Date**: 2025-11-07

---

## Task Execution Status

### Task 1: Cognitive State & Ethics Foundation
- [ ] **Create** `core/include/cognitive.h` (50+ lines)
  - Define `cognitive_state_t` struct
  - Include ethics_score, self_model, goal_vector, learning_rate, backend fields
  - Add struct comments for each field
- [ ] **Implement** `src/constitution.c` initialization (100+ lines)
  - Implement `cognitive_state_init()` function
  - Implement `cognitive_state_to_json()` serialization
  - Implement `cognitive_state_validate()` Constitution compliance check
  - Integrate with existing ethics scoring from `algorithms/ethics_*.c`
- [ ] **Create** unit tests `tests/test_cognitive_state.c` (60+ lines)
  - Test 1: Initialization with default values
  - Test 2: Ethics scoring integration (verify E = S + C + H)
  - Test 3: JSON round-trip serialization/deserialization
  - Test 4: Constitution § compliance validation
- [ ] **Update** `CMakeLists.txt` to compile cognitive state targets
- [ ] **Verify** all tests pass: `ctest -R cognitive_state`
- [ ] **Code Review** required before proceeding to Task 2

**Acceptance Criteria**:
- ✅ All 4 unit tests pass
- ✅ JSON serialization produces valid JSON readable by Python
- ✅ Constitution audit recognizes cognitive state as § IV compliant
- ✅ Build system compiles without errors or warnings

**Status**: [ ] NOT STARTED  
**Owner**: @User (implementation requires approval)  
**Effort**: 2 hours  
**Blocking**: Tasks 2-8

---

### Task 2: Bayesian Optimization Core Engine
- [ ] **Create** `src/mind/quantum_learn.c` (250+ lines)
  - Implement `bayesian_optimizer_t` struct
  - Implement Gaussian Process surrogate model (mean, covariance)
  - Implement Expected Improvement (EI) acquisition function
  - Implement `bayesian_optimize()` main function
- [ ] **Create** `core/include/quantum_learn.h` API header (100+ lines)
  - Function signatures for all public functions
  - Struct definitions for Bayesian optimizer state
  - Documentation comments for each function
- [ ] **Create** `src/mind/gp_surrogate.c` Gaussian Process implementation (150+ lines)
  - Implement Gaussian Process mean function
  - Implement covariance matrix (Matern kernel)
  - Implement posterior distribution estimation
  - Implement uncertainty quantification
- [ ] **Create** unit tests `tests/test_bayesian_opt.c` (100+ lines)
  - Test 1: Convergence on sphere function (x - 3)²
  - Test 2: Convergence on Rastrigin function
  - Test 3: Parameter bounds enforcement
  - Test 4: Telemetry CSV generation
  - Test 5: Memory leak check (valgrind)
- [ ] **Verify** all tests pass: `ctest -R bayesian_opt`
- [ ] **Benchmark** convergence speed: <500ms for 100 iterations on single CPU core
- [ ] **Code Review** required before proceeding to Tasks 3-4

**Acceptance Criteria**:
- ✅ Sphere function optimization: finds x≈3 within 30 iterations
- ✅ Rastrigin function optimization: converges within iteration budget
- ✅ Telemetry CSV output includes headers: iteration, parameters, loss, ethics_score, backend
- ✅ No memory leaks detected by valgrind/sanitizer
- ✅ Execution time <500ms for 100 iterations on CPU

**Status**: [ ] NOT STARTED  
**Owner**: @User (core algorithm - critical for AGI)  
**Effort**: 3-4 hours  
**Blocking**: Tasks 3, 5, 6  
**Prerequisite**: Task 1 complete

---

### Task 3: Quantum-Enhanced Sampling (CUDA-Q Backend)
- [ ] **Detect** CUDA-Q availability at compile-time
  - Check CMake for `libcudaqrt.so` in system paths
  - Set CMake flag `QALLOW_ENABLE_CUDAQ` (auto-detect)
  - Add CMakeLists.txt conditional compilation block
- [ ] **Create** `src/mind/quantum_sampling.c` (150+ lines)
  - Implement `quantum_sample_parameters()` function
  - CUDA-Q backend: Create superposition, measure to get samples
  - Importance weighting for probabilistic sampling
  - Fallback to classical Sobol sequence if CUDA-Q unavailable
- [ ] **Create** `core/include/quantum_sampling.h` API header (50+ lines)
- [ ] **Create** unit tests `tests/test_quantum_sampling.c` (80+ lines)
  - Test 1: CPU fallback always works (even without CUDA-Q)
  - Test 2: CUDA-Q backend if available (optional test, skip if not found)
  - Test 3: Unbiased sampling verification (statistical test)
  - Test 4: Importance weighting correctness
- [ ] **Verify** tests pass: `ctest -R quantum_sampling`
- [ ] **Benchmark** speedup: CUDA-Q backend ≥2x faster than classical (if available)
- [ ] **Code Review** required before proceeding to Task 5

**Acceptance Criteria**:
- ✅ CPU fallback works 100% of the time (zero crashes)
- ✅ CUDA-Q backend when available produces ≥2x speedup on convergence
- ✅ Quantum sampling produces unbiased parameter suggestions
- ✅ Telemetry correctly identifies quantum samples vs classical samples
- ✅ No buffer overflows (Snyk security scan passes)

**Status**: [ ] NOT STARTED  
**Owner**: @User (quantum integration)  
**Effort**: 2-3 hours  
**Blocking**: Task 5, 6  
**Prerequisite**: Task 2 complete

---

### Task 4: Cirq Python Backend Integration
- [ ] **Create** `python/quantum/metalearn_quantum_cirq.py` (100+ lines)
  - Accept parameters via stdin (JSON format)
  - Use Cirq to create parameterized quantum circuits
  - Execute quantum sampling on Cirq simulator
  - Return results via stdout (JSON format)
- [ ] **Create** C interface in `src/mind/quantum_sampling.c` (50+ lines)
  - Subprocess spawning for Cirq backend
  - JSON input/output communication
  - Timeout handling (max 5 seconds per call)
- [ ] **Detect** Cirq availability at runtime
  - Test `python -c "import cirq"` in build process
  - Log availability in telemetry
- [ ] **Create** unit tests `tests/test_cirq_backend.c` (60+ lines)
  - Test 1: Cirq backend subprocess spawning (optional, skip if Cirq unavailable)
  - Test 2: JSON parameter passing/return
  - Test 3: Timeout handling
  - Test 4: Fallback to CPU if subprocess fails
- [ ] **Verify** tests pass: `ctest -R cirq_backend`
- [ ] **Benchmark** overhead: <50ms per Cirq call (including subprocess spawn)
- [ ] **Code Review** required before proceeding to Task 5

**Acceptance Criteria**:
- ✅ Cirq backend available systems detected automatically
- ✅ Subprocess communication works without deadlocks
- ✅ Parameter passing/return via JSON correct format
- ✅ Graceful fallback to CPU if Cirq subprocess fails
- ✅ Execution overhead <50ms per Cirq sampling call

**Status**: [ ] NOT STARTED  
**Owner**: @User (Python quantum integration)  
**Effort**: 2-3 hours  
**Blocking**: Task 5, 6  
**Prerequisite**: Task 2 complete

---

### Task 5: Multi-Backend Orchestration & CLI Integration
- [ ] **Create** `src/mind/metalearn_executor.c` (100+ lines)
  - Implement `metalearn_execute()` wrapper function
  - Auto-detect available backends (CUDA, CUDA-Q, Cirq, CPU)
  - Implement precedence order: CUDA-Q > CUDA > Cirq > CPU
  - Allow user override via `--ml-backend` flag
- [ ] **Extend** CLI in `interface/launcher.c` (50+ lines)
  - Add command: `./qallow run meta-learning`
  - Add flags: `--iterations=N`, `--backend=auto|cuda|cuda-q|cirq|cpu`, `--learning-rate=X`
  - Add objective function options: `--objective=sphere|rastrigin|user_func`
- [ ] **Implement** argument parser (20+ lines)
  - Parse CLI flags
  - Validate iteration count (1-10000)
  - Validate backend selection
- [ ] **Update** help documentation
  - Update `./qallow --help` to list meta-learning commands
  - Add usage examples in help text
- [ ] **Create** integration test `tests/test_metalearn_executor.c` (100+ lines)
  - Test 1: CLI parsing (valid and invalid arguments)
  - Test 2: Backend auto-selection
  - Test 3: Backend override (force CPU, etc.)
  - Test 4: Help text display
- [ ] **Verify** tests pass: `ctest -R metalearn_executor`
- [ ] **Code Review** required before proceeding to Task 6

**Acceptance Criteria**:
- ✅ `./qallow run meta-learning --backend=auto` selects best available backend
- ✅ `./qallow run meta-learning --ml-backend=cpu --iterations=50` forces CPU
- ✅ `./qallow --help` displays meta-learning command documentation
- ✅ All backend selections logged to telemetry with backend name

**Status**: [ ] NOT STARTED  
**Owner**: @User (CLI integration)  
**Effort**: 2 hours  
**Blocking**: Tasks 6, 7, 8  
**Prerequisite**: Tasks 2, 3, 4 complete

---

### Task 6: Build System Integration & CUDA Targets
- [ ] **Update** `CMakeLists.txt` (30+ lines)
  - Add meta-learning source files to `QALLOW_SOURCES`
  - Create `qallow_metalearn` executable target (CPU-only)
  - Create `qallow_metalearn_cuda` target (CUDA-enabled)
  - Add CUDA-Q detection and conditional compilation
- [ ] **Implement** CUDA kernels for GP operations (optional, 100+ lines)
  - Create `backend/cuda/metalearn_kernel.cu`
  - Parallel covariance matrix computations
  - Batch parameter evaluation
  - Expected to achieve ≥2x speedup vs CPU
- [ ] **Update** build test targets
  - Add `ctest -R metalearn` test suite
  - Generate performance benchmark after build
  - Compare backend performance (CPU vs CUDA vs CUDA-Q vs Cirq)
- [ ] **Document** build steps in `BUILD_RUN_GUIDE.md`
  - Add meta-learning build instructions (50+ lines)
  - Add troubleshooting section (CUDA-Q not found → fallback)
  - Add performance tuning section
- [ ] **Verify** build system works
  - Run `./build_unified.sh` successfully
  - Verify all meta-learning targets compiled
  - Run `ctest -R metalearn` successfully
  - Parallel build is ≥2x faster than CPU build (if CUDA available)
- [ ] **Code Review** required before proceeding to Task 7

**Acceptance Criteria**:
- ✅ `./build_unified.sh` compiles all meta-learning targets without errors
- ✅ `ctest -R metalearn` passes all meta-learning tests
- ✅ Parallel CUDA build is ≥2x faster than CPU build (if CUDA available)
- ✅ `build/qallow_metalearn_cuda --help` displays correct options

**Status**: [ ] NOT STARTED  
**Owner**: @User (build integration)  
**Effort**: 2 hours  
**Blocking**: Tasks 7, 8  
**Prerequisite**: Tasks 2, 3, 4, 5 complete

---

### Task 7: Constitution Audit & Ethics Compliance Verification
- [ ] **Execute** Constitution audit on implementation
  - Run existing Constitution audit checks: `scripts/audit_constitution.sh` (if available)
  - Manually verify § 1.2 (Self-Optimization Without Escape) implementation
  - Verify ethics constraints are hard-coded (not policy-based)
  - Verify all loss functions respect ethics bounds
- [ ] **Create** automated audit tool: `scripts/audit_metalearn_ethics.sh` (80+ lines)
  - Parse `src/mind/quantum_learn.c` for ethics constraints
  - Verify loss function includes ethics penalty term
  - Check telemetry includes ethics_score column
  - Report pass/fail for each Constitution principle
- [ ] **Execute** audit tool
  - Run `bash scripts/audit_metalearn_ethics.sh`
  - Verify all § principles pass
- [ ] **Manual ethics review**
  - Code review of `src/mind/quantum_learn.c` for ethics implementation
  - Verify loss function: `L_effective(x) = L_user(x) + λ₁ · penalty_ethics_violation(x)`
  - Confirm penalty term prevents objective hacking
- [ ] **Generate** audit report: `specs/004-agi-evolution/CONSTITUTION_AUDIT_RESULTS.md`
  - Pass/fail for each principle (§0-§6.1)
  - Evidence for each pass (code location, test proof)
  - Summary: 100% principles passing or specific gaps
- [ ] **Code Review** required (ethics-critical task)

**Acceptance Criteria**:
- ✅ 100% of Constitution § checks pass
- ✅ Ethics constraints integrated into Bayesian loss function
- ✅ Telemetry proves all meta-learning decisions bounded by ethics
- ✅ Audit tool produces reproducible results
- ✅ No gaps between design (Task 1-6) and implementation

**Status**: [ ] NOT STARTED  
**Owner**: @User (ethics audit - critical)  
**Effort**: 1-2 hours  
**Blocking**: Task 8  
**Prerequisite**: Tasks 1-6 complete

---

### Task 8: Documentation, Examples & Integration Testing
- [ ] **Create** `docs/METALEARN_GUIDE.md` (150+ lines)
  - Quick-start: `./qallow run meta-learning --backend=auto --iterations=100`
  - Explanation of Bayesian optimization mechanics (50+ lines)
  - Quantum backend options and when to use each (CUDA-Q vs Cirq vs CPU)
  - Custom loss function examples (Python and C, 30+ lines)
  - Telemetry interpretation guide (20+ lines)
  - Troubleshooting section (backend not found → fallback steps)
- [ ] **Create** worked examples in `examples/` (150+ lines total)
  - Example 1: `examples/metalearn_sphere_function.c` (minimize sphere function)
  - Example 2: `examples/metalearn_rastrigin.py` (minimize Rastrigin with Python)
  - Example 3: `examples/metalearn_custom_loss.c` (user-defined loss template)
  - Each example: 40+ lines with comments
- [ ] **Create** integration test: `tests/test_metalearn_integration.c` (100+ lines)
  - Test 1: End-to-end meta-learning workflow
  - Test 2: All backends in sequence (CUDA-Q → CUDA → Cirq → CPU)
  - Test 3: Telemetry CSV output correctness
  - Test 4: No regressions in Phase 12-15 workflow
  - Test 5: Convergence speedup verified (≥60% vs baseline)
- [ ] **Execute** final validation
  - Run `./qallow run meta-learning --iterations=100 --backend=auto`
  - Verify CSV output: `data/logs/metalearn_*.csv` generated
  - Verify no regressions: Phase 12-15 still execute correctly
  - Run Constitution audit: all § pass
  - Run integration tests: `ctest -R integration`
- [ ] **Generate** final validation report
  - Pass/fail checklist for all success criteria (SC1-SC10 from spec)
  - Telemetry samples (screenshot of convergence plot)
  - Performance metrics (convergence speed, backend speedups)
  - Summary: Feature 004 ready for merge
- [ ] **Code Review** final deliverables
  - Documentation clarity review
  - Example code correctness review
  - Integration test completeness review

**Acceptance Criteria**:
- ✅ Documentation clear enough for user to run meta-learning without external help
- ✅ All 3+ examples run successfully and produce expected output
- ✅ Integration test passes (all backends functional)
- ✅ No regressions in existing Phases 12-15
- ✅ Constitution audit fully passes (100% principles)
- ✅ Final validation report confirms Feature 004 complete

**Status**: [ ] NOT STARTED  
**Owner**: @User (final delivery)  
**Effort**: 2-3 hours  
**Blocking**: Feature Complete  
**Prerequisite**: Tasks 1-7 complete

---

## Summary Table

| Task | Title | Effort | Blocked By | Blocks | Status |
|------|-------|--------|-----------|--------|--------|
| 1 | Cognitive State & Ethics | 2h | None | 2-8 | [ ] |
| 2 | Bayesian Optimization | 3-4h | 1 | 3,5,6 | [ ] |
| 3 | CUDA-Q Backend | 2-3h | 2 | 5,6 | [ ] |
| 4 | Cirq Python Backend | 2-3h | 2 | 5,6 | [ ] |
| 5 | Multi-Backend Orchestration | 2h | 2,3,4 | 6,7,8 | [ ] |
| 6 | Build System Integration | 2h | 2,3,4,5 | 7,8 | [ ] |
| 7 | Constitution Audit | 1-2h | 1-6 | 8 | [ ] |
| 8 | Documentation & Testing | 2-3h | 1-7 | COMPLETE | [ ] |

**Total Effort**: 9-13 hours  
**Critical Path**: 1 → 2 → 5 → 6 → 7 → 8 (9-13 hours)  
**Parallelization**: Tasks 3-4 can run simultaneously after Task 2

---

## Approval Gates & Sign-Offs

**Before Task 1 begins:**
- [ ] User approves task list and effort estimate
- [ ] User confirms Constitutional alignment (see CONSTITUTION_AUDIT.md)
- [ ] User confirms no breaking changes to existing system

**Between Task Groups:**
- [ ] Task 1-2 code review required (core engine)
- [ ] Task 3-4 quantum backend approval required
- [ ] Task 5-6 CLI/build integration approval required
- [ ] Task 7 Constitution audit 100% pass required
- [ ] Task 8 documentation and integration test review required

**Before Main Merge:**
- [ ] All 8 tasks marked [X] complete
- [ ] All success criteria met
- [ ] Constitution audit passes 100%
- [ ] No regressions in Phases 12-15
- [ ] User final approval to push to main

---

## Status Legend

- [ ] = NOT STARTED (0% complete)
- [/] = IN PROGRESS (25-75% complete)
- [~] = BLOCKED (waiting on dependency or approval)
- [!] = NEEDS REVIEW (complete but requires code review)
- [X] = COMPLETE (100% done, all acceptance criteria met)

---

**Feature 004 Status**: ✅ **SPECIFICATION COMPLETE, READY FOR IMPLEMENTATION**  
**Next Action**: User approval for Task 1 start  
**Estimated Completion**: ~9-13 hours from Task 1 start

---

*For detailed task descriptions, see `plan.md`. For Constitutional alignment, see `CONSTITUTION_AUDIT.md`. For feature goals, see `spec.md`.*
