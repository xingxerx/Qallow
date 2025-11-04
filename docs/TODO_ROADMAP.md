# Qallow Development Roadmap

**Last Updated:** November 4, 2025  
**Total Outstanding Items:** 39 tracked TODOs  
**Status:** Production-ready with planned improvements

---

## 📋 Issue Categories & Priorities

### 🔴 **CRITICAL** - Blocking Production Features

#### 1. **CUDA Parallel Execution** 
- **File:** `backend/cpu/multi_pocket.c` (Line: TODO comment)
- **Issue:** Multi-pocket scheduler lacks GPU kernels
- **Impact:** Cannot parallelize pocket execution on CUDA
- **Effort:** 8-16 hours
- **Status:** ⏳ Planned
- **Details:** 
  - Create `backend/cuda/multi_pocket.cu`
  - Implement parallel pocket state propagation
  - Add CUDA stream management
  - Benchmark against CPU implementation
- **Owners:** CUDA team

#### 2. **Semantic Memory Persistence**
- **File:** `backend/cpu/semantic_memory.c` (Line: TODO comment)
- **Issue:** Using in-memory storage, needs LMDB for persistence
- **Impact:** Memory loss on restart, no state recovery
- **Effort:** 4-8 hours
- **Status:** ⏳ Planned
- **Details:**
  - Integrate LMDB database backend
  - Implement snapshot/restore logic
  - Add migration utilities
  - Test crash recovery
- **Owners:** Backend team

#### 3. **Variable Naming Consistency**
- **Files:** 14 files with single/double-letter variables
- **Issue:** TODO comments for poor naming (e.g., `t`, `c`, `h`, `w`, `v`, `p`, `e`, `g`, `r`)
- **Impact:** Code readability, maintenance
- **Effort:** 6-10 hours
- **Status:** ⏳ High priority
- **Details:**
  - `interface/main.c`: `t` → `tick_index`
  - `interface/qallow_ui.c`: `c`, `w`, `h`, `y` → descriptive names
  - `backend/cpu/phase12_elasticity.c`: `t` → `time_step`
  - `backend/cpu/qallow_kernel.c`: `e`, `g` → descriptive names
  - `src/mind/attention.c`: `h` → `head_index`
  - `src/mind/memory.c`: `p` → `pattern_idx`
  - `src/quantum/entanglement_bridge.c`: `v` → `parsed_value`
  - `algorithms/quantum_suite.c`: `c` → `char_ptr`
  - `backend/cpu/chronometric.c`: `n` → `history_count`
  - `backend/cpu/ethics.c`: `w` → `weight_value`
  - `virtual_computer_c/photonic.c`: `r`, `t` → descriptive names
- **Owners:** Code quality team

---

### 🟡 **HIGH** - Important Features

#### 4. **Launcher Module Implementation**
- **File:** `interface/launcher.c` (Line: TODO comment)
- **Issue:** Several modules listed as unimplemented
- **Impact:** CLI entry point incomplete
- **Effort:** 4-6 hours
- **Status:** ⏳ Planned
- **Details:**
  - Add module initialization when ready
  - Complete phase registry
  - Add error handling for missing modules
- **Owners:** Interface team

#### 5. **Performance Optimization - malloc in loops**
- **Files:** Multiple C files with nested allocations
- **Issue:** Lightning agent flags malloc in loops (memory leak risk)
- **Impact:** Performance regression, memory waste
- **Effort:** 2-4 hours
- **Status:** ⏳ High priority
- **Details:**
  - Move allocation outside loops where applicable
  - Use object pooling for repeated allocations
  - Profile before/after allocation patterns
- **Owners:** Performance team

#### 6. **Unit Test Expansion**
- **Current:** 6 tests (ethics, DL, CUDA, gray code, kernels, alg_ccc)
- **Gap:** Missing tests for quantum algorithms, CLI, integration flows
- **Effort:** 8-12 hours
- **Status:** ⏳ Recommended
- **Target:** 15+ tests (25% increase)
- **Details:**
  - Add quantum algorithm tests (Grover's, VQE, Shor's)
  - Add CLI integration tests
  - Add phase execution tests
  - Add error path coverage
- **Owners:** QA team

---

### 🟢 **MEDIUM** - Nice-to-Have Improvements

#### 7. **CUDA-Q Integration**
- **File:** `python/agi_cuda_accelerator.py`
- **Issue:** CUDA kernel calls via ctypes not fully implemented
- **Impact:** Can use CUDA-Q when available
- **Effort:** 6-8 hours
- **Status:** ⏳ Future enhancement
- **Details:**
  - Complete ctypes bindings
  - Add CUDA-Q circuit compilation
  - Benchmark against Cirq backend
- **Owners:** Quantum team

#### 8. **Performance Telemetry**
- **Files:** Multiple analysis files
- **Issue:** No systematic profiling for phases 13-15
- **Impact:** Can't track optimization opportunities
- **Effort:** 4-6 hours
- **Status:** ⏳ Planned
- **Details:**
  - Add timing instrumentation to phases
  - Create telemetry dashboard
  - Profile CPU/CUDA execution
  - Generate comparison reports
- **Owners:** Performance team

#### 9. **Multi-GPU Support**
- **File:** `CMakeLists.txt` and CUDA backend
- **Issue:** Only single GPU supported (no `CUDA_VISIBLE_DEVICES` orchestration)
- **Impact:** 4-8x scaling potential on multi-GPU systems
- **Effort:** 8-12 hours
- **Status:** ⏳ Future feature
- **Details:**
  - Add GPU device enumeration
  - Implement work distribution
  - Add device affinity management
  - Test on 4-GPU system
- **Owners:** CUDA team

#### 10. **Error Handling Standardization**
- **Files:** Multiple modules
- **Issue:** Error codes and propagation not uniformly structured
- **Impact:** Inconsistent error messages, harder debugging
- **Effort:** 4-8 hours
- **Status:** ⏳ Planned
- **Details:**
  - Standardize error code definitions
  - Create error registry
  - Implement consistent error propagation
  - Add context-aware messages
- **Owners:** Backend team

---

### 💬 **LOW** - Documentation & Polish

#### 11. **Code Comments in Profiling Sections**
- **Files:** `sections/*.py` (IssueSlotUtilization, LaunchStatistics, TheoreticalOccupancy, RequestedMetrics)
- **Issue:** TODO comments suggest improvements for multi-launch workloads
- **Effort:** 2-3 hours
- **Status:** ⏳ Documentation
- **Details:**
  - Clarify weighted average calculations
  - Document parameter constraints
  - Add examples for multi-launch scenarios
- **Owners:** Documentation team

#### 12. **CI/CD Dashboard**
- **Issue:** No visualization of build/test metrics
- **Effort:** 6-8 hours
- **Status:** ⏳ Future nice-to-have
- **Details:**
  - Create dashboard with build times
  - Track test results over time
  - Visualize performance trends
  - Alert on regressions
- **Owners:** DevOps team

---

## 📊 Summary Statistics

| Category | Count | Est. Hours | Priority |
|----------|-------|-----------|----------|
| Critical | 3 | 20-34 | 🔴 |
| High | 3 | 14-22 | 🟡 |
| Medium | 4 | 22-34 | 🟢 |
| Low | 2 | 8-11 | 💬 |
| **Total** | **12** | **64-101** | — |

---

## 🗓️ Recommended Timeline

### **Phase 1: Critical Path (Week 1)**
- ✅ Variable naming (6-10 hrs) → Code quality
- ✅ CUDA parallel execution (8-16 hrs) → Performance
- ✅ Semantic memory persistence (4-8 hrs) → Reliability

**Outcome:** Production-ready multi-GPU support, persistent state

### **Phase 2: Robustness (Week 2)**
- ✅ Unit test expansion (8-12 hrs) → Coverage
- ✅ Error handling standardization (4-8 hrs) → Debugging
- ✅ Launcher module completion (4-6 hrs) → Feature complete

**Outcome:** 25+ tests, consistent error handling, feature complete

### **Phase 3: Optimization (Week 3)**
- ✅ Performance telemetry (4-6 hrs) → Data-driven optimization
- ✅ CUDA-Q integration (6-8 hrs) → Advanced features
- ✅ malloc pattern fixes (2-4 hrs) → Memory efficiency

**Outcome:** Performance profiling, advanced quantum support

### **Phase 4: Polish (Week 4)**
- ✅ Multi-GPU scaling (8-12 hrs) → Enterprise features
- ✅ Profiling section improvements (2-3 hrs) → Documentation
- ✅ CI/CD dashboard (6-8 hrs) → Visibility

**Outcome:** Enterprise-ready, fully observable system

---

## 🎯 Success Metrics

### Immediate (This Week)
- [ ] All 14 variable names refactored
- [ ] CUDA parallel execution designed
- [ ] Semantic memory persistence spec approved
- [ ] Unit tests increased to 12+

### Short-term (This Month)
- [ ] CUDA parallel execution implemented
- [ ] Semantic memory with LMDB operational
- [ ] 25+ unit tests passing
- [ ] Error handling standardized

### Medium-term (This Quarter)
- [ ] Multi-GPU support production-ready
- [ ] Performance telemetry dashboard live
- [ ] CUDA-Q integration complete
- [ ] 50+ unit tests with >80% coverage

---

## 🚀 Current Status: Production-Ready

The codebase is **production-ready** with these items tracked for future enhancement:

**Already Completed ✅**
- Build system (CMake, 16 targets)
- Test suite (6 tests, all passing)
- Agent safety (human review required)
- Documentation (14+ guides)
- CI pipeline (GitHub Actions ready)

**Ready for Implementation 📋**
- Variable naming refactoring
- CUDA parallelization
- Persistent state management
- Extended test coverage

---

## 📞 Contributing

To work on these items:

1. Pick an item from the roadmap
2. Check status (not already in progress)
3. Assign yourself as owner
4. Create a feature branch: `feature/roadmap-{number}-{name}`
5. Update status when complete
6. Submit PR with tests
7. Update this roadmap

---

## 📝 Notes

- **Total tracked TODOs:** 39 (across all categories)
- **New code TODOs added by agent:** Most are naming/performance suggestions
- **Blocked by:** CUDA toolkit (Phase 2), LMDB library (Phase 1)
- **Team coordination:** Requires 2-3 engineers, ~4 weeks for full completion
- **Risk:** Low (all items are enhancements, not blockers)

---

**Last reviewed:** November 4, 2025  
**Next review:** November 11, 2025
