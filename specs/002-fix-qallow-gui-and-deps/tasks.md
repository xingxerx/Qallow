# Tasks: Qallow GUI & Dependency Completion

**Input**: Design documents from `/specs/002-fix-qallow-gui-and-deps/`
**Prerequisites**: plan.md, spec.md

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

---

## Phase 1: Setup & Foundational Tasks (Blocking Prerequisites for US1)

**Purpose**: Establish a reproducible build environment. This corresponds to User Story 1 but contains foundational work required for all subsequent tasks.

**Goal**: A developer can run a single script to build the entire application, including all dependencies.

**Independent Test**: On a clean Ubuntu 24.04 + CUDA 12 environment, `bootstrap.sh` must succeed, and all binaries must be executable.

- [ ] T001 Create `requirements.txt` with `qiskit`, `qiskit-aer`, and `sentence-transformers` at the root of the repository.
- [ ] T002 [P] Modify `CMakeLists.txt` to add a CPU-only fallback for all CUDA-optional phases.
- [ ] T003 [P] Scan the codebase for loose files, missing CMake targets, and undefined symbols, and fix them.
- [ ] T004 Update `bootstrap.sh` to install Python dependencies from `requirements.txt` and any missing system packages.
- [ ] T005 Ensure the `data/logs/` directory is created on first run if it is missing.

---

## Phase 2: User Story 2 - Functional GUI Interaction (Priority: P2)

**Goal**: Wire up the GUI to make it fully interactive and functional.

**Independent Test**: Launch the GUI, click each phase button, and verify the correct backend phase is triggered and the GUI provides feedback.

- [ ] T006 [US2] Fix the broken Phase 11 Qiskit bridge in `python/quantum/run_phase11.py`.
- [ ] T007 [US2] Fix the broken Phase 14 QAOA tuner in `python/quantum/run_phase14.py`.
- [ ] T008 [US2] In `native_app/src/main.rs`, wire all GUI buttons to trigger the corresponding `qallow` phase via CLI subprocess.
- [ ] T009 [P] [US2] In `native_app/src/main.rs`, implement a status bar or telemetry panel that tails and displays the latest CSV log from `data/logs/`.

---

## Phase 3: User Story 3 - Ethics Layer Enforcement (Priority: P3)

**Goal**: Implement the critical ethics safety check.

**Independent Test**: Attempt to run Phase 13 without an ethics model loaded; verify it is blocked and an error is displayed.

- [ ] T010 [US3] In the backend logic (likely C/C++), add a guard to block Phase 13 if no ethics model is loaded.
- [ ] T011 [US3] In `native_app/src/main.rs`, ensure that when the Phase 13 block is triggered, an error message is displayed in the GUI status bar.

---

## Phase 4: Polish & Finalization

**Purpose**: Final checks and documentation.

- [ ] T012 Create a `fix_deps.sh` script that encapsulates all dependency installation and fixing logic.
- [ ] T013 Update `bootstrap.sh` to call the new `fix_deps.sh` script.
- [ ] T014 Perform a full, end-to-end test of the entire GUI flow and build process.

## Dependencies & Parallel Execution

- **Phase 1 (US1)** is a prerequisite for all other phases. Tasks within Phase 1 (T002, T003) can be done in parallel.
- **Phase 2 (US2)** can begin after Phase 1 is complete.
  - **Parallelism within US2**: T006, T007, T008, and T009 can be worked on in parallel, though T008 (button wiring) is the core task.
- **Phase 3 (US3)** can be worked on in parallel with Phase 2, as it primarily involves backend logic and GUI feedback for a specific case.
- **Phase 4** should be done last.

### Suggested MVP

The Minimum Viable Product (MVP) is the completion of **Phase 1**, which delivers a reproducible build and a runnable, if not fully interactive, application.
