# Feature Specification: Qallow GUI & Dependency Completion

**Feature Branch**: `002-fix-qallow-gui-and-deps`  
**Created**: 2025-11-10  
**Status**: Draft  
**Input**: User description: "Fully functional Qallow desktop GUI (Rust/FLTK) with all buttons wired to backend phases, zero missing dependencies, and complete build reproducibility."

## User Scenarios & Testing *(mandatory)*

<!--
  IMPORTANT: User stories should be PRIORITIZED as user journeys ordered by importance.
  Each user story/journey must be INDEPENDENTLY TESTABLE - meaning if you implement just ONE of them,
  you should still have a viable MVP (Minimum Viable Product) that delivers value.
  
  Assign priorities (P1, P2, P3, etc.) to each story, where P1 is the most critical.
  Think of each story as a standalone slice of functionality that can be:
  - Developed independently
  - Tested independently
  - Deployed independently
  - Demonstrated to users independently
-->

### User Story 1 - Seamless Build and Run (Priority: P1)

A developer on a fresh Ubuntu 24.04 environment with CUDA 12 installed can clone the repository, run a single script, and have a fully compiled and functional Qallow application, including the GUI.

**Why this priority**: This is the most critical step for developer onboarding and ensuring the project is reproducible. Without a reliable build, no other feature matters.

**Independent Test**: On a clean OS image, run `bootstrap.sh`. The script must complete without errors, and the `./build/qallow` and `native_app` binaries must be present and executable.

**Acceptance Scenarios**:

1.  **Given** a clean Ubuntu 24.04 system with NVIDIA drivers and CUDA 12 toolkit installed, **When** the user clones the repo and runs `./bootstrap.sh`, **Then** the script successfully installs all system, Python, and Rust dependencies and compiles the entire project.
2.  **Given** a successful bootstrap, **When** the user runs `./build/qallow run unified`, **Then** the application executes its main workflow without dependency-related errors.
3.  **Given** a successful bootstrap, **When** the user launches the Rust GUI from `native_app/`, **Then** the main window appears without any missing library errors.

---

### User Story 2 - Functional GUI Interaction (Priority: P2)

A user can launch the Qallow GUI and interact with all its buttons, triggering the correct backend phases and seeing visual feedback for the running process.

**Why this priority**: The GUI is the primary user-facing component for controlling the AGI. Its functionality is essential for non-developer interaction.

**Independent Test**: Launch the GUI and click each phase button. The application should execute the corresponding phase, and the GUI should indicate that the phase is running.

**Acceptance Scenarios**:

1.  **Given** the Qallow GUI is running, **When** the user clicks the "Run Phase 11" button, **Then** the `qallow` backend is invoked with `phase 11` arguments, and the GUI status bar updates to "Running Phase 11...".
2.  **Given** the Qallow GUI is running, **When** the user clicks the "Run Phase 14" button, **Then** the `qallow` backend is invoked with `phase 14` arguments, and the GUI status bar updates to "Running Phase 14...".
3.  **Given** a phase is running, **When** the backend process emits log data to `data/logs/`, **Then** the GUI's telemetry view updates in near real-time to show the latest log entries.

---

### User Story 3 - Ethics Layer Enforcement (Priority: P3)

The system's ethics layer actively prevents the execution of a critical phase if the necessary ethical constraints are not loaded, providing clear feedback to the user.

**Why this priority**: Ensures the AGI operates within its safety and ethical boundaries, a core principle of the Qallow project.

**Independent Test**: Attempt to run Phase 13 without a pre-loaded ethics model. The system must block the execution and the GUI must display an error.

**Acceptance Scenarios**:

1.  **Given** the Qallow application is started without loading an ethics model, **When** the user attempts to trigger Phase 13 via the GUI, **Then** the action is blocked, and the GUI status bar displays an error message like "Error: Ethics model not loaded. Phase 13 cannot be executed."

---

### Edge Cases

<!--
  ACTION REQUIRED: The content in this section represents placeholders.
  Fill them out with the right edge cases.
-->

-   **Missing CUDA**: If CUDA is not detected on the system, CUDA-specific phases are disabled in the GUI, and the build script uses CPU-only fallbacks where available.
-   **Missing Log Directory**: If `data/logs/` does not exist when the application is first run, it is created automatically.
-   **Partial Build Failure**: If a non-critical component fails to build (e.g., a specific test utility), the core `qallow` executable and GUI should still be usable.

## Requirements *(mandatory)*

<!--
  ACTION REQUIRED: The content in this section represents placeholders.
  Fill them out with the right functional requirements.
-->

### Functional Requirements

-   **FR-001**: The system MUST provide a `bootstrap.sh` script that installs all required system (`apt`), Python (`pip`), and Rust (`cargo`) dependencies.
-   **FR-002**: All GUI buttons in the `native_app` MUST be wired to trigger the corresponding `qallow` backend phase via a CLI subprocess call.
-   **FR-003**: The build system (CMake) MUST correctly link all C/CUDA components, resolving any undefined symbols or linkage errors.
-   **FR-004**: A `requirements.txt` file MUST exist and contain all necessary Python packages, including `qiskit`, `qiskit-aer`, and `sentence-transformers`.
-   **FR-005**: All phases that have a CUDA implementation MUST provide a CPU-only fallback path for systems without a compatible GPU.
-   **FR-006**: The application MUST create the `data/logs/` directory on its first run if it is not already present.
-   **FR-007**: The GUI MUST include a status bar or telemetry panel that displays the tailed output of the latest log file in `data/logs/`.
-   **FR-008**: The system MUST prevent the execution of Phase 13 if an ethics model has not been loaded, and provide feedback to the user.
-   **FR-009**: Broken phase links for Phase 11 (Qiskit bridge) and Phase 14 (QAOA tuner) MUST be repaired and made functional.

### Key Entities *(include if feature involves data)*

-   **Qallow Phase**: A distinct computational stage in the AGI's cognitive cycle, invoked via the `qallow` CLI.
-   **GUI Control**: A button or interactive element in the FLTK GUI.
-   **Telemetry Log**: A CSV file in `data/logs/` containing real-time operational data.
-   **Ethics Model**: A data structure or file that defines the ethical constraints for the AGI.

## Success Criteria *(mandatory)*

<!--
  ACTION REQUIRED: Define measurable success criteria.
  These must be technology-agnostic and measurable.
-->

### Measurable Outcomes

- **SC-001**: `bootstrap.sh` completes with a 100% success rate on a clean, specified environment.
- **SC-002**: All GUI buttons are confirmed to trigger the correct backend phases, with a 0% error rate in invocation.
- **SC-003**: Codebase scan results in zero unresolved dependencies, symbols, or linkage errors.
- **SC-004**: The GUI telemetry updates with a latency of less than 500ms from the log file write.
- **SC-005**: The ethics layer block on Phase 13 is successfully triggered in 100% of test cases where the model is not loaded.
