# Implementation Plan: Qallow GUI & Dependency Completion

**Feature Branch**: `002-fix-qallow-gui-and-deps`
**Created**: 2025-11-10
**Status**: Draft

## Tech Stack
- **GUI**: Rust + FLTK (`native_app/`)
- **Backend**: Existing C/CUDA (`backend/`) + Python bridges (`python/`)
- **Build**: CMake + Cargo + `bootstrap.sh`
- **Runtime**: `./build/qallow` CLI as subprocess from Rust
- **Logs**: Tail `data/logs/*.csv` → GUI table
- **Deps**: `requirements.txt`, system packages (`apt`), Rust packages (`cargo`)

## Implementation Strategy

The implementation will follow the user stories in order of priority, ensuring that each stage results in a testable and valuable increment.

1.  **P1: Seamless Build and Run**: The highest priority is to create a reproducible build environment. This involves creating a `requirements.txt` file, updating the `bootstrap.sh` script to handle all dependencies, and ensuring CMake can build the project with CPU fallbacks.
2.  **P2: Functional GUI Interaction**: Once the application builds reliably, the focus shifts to making the GUI functional. This involves wiring up the FLTK buttons in the Rust `native_app` to call the backend `qallow` executable. This also includes fixing the broken Python-based phases (11 and 14) and implementing the live telemetry view.
3.  **P3: Ethics Layer Enforcement**: With the core functionality in place, the final step is to implement the critical safety feature: ensuring the ethics layer correctly blocks Phase 13 if the required models are not loaded.

This approach ensures we have a working, buildable application before adding complex GUI interactions, and leaves the critical safety check for last, building upon a stable foundation.

## High-Level Task List

| Task | Component | Description |
|------|-----------|-------------|
| 1. Create `requirements.txt` | Python | Define all Python dependencies (`cirq`, `sentence-transformers`, etc.). |
| 2. Enhance `bootstrap.sh` | Build | Update the main setup script to install Python deps from `requirements.txt` and other system packages. |
| 3. Add CMake CUDA Fallback | Build | Modify `CMakeLists.txt` to allow CPU-only builds for CUDA-optional phases. |
| 4. Fix Broken Python Phases | Python | Repair the cirq bridge (Phase 11) and QAOA tuner (Phase 14). |
| 5. Wire GUI Buttons | Rust GUI | Connect buttons in `native_app/src/main.rs` to execute `qallow` phases as subprocesses. |
| 6. Implement GUI Telemetry | Rust GUI | Add a panel to the GUI to tail and display logs from `data/logs/`. |
| 7. Implement Ethics Guard | Backend/GUI | Add logic to block Phase 13 execution if the ethics model isn't loaded and show an error in the GUI. |
| 8. Create Log Directory | Backend | Ensure `data/logs/` is created on startup if it doesn't exist. |
| 9. Full System Test | Testing | Perform an end-to-end test of the `bootstrap.sh` script on a clean environment and test all GUI functionality. |
