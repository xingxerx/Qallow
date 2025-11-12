---
applyTo: '**'
---

# Qallow Agent Playbook

This document provides essential guidelines for AI agents contributing to the Qallow project. Adhering to these patterns is critical for maintaining codebase stability and architectural integrity.

## 1. Core Architecture & Philosophy

Qallow is a hybrid quantum-classical AGI platform. Its architecture is designed for high-performance, reproducible research into quantum ethics and cognition.

-   **C/CUDA Core Engine**: The performance-critical logic resides in C and CUDA.
    -   **CPU Implementations**: `backend/cpu/`
    -   **GPU (CUDA) Implementations**: `backend/cuda/`
    -   **Shared Data Structures**: `core/include/` and `include/qallow/`. Any change to a core data structure like `CognitiveState` must be propagated across both C and CUDA backends.
-   **Python Orchestration Layer**: Python is used for high-level orchestration, bridging to quantum libraries, and running experiments.
    -   **Main Orchestrator**: `python/quantum/orchestrator.py` is the primary entry point for quantum workflows.
    -   **Quantum Bridges**: `python/quantum/cuda_q_bridge.py` and `python/quantum/cirq_bridge.py` interface with quantum SDKs.
-   **Mandatory CUDA-Q Backend**: The `CudaQBridge` is **not optional**. The primary `QuantumOrchestrator` is hard-wired to use it and will raise an error if the `cudaq` library is not available. Fallback mechanisms to Cirq or other simulators have been removed from the production path to ensure accuracy and testability.

## 2. Key Technologies

-   **Core Languages**: C, C++ (for CUDA), Python 3.11+
-   **GPU Acceleration**: CUDA 12.0+
-   **Quantum SDK**: **NVIDIA CUDA-Q** (mandatory for core quantum tasks). Cirq is present for comparison/testing only.
-   **Build System**: CMake (`CMakeLists.txt`) is the source of truth for building all C/CUDA components.
-   **Testing**:
    -   C/CUDA tests are run with `ctest`.
    -   Python tests are run with `pytest`.
-   **AI Baseline**: The `DeepSeek-1` model is used for cognitive and ethics auditing, integrated via `python/deepseek_baseline.py`.

## 3. Build, Run, and Test Workflows

Follow these procedures to build, run, and validate changes. Avoid using legacy scripts (`build_all.sh`, `build_unified_linux.sh`).

### Build (CMake)

The standard workflow is a CMake out-of-source build.

```bash
# 1. Configure CMake (from project root)
# For CUDA-enabled builds:
cmake -S . -B build -DQALLOW_ENABLE_CUDA=ON

# For CPU-only builds:
cmake -S . -B build -DQALLOW_ENABLE_CUDA=OFF

# 2. Build the binaries
cmake --build build --parallel
```

Main binaries will be located in the `build/` directory (e.g., `build/qallow`, `build/qallow_unified_cuda`).

### Run

-   **Integrated Run (Recommended)**: Use the `unified` command to run the primary sequence of phases.
    ```bash
    ./build/qallow run unified
    ```
-   **Direct Phase Run**: Execute a specific phase.
    ```bash
    ./build/qallow phase 14 --ticks=600 --nodes=256
    ```

### Test

-   **C/CUDA Tests**: Run `ctest` from the build directory.
    ```bash
    ctest --test-dir build --output-on-failure
    ```
    To run only CUDA-related tests: `ctest -R cuda --test-dir build`.

-   **Python Tests**: Run `pytest` from the root directory.
    ```bash
    pytest
    ```

## 4. Development Patterns & Conventions

-   **Synchronized Backends**: When modifying a data structure or algorithm, ensure changes are reflected in both the C (`backend/cpu`) and CUDA (`backend/cuda`) implementations.
-   **Orchestrator is King**: All high-level quantum workflows **must** go through `python/quantum/orchestrator.py`. Do not directly instantiate quantum bridges.
-   **Testing with Mocks**: Because CUDA-Q is a hard dependency, Python tests that touch the orchestration layer must be runnable in environments without the full CUDA-Q SDK (e.g., CI/CD). Use the `mock_cudaq_backend` fixture from `tests/conftest.py` to patch the `cudaq` import.
    ```python
    # in tests/meta_learning/integration/test_orchestrator.py
    def test_orchestrator_initialization(mock_cudaq_backend):
        # Your test code here...
        # The 'cudaq' module is mocked for this test.
    ```
-   **Telemetry**: Use the `qallow_log_*` macros in C/CUDA for structured logging. Raw `printf` should be avoided outside of the immediate CLI interface layer (`interface/`). Logs are written to `data/logs/`.
-   **Ethics Module**: The core ethics math (`E = S + C + H`) is in `algorithms/ethics_*.c`. Any changes to metrics must be reflected in the `ethics_state_t` struct and its associated data exporters.
-   **Native App (Rust)**: The Rust-based desktop GUI lives in `native_app/`. It is built and run separately with `cargo run`. It communicates with the C/CUDA backend via the `process_manager.rs`.

## 5. Reporting

-   Report back with executed build and test commands to demonstrate validation.
-   Clearly state any impacts on telemetry or the ethics module. Transparency is required for any changes touching the core AGI phases.
