---
applyTo: '**'
---

# Qallow Agent Guide
- Maintain MCP persistent memory for architecture, ethics mandates, and user-specific workflows; refresh context when build scripts or docs change.
- Auto-confirm safe tasks (code, tests, docs, commits, pushes); pause for explicit approval before destructive commands like rm/kill/force-push/system configuration edits.
- Primary code roots: CPU backend in `backend/cpu/`, CUDA kernels in `backend/cuda/`, public APIs in `include/qallow/`, orchestration in `interface/main.c`, ethics engines under `algorithms/`.

## Architecture & Data Flow
- Phase pipeline lives in `interface/main.c`; it wires adaptive ingest → multi-pocket routing → ethics phases (8-10) → quantum bridge (11) → elasticity/harmonics (12-13).
- Shared structs and constants sit in `core/include/`; modify headers plus both backends whenever a phase contract changes.
- Telemetry and logging flow through `include/qallow/logging.h` into `data/logs/`; profiling relies on `QALLOW_PROFILE_SCOPE` from `include/qallow/profiling.h`.
- Python bridge for quantum hardware resides in `python/quantum/run_phase11_bridge.py`; CLI detects interpreters via `QALLOW_PYTHON` and `.env`.
- Optional SDL visualizer targets `interface/qallow_ui.c`; guard changes because it builds only when SDL2 + SDL2_ttf are detected.

## Build & Run
- Standard workflow: `./scripts/build_all.sh [--cpu|--cuda|--auto] [--build-type <cfg>]`; script seeds `build/` and runs `ctest`.
- Manual CMake path: `cmake -S . -B build -DQALLOW_ENABLE_CUDA=ON && cmake --build build --parallel`.
- Makefile shim: `make ACCELERATOR=CPU|CUDA` yields deterministic binaries under `build/CPU/` and `build/CUDA/`.
- Entry binaries: `build/qallow` (phase runner CLI) and `build/qallow_unified`; keep CLI flags in sync with docs and scripts.
- End-to-end demos: `./scripts/run_unified_agi.sh` for the unified pipeline; `python examples/quantum_adaptive_demo.py --runner ./build/qallow_unified` for quantum feedback loops.

## Testing & QA
- Core test suite lives in CTest (`unit_ethics_core`, `unit_dl_integration`, optional `unit_cuda_parallel`); run with `ctest --test-dir build --output-on-failure`.
- Smoke harness `tests/smoke/test_modules.sh` recompiles CPU artifacts and checks phases 12/13 plus governance markers in `build/test_modules.log`.
- CUDA edits require `ctest -R cuda` and Nsight evidence when tuning kernels or performance-critical loops.
- Integration tests under `tests/integration/` assert telemetry structure; update paired CSV fixtures in `data/logs/` when schemas evolve.

## Coding Patterns
- C targets C11, CUDA sources use C++17; keep cross-phase signatures in `core/include/` and `include/qallow/`.
- Always route logging through `qallow_log_*` and load environment defaults with `qallow_env_load`; avoid raw `printf` in backend code.
- Wrap hot loops in `QALLOW_PROFILE_SCOPE("label")`; implementations live in `src/runtime/profiling.cpp`.
- Ethics modules must preserve the Axiom `E = S + C + H`; coordinate changes across `algorithms/ethics_*.c` and propagate metrics types.
- Extending a phase means updating CPU (`backend/cpu/phaseXX_*.c`), CUDA (`backend/cuda/*.cu`), and the orchestration glue in `src/qallow_phase13.c`.

## Integration Notes
- IBM Quantum configuration flows through `.env` keys (`QALLOW_QISKIT`, `_BACKEND`) and the bridge invoked from `interface/main.c`.
- External dependencies: `spdlog` via FetchContent, optional SDL2/SDL2_ttf, CUDA Toolkit ≥ 12; ensure CMake options and docs stay aligned.
- Vendored `mcp-memory-service/` is an upstream MCP provider; avoid edits unless syncing with its own instructions and licensing.
- Telemetry outputs drive dashboards in `docs/` and `data/logs/`; document schema changes in `docs/ARCHITECTURE_SPEC.md` and `README.md`.

## Workflow Reminders
- Follow Conventional Commits and reference issues; run the formatters listed in `CONTRIBUTING.md` (clang-format, cmake-format, markdownlint).
- Prefer scripts in `scripts/` (`build_wrapper.sh`, `run_auto.sh`, `run_latest.sh`) to mirror CI build flags.
- Raise clarifying questions when ethics implications or module ownership is uncertain; err on transparency before modifying critical paths.
- After changes, report which builds/tests were executed and any residual risks so maintainers can validate quickly.

