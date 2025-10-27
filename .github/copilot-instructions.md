---
applyTo: '**'
---

# Qallow Agent Playbook
- Core runtime is C/CUDA: orchestration in `interface/launcher.c` + `interface/main.c`, CPU phases in `backend/cpu/`, GPU mirrors in `backend/cuda/`, shared contracts in `core/include/` and `include/qallow/`. Change a struct once, update both backends and the orchestrator.
- Phase flow: ingest/adaptive → ethics (8–10) → quantum bridge (11) → elasticity/harmonics (12–13) → lattice convergence (14–15). The CLI (`qallow run`) now exposes `--integrate` to execute these sequentially; use overrides like `--integrate-phase13-ticks=400` or `--integrate-ticks=256`.
- Phase 11 runs through `python/quantum/run_phase11_bridge.py`; it expects a Qiskit environment (set `QALLOW_QISKIT=1` and ensure the selected backend exposes `Circuit.num_qubits()`).
- Telemetry funnels through `src/runtime/telemetry_outputs.c` into `data/logs/`. Use `qallow_log_*`/`QALLOW_PROFILE_SCOPE` around hot paths; raw `printf` is reserved for the CLI layer.
- Ethics math (`E = S + C + H`) lives in `algorithms/ethics_*.c`; any metric addition must ripple into `ethics_state_t`, exporters, and the CSV/JSON summaries that dashboards consume.
- Build paths: `./scripts/build_all.sh [--cpu|--cuda]` is the maintainer workflow; manual alternative is `cmake -S . -B build -DQALLOW_ENABLE_CUDA=ON && cmake --build build --parallel`. Scripts such as `build_unified_linux.sh` or `scripts/build_unified_ethics.sh` are legacy—prefer the CMake build.
- Main binaries: `build/qallow` (phase runner CLI) and `build/qallow_unified_cpu|cuda` (packaged run). Rust native app lives in `native_app/` and runs with `cargo run`.
- Native app (Rust/FLTK): entry `native_app/src/main.rs`; UI under `native_app/src/ui/**`. Use FLTK `app::channel` + `UiMessage` (`native_app/src/messaging.rs`) for non-blocking tasks; start work in threads from `button_handlers.rs` (`start_*_async`) and handle results in the main loop. Process launching is in `native_app/src/backend/process_manager.rs`.
- Typical end-to-end run: `./build/qallow run unified` (defaults to phases 12–15 with ticks=120/120 and lattice ticks=64). Add overrides like `--integrate-phase13-k=0.003` after `unified` when needed. Phase 11 joins automatically once `QALLOW_QISKIT=1` is set and the bridge matches the active Qiskit release.
 - Direct phase runs: `./build/qallow phase 13|14|15 --ticks=...` (phase "1" is invalid). Use `--integrate-*` flags only with the integrated runner.
- Testing: run `ctest --test-dir build --output-on-failure` after C/CUDA changes. Use `tests/sequential_phase_benchmark.sh` to time phases 1–13 and regenerate `data/logs/sequential_benchmark.csv`. CUDA edits should also run `ctest -R cuda` plus Nsight profiling if performance is touched.
- Docs & conventions: follow the patterns in `README.md` and `docs/ARCHITECTURE_SPEC.md`. Logging schema changes or CLI flag updates must be reflected there and in `scripts/` wrappers.
- External surfaces: `mcp-memory-service/` is vendored—avoid modifications unless coordinating upstream. SDL-based UI (`interface/qallow_ui.c`) builds only when SDL2 + SDL2_ttf are detected; guard code with compile-time checks.
- **MCP Memory Server Integration**: GitHub Copilot integrates with a persistent memory MCP server via `.vscode/mcp.json`. The memory server (SQLite-vec backend) runs locally and provides semantic search, memory storage, and recall tools. Configure in VS Code: open Copilot Chat, select Agent mode, click the tools icon, and the memory server tools will appear. Use `/mcp.memory.*` commands in chat to store/recall context. The server persists memories in `/root/.local/share/mcp-memory/` and supports multi-session context awareness.
- Report back with executed build/test commands and any ethics/telemetry impacts; transparency is expected for changes touching the AGI phases.
