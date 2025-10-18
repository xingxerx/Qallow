# Phase-13 Accelerator Roadmap

## Overview
With all Bend artifacts removed and the threaded/shared-memory Phase-13 accelerator in place, the next objective is to integrate it cleanly with the existing CUDA-enabled Qallow VM. This roadmap breaks the work into iterative milestones so we can ship a stable accelerator-first experience while preserving the legacy VM pipeline.

## Milestones

### 1. Modularize & Surface the Accelerator (Completed)
- [x] Expose a reusable C entry point for the Phase-13 accelerator so it can be driven from other binaries.
- [x] Provide a public header describing the accelerator config and return codes.
- [x] Harden CLI parsing (threads/watch/file) for reuse by both the standalone binary and launcher integration.

### 2. Integrate with Qallow Launcher (Planned)
- [ ] Add an `accelerator` (or similar) mode to `interface/launcher.c` that dispatches into the Phase-13 engine.
- [ ] Extend build scripts to optionally embed the accelerator object file when building the unified CUDA binary.
- [ ] Gate accelerator launch behind a capability check (inotify availability, shm permissions) with clear diagnostics.

### 3. Telemetry & Observability (Planned)
- [ ] Emit structured telemetry (cache hits/misses, worker utilization, watcher events) via the existing CSV/telemetry subsystems.
- [ ] Add throttled logging for key lifecycle events (cache attach, watcher start/stop, worker shutdown).
- [ ] Build a lightweight diagnostic command (`qallow accelerator --status`) to inspect shared-memory state.

### 4. Validation & Tooling (Planned)
- [ ] Provide self-test scripts to simulate file events and assert cache behaviour.
- [ ] Document accelerator usage in the main README and quickstart docs.
- [ ] Integrate accelerator runs into CI smoke tests once sandboxed shm/inotify access is available.

## Immediate Next Steps
1. Finish the modularization effort (Step 1) and land the new header/API.
2. Wire the launcher `accelerator` mode while keeping the standalone binary build path intact.
3. Iterate on telemetry hooks and documentation after integration tests pass.
