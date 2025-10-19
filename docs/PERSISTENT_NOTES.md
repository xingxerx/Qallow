# Persistent Notes for Qallow Sessions

Use this file to capture decisions, configuration details, or context that you want to survive across assistant sessions. When a new conversation starts, paste the relevant section so the assistant can pick up where it left off.

## Template Entry

```
## [YYYY-MM-DD] [Topic]
- Summary: …
- Key files: …
- Pending actions: …
- Validation steps: …
```

Add new sections as work evolves. Keep the notes concise; one screenful is easier to reload.

## 2025-03-18 Session Notes
- Summary: Added real-time pocket telemetry (memory usage) with JSON output, updated Tk monitor to display metrics, cleaned CUDA build script (nvcc link, object cleanup), and created persistent notes workflow with reload script.
- Key files: backend/cpu/pocket.c, core/include/pocket.h, interface/main.c, ui/qallow_monitor.py, scripts/build_unified_cuda.sh, docs/PERSISTENT_NOTES.md, scripts/reload_context.sh.
- Pending actions: Generate C++/CUDA Phase 12 and Rust Phase 13 templates per new architecture plan; craft Python orchestration script once templates exist.
- Validation steps: Run `./scripts/build_unified_cuda.sh`, then `build/qallow_unified_cuda` to verify telemetry JSON updates; launch `python3 ui/qallow_monitor.py` (with Tk/GUI access) to confirm live metrics; rerun `./scripts/reload_context.sh` to ensure notes load for next session.
