# GitHub Copilot + MCP Memory Server - Quick Reference

## Enable Memory in Copilot Chat

1. Open Copilot Chat: `Ctrl+Shift+I` (Windows/Linux) or `Cmd+Shift+I` (Mac)
2. Select **Agent** mode from dropdown
3. Click **tools icon** (⚙️) in top-left
4. Memory server tools now available

## Common Memory Commands

### Store Context
```
Remember that [important information about the project/code]
```

**Examples:**
- "Remember that Qallow phases 11-15 handle quantum bridge and lattice convergence"
- "Remember that ethics formula is E = S + C + H (Sustainability + Compliance + Harmony)"
- "Remember that the native app uses FLTK for UI and Rust for backend"

### Recall Context
```
What do you remember about [topic]?
```

**Examples:**
- "What do you remember about the phase orchestration?"
- "What do you remember about the ethics metrics?"
- "What do you remember about the build system?"

### Search Memories
```
Find memories related to [query]
```

**Examples:**
- "Find memories related to CUDA optimization"
- "Find memories related to telemetry"
- "Find memories related to the native app architecture"

## Memory Storage Details

| Property | Value |
|----------|-------|
| **Backend** | SQLite-vec (vector database) |
| **Storage Path** | `~/.local/share/mcp-memory/` |
| **Persistence** | Automatic, survives restarts |
| **Search Type** | Semantic (vector similarity) |
| **Performance** | Sub-millisecond queries |

## Useful Memories to Store

### Architecture Decisions
```
Remember that Qallow uses a three-layer architecture:
1. C/CUDA core runtime (interface/launcher.c, backend/cpu/, backend/cuda/)
2. Python quantum bridge (python/quantum/run_phase11_bridge.py)
3. Rust native UI (native_app/ with FLTK)
```

### Build Commands
```
Remember these build commands:
- Full build: ./scripts/build_all.sh [--cpu|--cuda]
- CMake: cmake -S . -B build -DQALLOW_ENABLE_CUDA=ON && cmake --build build --parallel
- Native app: cargo run from native_app/
- Tests: ctest --test-dir build --output-on-failure
```

### Phase Information
```
Remember the Qallow phase flow:
- Phases 1-7: Ingest and adaptive processing
- Phases 8-10: Ethics evaluation (E = S + C + H)
- Phase 11: Quantum bridge (Qiskit)
- Phases 12-13: Elasticity and harmonics
- Phases 14-15: Lattice convergence
```

### Development Patterns
```
Remember FLTK UI patterns in native_app/:
- Use app::channel + UiMessage for non-blocking tasks
- Start async work in button_handlers.rs (start_*_async functions)
- Handle results in main event loop
- Process launching via process_manager.rs
```

## Tips & Tricks

### Multi-Session Context
Memories persist across VS Code sessions. Store important context once, recall it later.

### Semantic Search
The memory server uses vector embeddings. Similar concepts are found even with different wording:
- "quantum" finds memories about "Qiskit"
- "UI" finds memories about "FLTK"
- "compilation" finds memories about "build"

### Memory Organization
Store memories with clear context:
- ✅ "Remember that phase 11 requires QALLOW_QISKIT=1 environment variable"
- ❌ "Remember phase 11"

### Clearing Old Memories
If storage grows large, ask Copilot to list and delete old memories:
```
List all memories and delete the ones older than [date]
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Memory tools not showing | Reload VS Code (Cmd+Shift+P → "Reload Window") |
| Memories not persisting | Check `~/.local/share/mcp-memory/` exists and has write permissions |
| Slow semantic search | Clear old memories or rebuild index |
| MCP server errors | Check `.vscode/mcp.json` syntax and Python path |

## Integration with Qallow Development

### When Starting a New Feature
```
Remember that I'm working on [feature name] which involves [components].
The relevant files are [list files].
```

### When Debugging
```
Remember the error: [error message]
Remember the stack trace: [trace]
Remember what I've tried: [attempts]
```

### When Refactoring
```
Remember the old implementation pattern: [pattern]
Remember the new implementation pattern: [pattern]
Remember the migration steps: [steps]
```

## Security & Privacy

- ✅ Memories stored locally in `~/.local/share/mcp-memory/`
- ✅ No data sent to external services
- ✅ No API keys in memories (use `.env` files)
- ⚠️ Memories not encrypted by default (use OS encryption if needed)

## Advanced: Custom Storage Path

Edit `.vscode/mcp.json` to change storage location:

```json
"env": {
  "MCP_MEMORY_SQLITE_VEC_PATH": "/custom/path"
}
```

Then reload VS Code.

## See Also

- [Full MCP Setup Guide](.github/MCP_COPILOT_SETUP.md)
- [GitHub Copilot MCP Docs](https://docs.github.com/copilot/customizing-copilot/using-model-context-protocol/extending-copilot-chat-with-mcp)
- [Qallow Agent Playbook](.github/copilot-instructions.md)

