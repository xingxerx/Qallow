# GitHub Copilot + Persistent Memory MCP Server Integration

## ✅ Setup Complete

The Qallow project now has a fully integrated persistent memory MCP server for GitHub Copilot. This enables Copilot to maintain context across sessions using semantic memory storage.

## What Was Added

### 1. **Configuration Files**
- **`.vscode/mcp.json`** - MCP server configuration for GitHub Copilot
  - Uses `servers` format (GitHub Copilot compatible)
  - Configured with SQLite-vec backend
  - Storage path: `~/.local/share/mcp-memory/`

### 2. **Documentation**
- **`.github/copilot-instructions.md`** - Updated with MCP memory server integration details
- **`.github/MCP_COPILOT_SETUP.md`** - Comprehensive setup and configuration guide
- **`.github/MCP_MEMORY_QUICK_REFERENCE.md`** - Quick reference for common memory commands
- **`.github/verify-mcp-setup.sh`** - Verification script to test the setup

### 3. **Verification**
All components verified and working:
- ✅ `.vscode/mcp.json` configuration valid
- ✅ MCP Memory Service module installed
- ✅ Python virtual environment ready
- ✅ Storage directory configured and writable
- ✅ SQLite-vec backend enabled

## How to Use

### Quick Start (30 seconds)

1. **Open Copilot Chat** in VS Code
   - Windows/Linux: `Ctrl+Shift+I`
   - Mac: `Cmd+Shift+I`

2. **Select Agent Mode**
   - Click the dropdown and select "Agent"

3. **Enable Memory Tools**
   - Click the tools icon (⚙️) in the top-left
   - Memory server tools now available

4. **Start Using Memory**
   - "Remember that [important context]"
   - "What do you remember about [topic]?"
   - "Find memories related to [query]"

### Example Usage

```
You: Remember that Qallow uses a three-layer architecture:
     1. C/CUDA core (interface/launcher.c, backend/cpu/, backend/cuda/)
     2. Python quantum bridge (python/quantum/run_phase11_bridge.py)
     3. Rust native UI (native_app/ with FLTK)

Copilot: ✓ Memory stored successfully

You: What do you remember about the architecture?

Copilot: [Recalls the three-layer architecture with full details]
```

## Key Features

### Persistent Storage
- Memories survive VS Code restarts
- Stored in `~/.local/share/mcp-memory/`
- SQLite-vec vector database for fast semantic search

### Semantic Search
- Find memories by meaning, not just keywords
- "quantum" finds memories about "Qiskit"
- "UI" finds memories about "FLTK"
- Sub-millisecond search performance

### Multi-Session Context
- Store context once, use across multiple sessions
- Perfect for long-running projects
- Ideal for team collaboration

### No External Dependencies
- All storage is local
- No data sent to external services
- No API keys needed

## Architecture

```
GitHub Copilot Chat
        ↓
    Agent Mode
        ↓
    MCP Tools
        ↓
MCP Memory Server (.vscode/mcp.json)
        ↓
SQLite-vec Database (~/.local/share/mcp-memory/)
```

## Configuration Details

### MCP Server Configuration
```json
{
  "servers": {
    "memory": {
      "command": "/root/Qallow/mcp-memory-service/.venv/bin/python",
      "args": ["-m", "src.mcp_memory_service.server"],
      "cwd": "/root/Qallow/mcp-memory-service",
      "env": {
        "MCP_MEMORY_STORAGE_BACKEND": "sqlite_vec",
        "MCP_MEMORY_SQLITE_VEC_PATH": "/root/.local/share/mcp-memory",
        "PYTHONPATH": "/root/Qallow/mcp-memory-service/src",
        "LOG_LEVEL": "INFO"
      }
    }
  }
}
```

### Storage Backend
- **Type**: SQLite-vec (vector database)
- **Location**: `~/.local/share/mcp-memory/`
- **Persistence**: Automatic
- **Search**: Semantic (vector similarity)

## Verification

Run the verification script to ensure everything is set up correctly:

```bash
.github/verify-mcp-setup.sh
```

Expected output:
```
✓ All checks passed!
```

## Documentation

| Document | Purpose |
|----------|---------|
| `.github/copilot-instructions.md` | Qallow Agent Playbook with MCP integration |
| `.github/MCP_COPILOT_SETUP.md` | Full setup and configuration guide |
| `.github/MCP_MEMORY_QUICK_REFERENCE.md` | Quick reference for memory commands |
| `.github/verify-mcp-setup.sh` | Verification script |

## Common Use Cases

### 1. Store Architecture Decisions
```
Remember that we use [pattern] for [component]
```

### 2. Track Phase Changes
```
Remember the ethics formula: E = S + C + H
```

### 3. Document Build Commands
```
Remember: ./scripts/build_all.sh [--cpu|--cuda]
```

### 4. Recall During Development
```
What do you remember about the phase orchestration?
```

## Troubleshooting

### Memory tools not appearing?
1. Reload VS Code: `Cmd+Shift+P` → "Reload Window"
2. Check `.vscode/mcp.json` syntax
3. Verify Python path exists

### Memories not persisting?
1. Check storage directory: `ls -la ~/.local/share/mcp-memory/`
2. Verify write permissions: `chmod u+w ~/.local/share/mcp-memory`
3. Restart VS Code

### Performance issues?
1. List and delete old memories
2. Check database size: `ls -lh ~/.local/share/mcp-memory/`
3. Rebuild index if needed

## Security & Privacy

✅ **Local Storage**: All memories stored locally in `~/.local/share/mcp-memory/`
✅ **No External Services**: No data sent to external APIs
✅ **No API Keys**: Never store API keys in memories (use `.env` files)
⚠️ **Encryption**: Not encrypted by default (use OS-level encryption if needed)

## Next Steps

1. **Verify Setup**: Run `.github/verify-mcp-setup.sh`
2. **Read Quick Reference**: `.github/MCP_MEMORY_QUICK_REFERENCE.md`
3. **Start Using Memory**: Open Copilot Chat and try storing context
4. **Explore Features**: Use semantic search to find related memories

## Support

For issues or questions:
1. Check `.github/MCP_COPILOT_SETUP.md` troubleshooting section
2. Review `.github/MCP_MEMORY_QUICK_REFERENCE.md` for examples
3. Run `.github/verify-mcp-setup.sh` to diagnose issues

## References

- [GitHub Copilot MCP Documentation](https://docs.github.com/copilot/customizing-copilot/using-model-context-protocol/extending-copilot-chat-with-mcp)
- [MCP Protocol Specification](https://modelcontextprotocol.io/)
- [SQLite-vec Documentation](https://github.com/asg017/sqlite-vec)
- [Qallow Agent Playbook](.github/copilot-instructions.md)

---

**Status**: ✅ Ready to use
**Last Updated**: 2025-10-27
**Verification**: All checks passed

