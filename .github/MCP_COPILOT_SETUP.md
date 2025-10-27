# GitHub Copilot + Persistent Memory MCP Server Setup

This guide explains how to use the persistent memory MCP server with GitHub Copilot in VS Code.

## Overview

The Qallow project includes a **persistent memory MCP (Model Context Protocol) server** that integrates with GitHub Copilot. This server provides:

- **Semantic Memory Storage**: Store and retrieve context using vector embeddings
- **Multi-Session Persistence**: Memories persist across Copilot sessions
- **SQLite-vec Backend**: Local, fast vector database with no external dependencies
- **Natural Language Search**: Find relevant memories using semantic similarity

## Prerequisites

1. **VS Code 1.99+** with GitHub Copilot extension installed
2. **Python 3.9+** with virtual environment support
3. **MCP Memory Service** already configured in `.vscode/mcp.json`

## Quick Start

### 1. Verify MCP Configuration

Check that `.vscode/mcp.json` exists and contains the memory server configuration:

```json
{
  "mcpServers": {
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

### 2. Start Using Memory in Copilot Chat

1. Open **Copilot Chat** in VS Code (Ctrl+Shift+I or Cmd+Shift+I)
2. Select **Agent** mode from the dropdown
3. Click the **tools icon** (⚙️) in the top-left of the chat box
4. You should see **Memory Server** tools available

### 3. Use Memory Commands

In Copilot Chat, you can use memory commands:

- **Store Memory**: "Remember that [context]"
- **Recall Memory**: "What do you remember about [topic]?"
- **Search Memory**: "Find memories related to [query]"

## Memory Server Architecture

### Storage Backend: SQLite-vec

- **Location**: `~/.local/share/mcp-memory/`
- **Type**: Vector database with semantic search
- **Persistence**: Automatic, survives VS Code restarts
- **Performance**: Sub-millisecond semantic search

### MCP Tools Available

The memory server exposes these tools to Copilot:

- `store_memory`: Save context with semantic embeddings
- `recall_memory`: Retrieve memories by semantic similarity
- `search_memory`: Full-text and semantic search
- `list_memories`: View all stored memories
- `delete_memory`: Remove specific memories

## Troubleshooting

### Memory Server Not Appearing

1. Reload VS Code window (Cmd+Shift+P → "Developer: Reload Window")
2. Check `.vscode/mcp.json` syntax is valid JSON
3. Verify Python path exists: `/root/Qallow/mcp-memory-service/.venv/bin/python`
4. Check VS Code output panel for MCP errors

### Memories Not Persisting

1. Verify storage path exists: `mkdir -p ~/.local/share/mcp-memory`
2. Check file permissions: `ls -la ~/.local/share/mcp-memory/`
3. Restart the MCP server: Reload VS Code window

### Performance Issues

1. Clear old memories: Use "list_memories" and delete unused ones
2. Check database size: `ls -lh ~/.local/share/mcp-memory/`
3. Rebuild index if needed (automatic on startup)

## Advanced Configuration

### Custom Storage Path

Edit `.vscode/mcp.json` and change `MCP_MEMORY_SQLITE_VEC_PATH`:

```json
"env": {
  "MCP_MEMORY_SQLITE_VEC_PATH": "/custom/path/to/memory"
}
```

### Enable Debug Logging

Set `LOG_LEVEL` to `DEBUG`:

```json
"env": {
  "LOG_LEVEL": "DEBUG"
}
```

## Integration with Qallow Development

When developing Qallow:

1. **Store Architecture Decisions**: "Remember that we use [pattern] for [component]"
2. **Track Phase Changes**: "Remember the ethics formula: E = S + C + H"
3. **Document Patterns**: "Remember that phase orchestration uses [approach]"
4. **Recall Context**: Ask Copilot to recall relevant memories during coding

## Security Notes

- Memories are stored **locally** in `~/.local/share/mcp-memory/`
- No data is sent to external services
- API keys should be stored in `.env` files (never in memories)
- Memories are not encrypted by default (use OS-level encryption if needed)

## Further Reading

- [GitHub Copilot MCP Documentation](https://docs.github.com/copilot/customizing-copilot/using-model-context-protocol/extending-copilot-chat-with-mcp)
- [MCP Protocol Specification](https://modelcontextprotocol.io/)
- [SQLite-vec Documentation](https://github.com/asg017/sqlite-vec)

